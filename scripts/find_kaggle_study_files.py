"""List one RSNA Kaggle study's competition files without downloading them.

This is an intentionally small, temporary inspection utility. It invokes the
installed Kaggle CLI, follows page tokens, and writes only matching filenames.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


COMPETITION = "rsna-knee-abnormality-detection"
STUDY_UID = "1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260"
PAGE_SIZE = 200
EXPECTED_PREFIX = f"train_series/{STUDY_UID}/"
NEXT_TOKEN = re.compile(r"Next Page Token\s*=\s*(\S+)")


def parse_page(stdout: str, stderr: str):
    combined = f"{stdout}\n{stderr}"
    token_match = NEXT_TOKEN.search(combined)
    # The CLI prints the token before the JSON array. Decode from the first '['
    # so diagnostic/token text cannot be mistaken for a file record.
    start = combined.find("[")
    if start < 0:
        raise RuntimeError(f"Kaggle CLI returned no JSON file page:\n{combined[:1000]}")
    try:
        payload, _ = json.JSONDecoder().raw_decode(combined[start:])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not decode Kaggle file page: {exc}\n{combined[:1000]}") from exc
    if not isinstance(payload, list):
        raise RuntimeError("Kaggle file page JSON was not a list")
    return payload, token_match.group(1) if token_match else None


def fetch_page(token: str | None, retries: int = 3):
    command = ["kaggle", "competitions", "files", COMPETITION,
               "--page-size", str(PAGE_SIZE), "--format", "json", "--quiet"]
    if token:
        command.extend(["--page-token", token])
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        except FileNotFoundError as exc:
            raise RuntimeError("Kaggle CLI was not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            if attempt == retries:
                raise RuntimeError("Kaggle CLI request timed out") from exc
            print(f"  transient timeout; retry {attempt}/{retries - 1}", file=sys.stderr, flush=True)
            time.sleep(attempt * 2)
            continue
        if result.returncode == 0:
            try:
                return parse_page(result.stdout, result.stderr)
            except RuntimeError:
                if attempt == retries:
                    raise
        else:
            diagnostics = (result.stderr or result.stdout).strip()
            transient = any(text in diagnostics.lower() for text in
                            ("remote end closed", "connection aborted", "timed out", "temporarily"))
            if not transient or attempt == retries:
                raise RuntimeError(f"Kaggle CLI failed (exit {result.returncode}); check authentication/network access.\n{diagnostics[:2000]}")
            print(f"  transient Kaggle error; retry {attempt}/{retries - 1}", file=sys.stderr, flush=True)
        time.sleep(attempt * 2)
    raise RuntimeError("Kaggle page request failed after retries")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/rsna-knee/first_study_files.txt"))
    args = parser.parse_args(argv)
    matches = []
    sizes = {}
    token = None
    page = 0
    try:
        while True:
            page += 1
            records, token = fetch_page(token)
            for record in records:
                name = str(record.get("name", ""))
                if name.startswith(EXPECTED_PREFIX):
                    matches.append(name)
                    sizes[name] = int(record.get("size", 0))
            print(f"page {page}: received {len(records)} files; matching total {len(matches)}", flush=True)
            if not token:
                break
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # The API should not duplicate records across pages; de-duplicate defensively
    # while preserving first-seen order in case a service retries a page.
    unique_matches = list(dict.fromkeys(matches))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(unique_matches) + ("\n" if unique_matches else ""), encoding="utf-8")
    by_series = Counter()
    for name in unique_matches:
        relative = name[len(EXPECTED_PREFIX):]
        series_uid = relative.split("/", 1)[0] if "/" in relative else "<unparsed>"
        by_series[series_uid] += 1
    print(f"pagination complete after {page} page(s)")
    print(f"matching DICOM files: {len(unique_matches)}")
    print("count by SeriesInstanceUID:")
    for series_uid, count in sorted(by_series.items()):
        print(f"  {series_uid}: {count}")
    print(f"total estimated bytes: {sum(sizes.get(name, 0) for name in unique_matches)}")
    print(f"saved filenames: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
