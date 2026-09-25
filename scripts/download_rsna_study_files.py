"""Restartable downloader for the RSNA first-study DICOM files.

Downloads only the files named in first_study_files.txt, one at a time, through the
Kaggle CLI. Each file is staged in its own scratch directory, validated, then moved
into place atomically, so a final path never holds a partial file and a rerun skips
everything already completed.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

COMPETITION = "rsna-knee-abnormality-detection"
UID = r"[0-9]+(?:\.[0-9]+)*"
FILE_PATTERN = re.compile(rf"^train_series/({UID})/({UID})/({UID}\.dcm)$")
TRANSIENT_MARKERS = ("remote end closed", "connection aborted", "connection reset", "timed out",
                     "timeout", "temporarily", "try again", "429", "500", "502", "503", "504")


class TransientError(RuntimeError):
    """Worth retrying: network trouble, or a download that did not validate."""


class FatalError(RuntimeError):
    """Retrying will not help: missing CLI, authentication, rules not accepted, bad input."""


def read_file_list(path: Path):
    if not path.is_file():
        raise FatalError(f"Input list not found: {path}")
    names = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        name = line.strip()
        if not name:
            continue
        if not FILE_PATTERN.match(name):
            raise FatalError(f"{path}:{number}: unexpected file name {name!r}")
        names.append(name)
    return list(dict.fromkeys(names))


def destination_for(name: str, output_root: Path) -> Path:
    return output_root.joinpath(*name.split("/"))


def already_present(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def run_kaggle_download(name: str, staging: Path, timeout: int = 600):
    command = ["kaggle", "competitions", "download", COMPETITION, "-f", name, "-p", str(staging), "-q"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise FatalError("Kaggle CLI was not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise TransientError(f"Kaggle CLI timed out after {timeout}s") from exc
    if result.returncode != 0:
        diagnostics = (result.stderr or result.stdout).strip()
        message = f"Kaggle CLI failed (exit {result.returncode}): {diagnostics[:1000]}"
        if any(marker in diagnostics.lower() for marker in TRANSIENT_MARKERS):
            raise TransientError(message)
        raise FatalError(message)


def locate_download(staging: Path, expected_name: str) -> Path:
    files = [p for p in staging.rglob("*") if p.is_file()]
    exact = [p for p in files if p.name == expected_name]
    if exact:
        return exact[0]
    archives = [p for p in files if p.suffix.lower() == ".zip"]
    if archives:
        try:
            with zipfile.ZipFile(archives[0]) as archive:
                members = [m for m in archive.namelist() if Path(m).name == expected_name]
                if len(members) != 1:
                    raise TransientError(f"Archive did not contain exactly one {expected_name}")
                target = staging / f"{expected_name}.extracted"
                with archive.open(members[0]) as source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out)
                return target
        except zipfile.BadZipFile as exc:
            raise TransientError("Downloaded archive is corrupt") from exc
    found = ", ".join(p.name for p in files) or "nothing"
    raise TransientError(f"Expected {expected_name} after download, found {found}")


def validate_dicom(path: Path):
    if path.stat().st_size == 0:
        raise TransientError("Downloaded file is empty")
    try:
        import pydicom
    except ImportError:
        return
    try:
        pydicom.dcmread(path, stop_before_pixels=True)
    except Exception as exc:
        raise TransientError(f"Downloaded file is not a readable DICOM: {exc}") from exc


def fetch_one(name: str, destination: Path, staging_root: Path, retries: int, backoff: float) -> int:
    staging = staging_root / destination.stem
    for attempt in range(1, retries + 1):
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            run_kaggle_download(name, staging)
            candidate = locate_download(staging, destination.name)
            validate_dicom(candidate)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(candidate, destination)
            return destination.stat().st_size
        except TransientError as exc:
            if attempt == retries:
                raise
            delay = backoff * 2 ** (attempt - 1)
            print(f"    transient failure ({exc}); retry {attempt}/{retries - 1} in {delay:.0f}s",
                  file=sys.stderr, flush=True)
            time.sleep(delay)
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/rsna-knee/first_study_files.txt"))
    parser.add_argument("--output-root", type=Path, default=Path("data/rsna-knee/raw"))
    parser.add_argument("--dry-run", action="store_true", help="list what would be downloaded; touch nothing")
    parser.add_argument("--limit", type=positive_int, help="download at most N not-yet-present files")
    parser.add_argument("--retries", type=positive_int, default=5, help="attempts per file (default 5)")
    parser.add_argument("--backoff", type=float, default=2.0, help="first retry delay in seconds; doubles each retry")
    args = parser.parse_args(argv)

    try:
        names = read_file_list(args.input)
    except FatalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    present = [n for n in names if already_present(destination_for(n, args.output_root))]
    present_set = set(present)
    pending = [n for n in names if n not in present_set]
    selected = pending[:args.limit] if args.limit else pending
    print(f"input: {args.input} ({len(names)} files)")
    print(f"output root: {args.output_root}")
    print(f"already present (non-empty): {len(present)}; pending: {len(pending)}; "
          f"selected this run: {len(selected)}{' [dry run]' if args.dry_run else ''}")

    if args.dry_run:
        for index, name in enumerate(selected, 1):
            print(f"[{index}/{len(selected)}] would download {name}\n      -> {destination_for(name, args.output_root)}")
        return 0

    staging_root = args.output_root / ".partial"
    downloaded, failed, total_bytes = 0, [], 0
    started = time.monotonic()
    try:
        for index, name in enumerate(selected, 1):
            destination = destination_for(name, args.output_root)
            print(f"[{index}/{len(selected)}] downloading {destination.name}", flush=True)
            try:
                size = fetch_one(name, destination, staging_root, args.retries, args.backoff)
            except TransientError as exc:
                print(f"    FAILED after {args.retries} attempts: {exc}", file=sys.stderr, flush=True)
                failed.append(name)
                continue
            downloaded += 1
            total_bytes += size
            print(f"    ok ({size} bytes)", flush=True)
    except FatalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; completed files are kept, rerun to resume.", file=sys.stderr)
        return 130
    finally:
        try:
            staging_root.rmdir()
        except OSError:
            pass

    print("summary:")
    print(f"  listed: {len(names)}")
    print(f"  skipped (already present): {len(present)}")
    print(f"  downloaded this run: {downloaded} ({total_bytes} bytes)")
    print(f"  failed: {len(failed)}")
    for name in failed:
        print(f"    {name}")
    print(f"  not attempted (pending beyond --limit): {len(pending) - len(selected)}")
    print(f"  elapsed: {time.monotonic() - started:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
