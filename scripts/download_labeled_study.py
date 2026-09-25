"""Download DICOM files for a specific labeled study using exact file discovery.

Uses kaggle competitions files to list exact .dcm paths, groups by series,
verifies against metadata, and downloads each file individually via Kaggle CLI.

Features:
- Discovers exact DICOM file paths via Kaggle CLI pagination
- Validates discovered series match metadata
- Downloads individual files with retry logic
- Validates DICOM before final placement
- Staged writes (temp -> destination)
- Resumable (skips valid files)
- Transient error retry with exponential backoff

Usage:
  python scripts/download_labeled_study.py <study_uid> [--dry-run]

Example:
  python scripts/download_labeled_study.py 1.2.826.0.1.3680043.8.498.10095687747295410396510538520594649149
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_simple_metadata(data_dir: Path):
    """Load train_series.csv without heavy dependencies."""
    root = Path(data_dir)
    train_series = pd.read_csv(
        root / "train_series.csv",
        dtype={"StudyInstanceUID": "string", "SeriesInstanceUID": "string"}
    )

    class SimpleMetadata:
        def __init__(self, series_df):
            self.train_series = series_df

        def series_for(self, study_uid: str, split: str = "train"):
            if split != "train":
                raise ValueError("Only train split supported")
            return self.train_series[self.train_series["StudyInstanceUID"] == str(study_uid)].copy()

    return SimpleMetadata(train_series)


COMPETITION = "rsna-knee-abnormality-detection"
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]  # Exponential backoff in seconds


def is_valid_dicom(path: Path) -> bool:
    """Validate DICOM by checking magic bytes at offset 128."""
    if not path.is_file() or path.stat().st_size < 132:
        return False
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except:
        return False


def discover_study_files(study_uid: str) -> dict:
    """Discover all DICOM files for a study by querying Kaggle file listing.

    Returns: {
        'study_uid': study_uid,
        'total_files': int,
        'by_series': {series_uid: [file_paths]},
        'series_order': [series_uids],
        'success': bool,
        'error': str or None
    }
    """

    result = {
        'study_uid': study_uid,
        'total_files': 0,
        'by_series': defaultdict(list),
        'series_order': [],
        'success': False,
        'error': None,
    }

    print(f"Discovering files for study {study_uid[-12:]}...", flush=True)

    cmd = ["kaggle", "competitions", "files", COMPETITION, "-v", "--page-size", "100"]

    page_token = None
    pages_searched = 0

    while pages_searched < 300:  # Safety limit
        try:
            if page_token:
                cmd_with_token = cmd + ["--page-token", page_token]
            else:
                cmd_with_token = cmd

            result_proc = subprocess.run(
                cmd_with_token,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result_proc.returncode != 0:
                error_msg = result_proc.stderr or result_proc.stdout
                # Check for rate limiting - if so, add delay and retry
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    if pages_searched > 0:  # Only retry if we made progress
                        print(f"  Rate limited, waiting 10s before retry...", flush=True)
                        time.sleep(10)
                        continue
                result['error'] = f"Kaggle CLI error: {error_msg[:200]}"
                return result

            lines = result_proc.stdout.strip().split('\n')
            pages_searched += 1

            # Rate limiting: add delay between requests to avoid 429
            if pages_searched % 20 == 0:
                print(f"  Discovered in {pages_searched} pages, continuing...", flush=True)
                time.sleep(1)

            # Extract next page token
            next_token = None
            for line in lines:
                if line.startswith("Next Page Token = "):
                    next_token = line.replace("Next Page Token = ", "").strip()
                    break

            # Collect files for this study
            for line in lines:
                if study_uid in line and ".dcm" in line and "train_series" in line:
                    parts = line.split(',')
                    if len(parts) >= 1:
                        file_path = parts[0].strip()

                        # Extract series UID from path
                        path_parts = file_path.split('/')
                        if len(path_parts) >= 3:
                            series_uid = path_parts[2]
                            result['by_series'][series_uid].append(file_path)

            if not next_token:
                break

            page_token = next_token

        except subprocess.TimeoutExpired:
            result['error'] = "Kaggle CLI timeout"
            return result
        except Exception as e:
            result['error'] = f"Discovery failed: {e}"
            return result

    if not result['by_series']:
        result['error'] = "No files found for this study"
        return result

    # Sort series by first appearance
    result['series_order'] = list(result['by_series'].keys())
    result['total_files'] = sum(len(files) for files in result['by_series'].values())
    result['success'] = True

    return result


def verify_against_metadata(study_uid: str, discovered_series: list, metadata) -> dict:
    """Verify discovered series UIDs match metadata.

    Returns: {
        'valid': bool,
        'message': str,
        'metadata_series': [uids],
        'discovered_series': [uids],
        'missing': [uids],
        'extra': [uids],
    }
    """

    series_rows = metadata.series_for(study_uid, "train")
    metadata_series = sorted(series_rows.SeriesInstanceUID.astype(str).tolist())
    discovered_sorted = sorted([str(s) for s in discovered_series])

    missing = set(metadata_series) - set(discovered_sorted)
    extra = set(discovered_sorted) - set(metadata_series)

    valid = len(missing) == 0 and len(extra) == 0

    if valid:
        message = f"All {len(metadata_series)} metadata series found"
    else:
        parts = []
        if missing:
            parts.append(f"{len(missing)} missing: {','.join(list(missing)[:2])}")
        if extra:
            parts.append(f"{len(extra)} extra")
        message = " + ".join(parts) if parts else "Mismatch"

    return {
        'valid': valid,
        'message': message,
        'metadata_series': metadata_series,
        'discovered_series': discovered_sorted,
        'missing': list(missing),
        'extra': list(extra),
    }


def download_file_with_retry(file_path: str, destination: Path) -> tuple[bool, str]:
    """Download one DICOM file with retry logic.

    Returns: (success: bool, error_msg: str or None)
    """

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with tempfile.TemporaryDirectory(prefix="dcm_") as staging_dir:
                staging = Path(staging_dir)

                cmd = [
                    "kaggle", "competitions", "download",
                    COMPETITION,
                    "-f", file_path,
                    "-p", str(staging),
                    "-q"
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode != 0:
                    error_msg = (result.stderr or result.stdout).strip()[:100]

                    # Classify error as transient or permanent
                    is_transient = any(x in error_msg.lower() for x in [
                        "timeout", "connection", "temporarily", "429", "503", "502"
                    ])

                    if is_transient and attempt < MAX_RETRIES:
                        wait = RETRY_BACKOFF[attempt - 1]
                        time.sleep(wait)
                        continue
                    else:
                        return False, error_msg

                # Find downloaded file
                dcm_files = list(staging.rglob("*.dcm"))
                if not dcm_files:
                    return False, "No .dcm file in download"

                downloaded = dcm_files[0]

                # Validate
                if not is_valid_dicom(downloaded):
                    return False, "Downloaded file is not valid DICOM"

                # Move to destination
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(downloaded), str(destination))
                return True, None

        except subprocess.TimeoutExpired:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF[attempt - 1])
                continue
            return False, "Timeout after retries"

        except Exception as e:
            return False, str(e)[:100]

    return False, f"Failed after {MAX_RETRIES} attempts"


def download_study(study_uid: str, data_dir: Path, dry_run: bool = False) -> dict:
    """Download all DICOM files for one labeled study."""

    print("=" * 90)
    print(f"LABELED STUDY DOWNLOAD: {study_uid[-12:]}")
    print("=" * 90)
    print()

    # Load metadata
    md = load_simple_metadata(Path(data_dir))
    output_root = Path(data_dir) / "raw" / "train_series"

    # Discover files
    discovery = discover_study_files(study_uid)
    if not discovery['success']:
        print(f"ERROR: {discovery['error']}")
        return {'success': False, 'errors': [discovery['error']]}

    print(f"Discovery: {discovery['total_files']} files in {len(discovery['by_series'])} series")
    print()

    # Verify against metadata
    verify = verify_against_metadata(study_uid, discovery['series_order'], md)
    print(f"Verification: {verify['message']}")
    if not verify['valid']:
        print(f"  Missing: {verify['missing']}")
        print(f"  Extra: {verify['extra']}")
        return {'success': False, 'errors': [f"Metadata mismatch: {verify['message']}"]}
    print()

    if dry_run:
        print("[DRY-RUN] Would download:")
        for series_uid in discovery['series_order']:
            files = discovery['by_series'][series_uid]
            print(f"  {series_uid[-12:]}: {len(files)} files")
        print()
        return {
            'success': True,
            'dry_run': True,
            'study_uid': study_uid,
            'series_count': len(discovery['by_series']),
            'total_files': discovery['total_files'],
            'by_series': {k: len(v) for k, v in discovery['by_series'].items()},
        }

    # Download files
    print("Downloading files:")
    print()

    stats = {
        'by_series': {},
        'total_downloaded': 0,
        'total_skipped': 0,
        'total_failed': 0,
        'total_size_mb': 0.0,
        'errors': [],
    }

    for series_uid in discovery['series_order']:
        file_paths = discovery['by_series'][series_uid]
        series_dest = output_root / study_uid / series_uid

        downloaded = 0
        skipped = 0
        failed = 0
        size = 0.0

        for file_path in file_paths:
            # Extract filename
            filename = file_path.split('/')[-1]
            destination = series_dest / filename

            # Skip if already valid
            if destination.exists() and is_valid_dicom(destination):
                skipped += 1
                continue

            # Download
            success, error = download_file_with_retry(file_path, destination)

            if success:
                downloaded += 1
                size += destination.stat().st_size / (1024 * 1024)
            else:
                failed += 1
                stats['errors'].append(f"{series_uid[-12:]}/{filename}: {error}")

        stats['by_series'][series_uid] = {
            'downloaded': downloaded,
            'skipped': skipped,
            'failed': failed,
            'size_mb': size,
        }
        stats['total_downloaded'] += downloaded
        stats['total_skipped'] += skipped
        stats['total_failed'] += failed
        stats['total_size_mb'] += size

        status = "OK" if failed == 0 else "PARTIAL"
        print(f"  {series_uid[-12:]}: {downloaded} new + {skipped} existing "
              f"({size:.1f} MB) [{status}]")

    print()
    print("=" * 90)
    print("DOWNLOAD SUMMARY")
    print("=" * 90)
    print(f"Study: {study_uid[-12:]}")
    print(f"Series: {len(discovery['by_series'])}")
    print(f"  Per-series breakdown:")
    for series_uid in discovery['series_order']:
        s = stats['by_series'][series_uid]
        print(f"    {series_uid[-12:]}: {s['downloaded']} + {s['skipped']} "
              f"({s['size_mb']:.1f} MB)")
    print(f"Total files:")
    print(f"  Downloaded: {stats['total_downloaded']}")
    print(f"  Already present: {stats['total_skipped']}")
    print(f"  Failed: {stats['total_failed']}")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB")

    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for err in stats['errors'][:5]:
            print(f"  {err}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")

    all_success = stats['total_failed'] == 0
    print(f"\nStatus: {'SUCCESS' if all_success else 'PARTIAL'}")
    print("=" * 90)

    return {
        'success': all_success,
        'study_uid': study_uid,
        'series_count': len(discovery['by_series']),
        'by_series_counts': {k: len(v) for k, v in discovery['by_series'].items()},
        'total_files_expected': discovery['total_files'],
        'files_downloaded': stats['total_downloaded'],
        'files_skipped': stats['total_skipped'],
        'files_failed': stats['total_failed'],
        'total_size_mb': stats['total_size_mb'],
        'destination_root': str(output_root / study_uid),
        'errors': stats['errors'],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_uid", help="Study Instance UID")
    parser.add_argument("--data-dir", type=Path, default=Path("data/rsna-knee"),
                        help="Dataset directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")

    args = parser.parse_args(argv)
    result = download_study(args.study_uid, args.data_dir, args.dry_run)

    return 0 if result['success'] else 1


if __name__ == "__main__":
    raise SystemExit(main())
