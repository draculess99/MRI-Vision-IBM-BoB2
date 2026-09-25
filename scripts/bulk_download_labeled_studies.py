"""Optimized bulk downloader for RSNA labeled studies.

Caches Kaggle competition file listing once, filters for labeled studies,
builds a manifest, and downloads with bounded concurrency.

Features:
- Single Kaggle API call to cache all files (not per-study)
- Filters cached listing for 58 labeled StudyInstanceUIDs
- Builds .dcm manifest grouped by study/series
- Bounded concurrency (3-5 simultaneous downloads)
- DICOM validation, staged writes, retries, resumability

Usage:
  # Cache file listing and build manifest
  python scripts/bulk_download_labeled_studies.py --cache-only [--data-dir DIR]

  # Download 3 studies (test)
  python scripts/bulk_download_labeled_studies.py --dry-run  # Preview
  python scripts/bulk_download_labeled_studies.py --limit 3  # Download first 3

  # Download all 58
  python scripts/bulk_download_labeled_studies.py
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue

import pandas as pd

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
except ImportError:
    ThreadPoolExecutor = None


COMPETITION = "rsna-knee-abnormality-detection"
CACHE_FILE = Path("data/rsna-knee/kaggle_files_cache.json")
CACHE_PARTIAL_FILE = Path("data/rsna-knee/kaggle_files_cache.partial.json")
MAX_WORKERS = 4
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]
CHECKPOINT_INTERVAL = 20
CACHE_PAGE_SAFETY_LIMIT = 10000
CACHE_SUCCESS_DELAY = 0.75
KAGGLE_PAGE_SIZE = 200
TRANSIENT_PAGE_RETRIES = 3
CIRCUIT_BREAKER_THRESHOLD = 20
MAX_FAILURE_EXAMPLES = 5
ERROR_TEXT_LIMIT = 200

CATEGORY_RATE_LIMIT = "HTTP 429 / too many requests"
CATEGORY_AUTH = "HTTP 401 / authentication"
CATEGORY_FORBIDDEN = "HTTP 403 / forbidden"
CATEGORY_NOT_FOUND = "HTTP 404 / not found"
CATEGORY_SERVER_ERROR = "HTTP 5xx"
CATEGORY_TIMEOUT = "timeout"
CATEGORY_CONNECTION = "connection / temporary error"
CATEGORY_DICOM = "DICOM validation failure"
CATEGORY_UNKNOWN = "unknown Kaggle CLI failure"
TRANSIENT_CATEGORIES = frozenset(
    {CATEGORY_RATE_LIMIT, CATEGORY_SERVER_ERROR, CATEGORY_TIMEOUT, CATEGORY_CONNECTION}
)

_SECRET_ENV_NAMES = ("KAGGLE_KEY", "KAGGLE_USERNAME", "KAGGLE_API_TOKEN")
_SECRET_PATTERNS = (
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer <redacted>"),
    (re.compile(r"(?i)([\"']?\b(?:api[_-]?key|key|username|access[_-]?token|refresh[_-]?token|token|password|secret|authorization)\b[\"']?)"
                r"(\s*[:=]\s*)([\"']?)[^\s,\"'}]+"), r"\1\2\3<redacted>"),
    # Opaque 32+ character runs (legacy 32-hex keys, KGAT_ tokens). Requiring a letter or underscore keeps
    # all-digit UID segments (up to 38 digits) readable in file paths.
    (re.compile(r"\b(?=[A-Za-z0-9_-]*[A-Za-z_])[A-Za-z0-9_-]{32,}\b"), "<redacted>"),
)


@dataclass(frozen=True)
class DownloadFailure:
    """A permanently failed file download, kept for the end-of-run failure report."""
    file_path: str
    message: str
    category: str
    attempts: int
    returncode: int | None = None


class CircuitBreaker:
    """Trips after `threshold` consecutive permanent failures; any success resets the count."""

    def __init__(self, threshold: int | None = None):
        self.threshold = CIRCUIT_BREAKER_THRESHOLD if threshold is None else threshold
        self.consecutive_failures = 0
        self.tripped = False

    def record(self, success: bool) -> bool:
        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.threshold:
                self.tripped = True
        return self.tripped


def sanitize_error_text(text, limit: int = ERROR_TEXT_LIMIT) -> str:
    """Collapse whitespace and redact credentials/home paths before truncating, so no secret is half-cut."""
    text = " ".join(str(text or "").split())
    for name in _SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if value and len(value) >= 4:
            text = text.replace(value, "<redacted>")
    text = text.replace(str(Path.home()), "~")
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text[:limit]


def _has_status(text: str, *codes: str) -> bool:
    """True if an HTTP status code appears as its own token (not inside a UID, path, or larger number)."""
    return re.search(r"(?<![\d./_-])(?:%s)(?![\d/_-]|\.\w)" % "|".join(codes), text) is not None


def classify_error(message) -> str:
    text = " ".join(str(message or "").split()).lower()
    if _has_status(text, "429") or "too many requests" in text:
        return CATEGORY_RATE_LIMIT
    if _has_status(text, "401") or "unauthorized" in text or "authenticat" in text:
        return CATEGORY_AUTH
    if _has_status(text, "403") or "forbidden" in text:
        return CATEGORY_FORBIDDEN
    if _has_status(text, "404"):
        return CATEGORY_NOT_FOUND
    if _has_status(text, "500", "502", "503", "504"):
        return CATEGORY_SERVER_ERROR
    if "timeout" in text or "timed out" in text:
        return CATEGORY_TIMEOUT
    if "connection" in text or "temporarily" in text:
        return CATEGORY_CONNECTION
    if "not valid dicom" in text or "no .dcm file" in text:
        return CATEGORY_DICOM
    return CATEGORY_UNKNOWN


def is_transient_error(message) -> bool:
    return classify_error(message) in TRANSIENT_CATEGORIES


def quarantine_incompatible_checkpoint(checkpoint_file: Path, reason: str):
    """Preserve incompatible checkpoint with timestamp for inspection/recovery."""
    if checkpoint_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_file = checkpoint_file.parent / f"{checkpoint_file.name}.incompatible.{timestamp}"
        checkpoint_file.rename(quarantine_file)
        print(f"Checkpoint incompatible ({reason}). Preserved at: {quarantine_file}", flush=True)


def compute_expected_series_mapping(data_dir: Path, target_studies: set) -> dict:
    """Load expected series per study from metadata.

    Returns dict: study_uid -> set of series_uid
    """
    expected = {}

    # Try test manifest first (for test environments)
    test_manifest_file = Path(data_dir) / "labeled_studies_to_download.json"
    if test_manifest_file.exists():
        try:
            with open(test_manifest_file) as f:
                manifest = json.load(f)
            return {k: set(v) for k, v in manifest.items() if k in target_studies}
        except (json.JSONDecodeError, IOError):
            pass

    # Try loading real metadata
    try:
        md = load_simple_metadata(data_dir)
        for study_uid in target_studies:
            study_uid_str = str(study_uid)
            series_df = md.series_for(study_uid_str, "train")
            if len(series_df) > 0:
                expected[study_uid_str] = set(
                    series_df.SeriesInstanceUID.astype(str).tolist()
                )
    except FileNotFoundError:
        pass

    return expected


def compute_full_manifest_checksum(target_studies: set, expected_series_mapping: dict) -> str:
    """Compute checksum based on full expected study->series mapping.

    This ensures checkpoints are invalidated if expected series change.
    """
    manifest = {
        "target_studies": sorted(str(s) for s in target_studies),
        "expected_series": {
            study: sorted(str(s) for s in series)
            for study, series in sorted(expected_series_mapping.items())
        }
    }
    return compute_manifest_checksum(manifest)


def compute_manifest_checksum(manifest: dict) -> str:
    """Compute deterministic SHA-256 checksum of manifest.

    Uses canonical JSON ordering to ensure same manifest always produces same checksum.
    """
    canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


def save_checkpoint(checkpoint_file: Path, checkpoint_data: dict):
    """Save checkpoint atomically using temp file + rename."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = checkpoint_file.parent / f".{checkpoint_file.name}.tmp"
    with open(temp_file, 'w') as f:
        json.dump(checkpoint_data, f)
    temp_file.replace(checkpoint_file)


def load_checkpoint(checkpoint_file: Path) -> dict | None:
    """Load checkpoint if it exists and is valid."""
    if not checkpoint_file.exists():
        return None
    try:
        with open(checkpoint_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


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


def load_labeled_studies(data_dir: Path) -> set:
    """Get set of 58 labeled StudyInstanceUIDs."""
    md = load_simple_metadata(data_dir)
    labeled = md.train_series  # All rows from train_series.csv

    # Read train.csv to find which studies have all labels
    train_file = Path(data_dir) / "train.csv"
    train_df = pd.read_csv(train_file, dtype={"StudyInstanceUID": "string"})

    target_cols = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
    ]

    # Studies with all labels (no NaN in any target column)
    complete_labeled = train_df.dropna(subset=target_cols)
    return set(complete_labeled["StudyInstanceUID"].astype(str).tolist())


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


def fetch_and_cache_file_listing(data_dir: Path, cache_file: Path = None, target_studies: set = None) -> dict:
    """Fetch competition file listing from Kaggle and cache locally.

    Args:
        data_dir: Dataset directory
        cache_file: Where to save final cache. Defaults to the module-level CACHE_FILE,
            resolved at call time (not at import time) so tests can monkeypatch
            CACHE_FILE and be guaranteed the real path is never touched.
        target_studies: Set of study UIDs to collect (required for checkpoint validation)

    Returns: {study_uid: {series_uid: [file_paths]}}

    Hardened features:
    - Only stores target_studies files (no unrelated studies in memory)
    - Preserves incompatible checkpoints for inspection
    - Bounded retries per page with checkpoint save before retry
    - Saves checkpoint on all errors before exiting
    - Uses sets to prevent file path duplication on replay
    - Full manifest checksum (study->series mapping)
    - Atomic final cache write with temp file
    """

    if cache_file is None:
        cache_file = CACHE_FILE

    if target_studies is None:
        target_studies = set()

    # Canonical target studies set for consistency checking
    canonical_target_studies = sorted(set(str(s) for s in target_studies))

    # Load expected series mapping early for checkpoint validation
    expected_series_mapping = {}
    if target_studies:
        expected_series_mapping = compute_expected_series_mapping(data_dir, target_studies)

    print("Fetching Kaggle file listing (this takes ~10-15 minutes)...")
    print(f"(Querying with page size {KAGGLE_PAGE_SIZE}, safety limit {CACHE_PAGE_SAFETY_LIMIT})")

    # Use lists for files, but track as sets per-page to prevent duplication on resume
    all_files = defaultdict(lambda: defaultdict(list))
    found_series_per_study = defaultdict(set)
    seen_files_per_study_series = defaultdict(lambda: defaultdict(set))  # Track files seen to prevent resume duplication

    # Load checkpoint if available
    checkpoint = load_checkpoint(CACHE_PARTIAL_FILE)
    pages = 0
    page_token = None
    pages_at_checkpoint = 0  # Track how many pages were completed when resuming

    if checkpoint:
        # Verify checkpoint compatibility using full manifest checksum
        full_checksum = compute_full_manifest_checksum(target_studies, expected_series_mapping)
        checkpoint_checksum = checkpoint.get("manifest_checksum", "")
        checkpoint_target = sorted(set(str(s) for s in checkpoint.get("target_studies", [])))

        if checkpoint_target == canonical_target_studies and full_checksum == checkpoint_checksum:
            # Resume from checkpoint
            pages = checkpoint.get("pages", 0)
            page_token = checkpoint.get("page_token")
            pages_at_checkpoint = pages  # Remember checkpoint state for deduplication logic
            # Restore nested list structure and build seen_files tracker for deduplication on resume
            checkpoint_files = checkpoint.get("labeled_files", {})
            all_files = defaultdict(lambda: defaultdict(list))
            seen_files_per_study_series = defaultdict(lambda: defaultdict(set))
            for study, series_dict in checkpoint_files.items():
                for series, files in series_dict.items():
                    files_list = list(files) if not isinstance(files, list) else files
                    all_files[study][series] = files_list
                    # Track what we've already seen to prevent duplication on resume only
                    seen_files_per_study_series[study][series] = set(files_list)
            found_series_per_study = defaultdict(set, {
                k: set(v) for k, v in checkpoint.get("found_series_per_study", {}).items()
            })
            print(f"Resuming from checkpoint: page {pages}, token {page_token[:20] if page_token else 'None'}...")
        else:
            # Checkpoint mismatch - preserve it for inspection
            reason = "target_studies changed" if checkpoint_target != canonical_target_studies else "series mapping changed"
            quarantine_incompatible_checkpoint(CACHE_PARTIAL_FILE, reason)
            pages = 0
            page_token = None

    cmd_base = ["kaggle", "competitions", "files", COMPETITION, "-v", "--page-size", str(KAGGLE_PAGE_SIZE)]

    def save_progress_checkpoint(current_page: int, current_token: str | None):
        """Save current progress to checkpoint."""
        if target_studies:
            checkpoint_data = {
                "pages": current_page,
                "page_token": current_token,
                "labeled_files": {
                    k: {series: sorted(list(files)) for series, files in v.items()}
                    for k, v in all_files.items()
                },
                "found_series_per_study": {
                    k: sorted(list(v)) for k, v in found_series_per_study.items()
                },
                "target_studies": canonical_target_studies,
                "manifest_checksum": compute_full_manifest_checksum(target_studies, expected_series_mapping),
            }
            save_checkpoint(CACHE_PARTIAL_FILE, checkpoint_data)

    try:
        while pages < CACHE_PAGE_SAFETY_LIMIT:
            retry_count = 0

            while retry_count < TRANSIENT_PAGE_RETRIES:
                try:
                    if page_token:
                        cmd = cmd_base + ["--page-token", page_token]
                    else:
                        cmd = cmd_base

                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                    if result.returncode != 0:
                        error_msg = result.stderr or result.stdout

                        # Check for transient errors that should be retried
                        if any(code in error_msg for code in ["429", "500", "502", "503", "504"]):
                            retry_count += 1
                            if retry_count < TRANSIENT_PAGE_RETRIES:
                                if "429" in error_msg or "Too Many Requests" in error_msg:
                                    wait_time = min(60, 5 + pages // 50)  # Exponential backoff for 429
                                else:
                                    wait_time = min(30, 2 + (pages % 10))  # Gentle backoff for 5xx
                                print(f"  Transient error on page {pages} (retry {retry_count}/{TRANSIENT_PAGE_RETRIES}), waiting {wait_time}s...", flush=True)
                                save_progress_checkpoint(pages, page_token)
                                time.sleep(wait_time)
                                continue
                            else:
                                # Retries exhausted - save checkpoint and fail
                                print(f"  Retries exhausted on page {pages}. Saving checkpoint for recovery.", flush=True)
                                save_progress_checkpoint(pages, page_token)
                                raise Exception(f"Kaggle error after {TRANSIENT_PAGE_RETRIES} retries on page {pages}: {error_msg[:100]} - checkpoint saved, can resume")
                        elif not error_msg:
                            # Empty error - might be transient, retry
                            retry_count += 1
                            if retry_count < TRANSIENT_PAGE_RETRIES:
                                print(f"  Empty response on page {pages} (retry {retry_count}/{TRANSIENT_PAGE_RETRIES}), retrying...", flush=True)
                                save_progress_checkpoint(pages, page_token)
                                time.sleep(5)
                                continue
                            else:
                                save_progress_checkpoint(pages, page_token)
                                raise Exception(f"Empty response after {TRANSIENT_PAGE_RETRIES} retries on page {pages} - checkpoint saved, can resume")
                        else:
                            # Non-transient error (auth, etc) - save checkpoint and fail
                            print(f"  Non-transient error on page {pages}. Saving checkpoint for recovery.", flush=True)
                            save_progress_checkpoint(pages, page_token)
                            raise Exception(f"Kaggle CLI error: {error_msg[:100]} - checkpoint saved, can resume")

                    # Success - process the page
                    lines = result.stdout.strip().split('\n')
                    pages += 1

                    # Rate limiting: delay after every request
                    time.sleep(CACHE_SUCCESS_DELAY)

                    if pages % 20 == 0:
                        print(f"  Page {pages}...", flush=True)

                    # Extract next page token
                    next_token = None
                    for line in lines:
                        if line.startswith("Next Page Token = "):
                            next_token = line.replace("Next Page Token = ", "").strip()
                            break

                    # Collect train_series files - ONLY target studies
                    for line in lines:
                        if "train_series" in line and ".dcm" in line:
                            parts = line.split(',')
                            if len(parts) >= 1:
                                file_path = parts[0].strip()
                                path_parts = file_path.split('/')
                                if len(path_parts) >= 3:
                                    study_uid = path_parts[1]
                                    series_uid = path_parts[2]
                                    # Only collect if target_studies specified AND study is in targets
                                    if not target_studies or study_uid in target_studies:
                                        # On resume: deduplicate to prevent duplication when replaying pages
                                        # On fresh start: keep all files even if duplicates appear
                                        should_add = True
                                        if pages_at_checkpoint > 0 and file_path in seen_files_per_study_series[study_uid][series_uid]:
                                            should_add = False

                                        if should_add:
                                            all_files[study_uid][series_uid].append(file_path)
                                            seen_files_per_study_series[study_uid][series_uid].add(file_path)
                                        if study_uid in target_studies:
                                            found_series_per_study[study_uid].add(series_uid)

                    # Save checkpoint periodically
                    if pages % CHECKPOINT_INTERVAL == 0 and target_studies:
                        save_progress_checkpoint(pages, next_token)

                    if not next_token:
                        break

                    page_token = next_token
                    break  # Break retry loop on success

                except subprocess.TimeoutExpired:
                    retry_count += 1
                    if retry_count < TRANSIENT_PAGE_RETRIES:
                        print(f"  Timeout on page {pages} (retry {retry_count}/{TRANSIENT_PAGE_RETRIES}), retrying...", flush=True)
                        save_progress_checkpoint(pages, page_token)
                        time.sleep(5)
                        continue
                    else:
                        # Timeout after retries - preserve checkpoint
                        print(f"  Timeout after {TRANSIENT_PAGE_RETRIES} retries on page {pages}. Saving checkpoint.", flush=True)
                        save_progress_checkpoint(pages, page_token)
                        raise Exception(f"Kaggle CLI timeout on page {pages} after {TRANSIENT_PAGE_RETRIES} retries - checkpoint saved at {CACHE_PARTIAL_FILE}, can resume")

            # Check if we need to break after successful page
            if not next_token:
                break

        # Check if we hit safety limit
        if pages >= CACHE_PAGE_SAFETY_LIMIT:
            raise Exception(f"Pagination exceeded safety limit of {CACHE_PAGE_SAFETY_LIMIT} pages")

    except KeyboardInterrupt:
        # Save checkpoint before exiting on Ctrl+C
        if target_studies and pages >= 0:
            save_progress_checkpoint(pages, page_token)
        print("\nInterrupted. Checkpoint saved.", flush=True)
        sys.exit(1)

    print(f"\nCached {len(all_files)} studies, {sum(len(s) for s in all_files.values())} series")

    # Validate completeness if target_studies specified
    if target_studies:
        expected_series_per_study = {}

        # Try to load test manifest first (for test environments)
        test_manifest_file = Path(data_dir) / "labeled_studies_to_download.json"
        if test_manifest_file.exists():
            try:
                with open(test_manifest_file) as f:
                    manifest = json.load(f)
                expected_series_per_study = {k: set(v) for k, v in manifest.items()}
            except (json.JSONDecodeError, IOError):
                pass

        # If no test manifest, try loading real metadata
        if not expected_series_per_study:
            try:
                md = load_simple_metadata(data_dir)
                for study_uid in target_studies:
                    study_uid_str = str(study_uid)
                    series_df = md.series_for(study_uid_str, "train")
                    if len(series_df) > 0:
                        expected_series_per_study[study_uid_str] = set(
                            series_df.SeriesInstanceUID.astype(str).tolist()
                        )
            except FileNotFoundError:
                pass

        # Validate if we have expected series
        if expected_series_per_study:
            missing_studies = set()
            missing_series_per_study = {}

            for study_uid in target_studies:
                study_uid_str = str(study_uid)
                if study_uid_str not in all_files:
                    missing_studies.add(study_uid_str)
                elif study_uid_str in expected_series_per_study:
                    expected_series = expected_series_per_study[study_uid_str]
                    found_series = set(all_files[study_uid_str].keys())
                    missing = expected_series - found_series
                    if missing:
                        missing_series_per_study[study_uid_str] = missing

            if missing_studies or missing_series_per_study:
                errors = []
                for study in missing_studies:
                    errors.append(f"Missing study: {study}")
                for study, series in missing_series_per_study.items():
                    errors.append(f"Study {study}: missing series {len(series)}")
                raise Exception(f"Incomplete cache: {'; '.join(errors[:3])}")

    # Filter to target studies only if specified
    if target_studies:
        filtered_files = {
            study: series_files
            for study, series_files in all_files.items()
            if study in target_studies
        }
    else:
        filtered_files = all_files

    # Convert sets to sorted lists for JSON serialization
    cache_data = {
        study: {
            series: sorted(list(files)) if isinstance(files, set) else files
            for series, files in series_files.items()
        }
        for study, series_files in filtered_files.items()
    }

    # Atomic final cache write: write to temp file, then rename
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temp_cache = cache_file.parent / f".{cache_file.name}.tmp"
    with open(temp_cache, 'w') as f:
        json.dump(cache_data, f)
    # Atomic rename - replaces existing file atomically
    temp_cache.replace(cache_file)

    # Delete partial checkpoint only after successful final cache creation
    CACHE_PARTIAL_FILE.unlink(missing_ok=True)

    print(f"Cache saved to: {cache_file}")
    print(f"Cache size: {cache_file.stat().st_size / (1024*1024):.1f} MB")

    return filtered_files


def filter_for_labeled_studies(all_files: dict, labeled_studies: set) -> dict:
    """Filter cached file listing for labeled studies only."""
    labeled_files = {
        study: series_files
        for study, series_files in all_files.items()
        if study in labeled_studies
    }

    print(f"\nFiltered: {len(labeled_files)} labeled studies found")
    print(f"  Total series: {sum(len(s) for s in labeled_files.values())}")
    print(f"  Total files: {sum(len(f) for s in labeled_files.values() for f in s.values())}")

    return labeled_files


def build_manifest(labeled_files: dict, data_dir: Path) -> dict:
    """Build manifest with study/series/file organization and metadata verification."""
    md = load_simple_metadata(data_dir)

    manifest = {}
    errors = []

    for study_uid in sorted(labeled_files.keys()):
        series_files = labeled_files[study_uid]

        # Verify against metadata
        metadata_series = set(
            md.series_for(study_uid, "train").SeriesInstanceUID.astype(str).tolist()
        )
        discovered_series = set(series_files.keys())

        missing = metadata_series - discovered_series
        extra = discovered_series - metadata_series

        if missing or extra:
            errors.append(f"{study_uid[-12:]}: missing {len(missing)}, extra {len(extra)}")

        manifest[study_uid] = {
            'series': {
                series_uid: {
                    'files': files,
                    'count': len(files),
                }
                for series_uid, files in series_files.items()
            },
            'metadata_valid': len(missing) == 0 and len(extra) == 0,
        }

    if errors:
        print(f"\nWarnings:")
        for err in errors[:5]:
            print(f"  {err}")

    return manifest


def download_file_with_retry(file_path: str, destination: Path) -> tuple[bool, DownloadFailure | None]:
    """Download one DICOM file with retry logic.

    Returns (True, None) on success, or (False, DownloadFailure) once the file has failed permanently.
    """

    def failure(message, attempts, returncode=None):
        message = sanitize_error_text(message)
        return False, DownloadFailure(file_path, message, classify_error(message), attempts, returncode)

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
                    error_msg = sanitize_error_text(result.stderr or result.stdout)

                    if is_transient_error(error_msg) and attempt < MAX_RETRIES:
                        time.sleep(RETRY_BACKOFF[attempt - 1])
                        continue
                    else:
                        return failure(error_msg, attempt, result.returncode)

                dcm_files = list(staging.rglob("*.dcm"))
                if not dcm_files:
                    return failure("No .dcm file in download", attempt, result.returncode)

                downloaded = dcm_files[0]

                if not is_valid_dicom(downloaded):
                    return failure("Downloaded file is not valid DICOM", attempt, result.returncode)

                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(downloaded), str(destination))
                return True, None

        except subprocess.TimeoutExpired:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF[attempt - 1])
                continue
            return failure("Timeout after retries", attempt)
        except Exception as e:
            return failure(str(e), attempt)

    return failure(f"Failed after {MAX_RETRIES} attempts", MAX_RETRIES)


def download_studies_concurrent(
    studies_to_download: list,
    labeled_files: dict,
    output_root: Path,
    dry_run: bool = False,
    max_workers: int = MAX_WORKERS,
    breaker: CircuitBreaker | None = None
) -> dict:
    """Download multiple studies with concurrent file downloads.

    One circuit breaker spans all studies: once it trips, the remaining studies are not started.
    """

    results = {}
    studies_not_started = []
    breaker = breaker if breaker is not None else CircuitBreaker()
    start_time = time.time()

    for study_uid in studies_to_download:
        if breaker.tripped:
            studies_not_started.append(study_uid)
            continue
        study_result = download_single_study(
            study_uid,
            labeled_files[study_uid],
            output_root,
            dry_run=dry_run,
            max_workers=max_workers,
            breaker=breaker
        )
        results[study_uid] = study_result

    elapsed = time.time() - start_time
    failures = [failure for r in results.values() for failure in r['failures']]

    return {
        'studies': results,
        'elapsed_seconds': elapsed,
        'total_downloaded': sum(r['downloaded'] for r in results.values()),
        'total_skipped': sum(r['skipped'] for r in results.values()),
        'total_failed': sum(r['failed'] for r in results.values()),
        'total_size_mb': sum(r['size_mb'] for r in results.values()),
        'total_not_attempted': sum(r['not_attempted'] for r in results.values()),
        'studies_not_started': studies_not_started,
        'circuit_breaker_tripped': breaker.tripped,
        'circuit_breaker_threshold': breaker.threshold,
        'failure_categories': Counter(failure.category for failure in failures),
        'failure_examples': failures[:MAX_FAILURE_EXAMPLES],
    }


def print_failure_report(result: dict):
    """Print aggregated failure categories and a few sanitized examples (never one line per failure)."""
    categories = result.get('failure_categories') or {}
    if not categories:
        return
    print("\nFailure categories:")
    for category, count in sorted(categories.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {category}: {count}")
    examples = result.get('failure_examples') or []
    print(f"\nExample failed files (first {len(examples)} of {sum(categories.values())}):")
    for failure in examples:
        returncode = "n/a" if failure.returncode is None else failure.returncode
        print(f"  {failure.file_path} | returncode={returncode} | attempts={failure.attempts} | "
              f"{failure.category}: {failure.message}")


def download_single_study(
    study_uid: str,
    series_files: dict,
    output_root: Path,
    dry_run: bool = False,
    max_workers: int = MAX_WORKERS,
    breaker: CircuitBreaker | None = None
) -> dict:
    """Download all files for one study with concurrent downloads."""

    stats = {
        'study_uid': study_uid,
        'series': {},
        'downloaded': 0,
        'skipped': 0,
        'failed': 0,
        'not_attempted': 0,
        'size_mb': 0.0,
        'errors': [],
        'failures': [],
    }

    print(f"\nStudy {study_uid[-12:]}:")

    if dry_run:
        total_files = sum(len(files) for files in series_files.values())
        print(f"  [DRY-RUN] Would download {total_files} files across {len(series_files)} series")
        for series_uid, files in series_files.items():
            stats['series'][series_uid] = {
                'downloaded': 0,
                'skipped': 0,
                'failed': 0,
                'size_mb': 0.0,
                'files': len(files),
            }
        return stats

    # Check what's already downloaded
    study_dir = output_root / study_uid
    existing_valid = {}
    for series_uid, files in series_files.items():
        series_dir = study_dir / series_uid
        existing_valid[series_uid] = set()
        if series_dir.exists():
            for file_path in files:
                filename = file_path.split('/')[-1]
                local_path = series_dir / filename
                if local_path.exists() and is_valid_dicom(local_path):
                    existing_valid[series_uid].add(filename)

    # Build download queue
    download_queue = []
    for series_uid, files in series_files.items():
        series_dest = study_dir / series_uid
        for file_path in files:
            filename = file_path.split('/')[-1]
            if filename not in existing_valid[series_uid]:
                download_queue.append((file_path, series_dest / filename))

    skipped_per_series = {
        series_uid: len(existing_valid[series_uid])
        for series_uid in series_files.keys()
    }

    # Download with bounded concurrency
    if download_queue and ThreadPoolExecutor:
        breaker = breaker if breaker is not None else CircuitBreaker()

        def tally(future) -> bool:
            file_path, dest = futures[future]
            series_uid = dest.parent.name
            try:
                success, failure = future.result()
                if success:
                    stats['downloaded'] += 1
                    stats['size_mb'] += dest.stat().st_size / (1024 * 1024)
                else:
                    stats['failed'] += 1
                    stats['errors'].append(f"{series_uid[-12:]}: {failure.message}")
                    stats['failures'].append(failure)
            except Exception as e:
                message = sanitize_error_text(e)
                stats['failed'] += 1
                stats['errors'].append(f"{series_uid[-12:]}: {message}")
                stats['failures'].append(DownloadFailure(file_path, message, classify_error(message), 1))
                success = False
            return success

        handled = set()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_file_with_retry, file_path, dest): (file_path, dest)
                for file_path, dest in download_queue
            }

            for future in as_completed(futures):
                handled.add(future)
                if breaker.record(tally(future)):
                    # Cancel only files that have not started; downloads already running finish normally
                    # (staged write + validation + move), so no partial file is ever left behind.
                    for pending in futures:
                        pending.cancel()
                    print(f"  Circuit breaker tripped: {breaker.consecutive_failures} consecutive permanent "
                          f"failures. No new downloads will be started.")
                    break

        # The executor has now waited for in-flight downloads; account for them without feeding the breaker.
        for future in futures:
            if future in handled:
                continue
            if future.cancelled():
                stats['not_attempted'] += 1
            else:
                tally(future)

    # Summarize per-series
    for series_uid in series_files.keys():
        series_dir = study_dir / series_uid
        final_count = sum(
            1 for f in series_dir.glob("*.dcm") if is_valid_dicom(f)
        ) if series_dir.exists() else 0

        stats['series'][series_uid] = {
            'total_files': len(series_files[series_uid]),
            'downloaded': sum(1 for f in download_queue if f[1].parent.name == series_uid),
            'skipped': skipped_per_series[series_uid],
            'final_valid': final_count,
        }

        stats['skipped'] += skipped_per_series[series_uid]

    # Print summary
    total_files = sum(len(files) for files in series_files.values())
    not_attempted_note = f" [{stats['not_attempted']} not attempted]" if stats['not_attempted'] else ""
    print(f"  Files: {stats['downloaded']} downloaded + {stats['skipped']} existing "
          f"({stats['size_mb']:.1f} MB) [{stats['failed']} failed]{not_attempted_note}")

    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/rsna-knee"),
                        help="Dataset directory")
    parser.add_argument("--cache-only", action="store_true",
                        help="Fetch and cache file listing only, don't download")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview downloads without fetching files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Download only N studies (for testing)")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS,
                        help="Concurrent downloads")

    args = parser.parse_args(argv)

    try:
        # Load labeled studies
        labeled_studies = load_labeled_studies(args.data_dir)
        print(f"Found {len(labeled_studies)} labeled studies in metadata\n")

        # Check for cached listing
        if CACHE_FILE.exists() and not args.cache_only:
            print(f"Using cached file listing: {CACHE_FILE}")
            with open(CACHE_FILE) as f:
                all_files = json.load(f)
            print(f"  Studies: {len(all_files)}, Size: {CACHE_FILE.stat().st_size / (1024*1024):.1f} MB\n")
        else:
            # Fetch and cache with target_studies for checkpoint validation
            all_files = fetch_and_cache_file_listing(args.data_dir, target_studies=labeled_studies)

        if args.cache_only:
            return 0

        # Filter for labeled studies
        labeled_files = filter_for_labeled_studies(all_files, labeled_studies)

        # Build manifest
        manifest = build_manifest(labeled_files, args.data_dir)

        # Determine studies to download
        studies = sorted(labeled_files.keys())
        if args.limit:
            studies = studies[:args.limit]

        print(f"\nDownloading {len(studies)} studies...")

        # Download
        output_root = Path(args.data_dir) / "raw" / "train_series"
        result = download_studies_concurrent(
            studies,
            labeled_files,
            output_root,
            dry_run=args.dry_run,
            max_workers=args.max_workers
        )

        # Report
        print("\n" + "=" * 90)
        print("BULK DOWNLOAD SUMMARY")
        print("=" * 90)

        for study_uid, study_result in result['studies'].items():
            print(f"\n{study_uid[-12:]}:")
            for series_uid, s_stats in study_result['series'].items():
                print(f"  {series_uid[-12:]}: {s_stats.get('final_valid', 0)} valid DICOMs")

        print(f"\nTotals:")
        print(f"  Downloaded: {result['total_downloaded']}")
        print(f"  Skipped: {result['total_skipped']}")
        print(f"  Failed: {result['total_failed']}")
        print(f"  Size: {result['total_size_mb']:.1f} MB")
        print(f"  Time: {result['elapsed_seconds']:.1f}s")

        print_failure_report(result)

        if result['circuit_breaker_tripped']:
            print(f"\nCircuit breaker triggered after {result['circuit_breaker_threshold']} "
                  f"consecutive permanent download failures.")
            print(f"  {result['total_not_attempted']} queued files and "
                  f"{len(result['studies_not_started'])} studies were not attempted. "
                  f"Valid files on disk are kept and skipped on rerun.")
        print("=" * 90)

        return 0 if result['total_failed'] == 0 and not result['circuit_breaker_tripped'] else 1

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
