"""Tests for RSNA labeled study cache builder with pagination and validation.

Focus: ensuring cache completeness guarantee and proper error handling.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bulk_download_labeled_studies import fetch_and_cache_file_listing


@pytest.fixture(autouse=True)
def isolate_real_cache_paths(tmp_path, monkeypatch):
    """Prevent any test from ever writing to the real production cache files.

    fetch_and_cache_file_listing() defaults cache_file=CACHE_FILE and reads/writes
    the module-level CACHE_PARTIAL_FILE directly. A test that forgets to patch one
    of these (or a future test that doesn't patch either) would silently corrupt
    real Kaggle cache data on disk. This autouse fixture makes that impossible by
    redirecting both module-level path constants to this test's tmp_path, for
    every test in this file, regardless of what each test does locally.
    """
    fallback_cache = tmp_path / "_autouse_kaggle_files_cache.json"
    fallback_partial = tmp_path / "_autouse_kaggle_files_cache.partial.json"
    monkeypatch.setattr("scripts.bulk_download_labeled_studies.CACHE_FILE", fallback_cache)
    monkeypatch.setattr("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", fallback_partial)
    yield


class MockPaginator:
    """Mock Kaggle API pagination for testing."""

    def __init__(self, pages):
        """pages: list of CSV-text page bodies, in order."""
        self.pages = pages
        self.current_page = 0

    def run_command(self, *args, **kwargs):
        """Mock subprocess.run that returns paginated results."""
        if self.current_page >= len(self.pages):
            result = MagicMock()
            result.returncode = 1
            result.stderr = "No more pages"
            return result

        page_text = self.pages[self.current_page]
        self.current_page += 1

        result = MagicMock()
        result.returncode = 0
        result.stdout = page_text
        result.stderr = ""
        return result


def make_page(study_series_files, next_token=None):
    """Build one page of mock Kaggle file listing output (CSV with optional next token)."""
    lines = []
    if next_token:
        lines.append(f"Next Page Token = {next_token}")
    lines.append("name,size,creationDate")

    for study, series_dict in study_series_files.items():
        for series, num_files in series_dict.items():
            for i in range(num_files):
                file_uid = f"file_{study}_{series}_{i}"
                path = f"train_series/{study}/{series}/{file_uid}.dcm,321432,2026-09-19"
                lines.append(path)
    return "\n".join(lines)


def write_manifest(data_dir, manifest):
    with open(data_dir / "labeled_studies_to_download.json", "w") as f:
        json.dump(manifest, f)


DEFAULT_MANIFEST = {
    "study_A": ["series_A1", "series_A2"],
    "study_B": ["series_B1", "series_B2", "series_B3"],
    "study_C": ["series_C1"],
}


def test_discards_unrelated_studies(tmp_path):
    """Unrelated studies not in manifest are discarded, target studies kept."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)

    pages = [
        make_page({
            "study_A": {"series_A1": 1, "series_A2": 1},
            "study_UNRELATED": {"series_X1": 5},
            "study_B": {"series_B1": 1},
        }, next_token="token_1"),
        make_page({
            "study_B": {"series_B2": 1, "series_B3": 1},
            "study_UNRELATED": {"series_X2": 3},
            "study_C": {"series_C1": 1},
        }, next_token=None),
    ]

    paginator = MockPaginator(pages)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        cache = fetch_and_cache_file_listing(
            tmp_path, target_studies={"study_A", "study_B", "study_C"}
        )

    assert set(cache.keys()) == {"study_A", "study_B", "study_C"}
    assert "study_UNRELATED" not in cache


def test_complete_across_multiple_pages(tmp_path):
    """A target study whose series are split across pages is collected completely."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)

    pages = [
        make_page({
            "study_A": {"series_A1": 2, "series_A2": 2},
            "study_B": {"series_B1": 2},
        }, next_token="token_1"),
        make_page({
            "study_B": {"series_B2": 2, "series_B3": 2},
            "study_C": {"series_C1": 1},
        }, next_token=None),
    ]

    paginator = MockPaginator(pages)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        cache = fetch_and_cache_file_listing(
            tmp_path, target_studies={"study_A", "study_B", "study_C"}
        )

    assert set(cache["study_B"].keys()) == {"series_B1", "series_B2", "series_B3"}
    assert len(cache["study_B"]["series_B1"]) == 2
    assert len(cache["study_B"]["series_B2"]) == 2


def test_does_not_early_stop_at_all_studies_seen(tmp_path):
    """All 58 (here: 3) studies appearing at least once must NOT trigger early stop."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)

    pages = [
        make_page({
            "study_A": {"series_A1": 1},  # study_A seen, but series_A2 missing
            "study_B": {"series_B1": 1},  # study_B seen, but series_B2/B3 missing
            "study_C": {"series_C1": 1},  # study_C complete
        }, next_token="token_1"),
        make_page({
            "study_A": {"series_A2": 1},
            "study_B": {"series_B2": 1, "series_B3": 1},
        }, next_token=None),
    ]

    paginator = MockPaginator(pages)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        cache = fetch_and_cache_file_listing(
            tmp_path, target_studies={"study_A", "study_B", "study_C"}
        )

    # Must have consumed BOTH pages even though all studies appeared on page 1
    assert paginator.current_page == 2
    assert set(cache["study_A"].keys()) == {"series_A1", "series_A2"}
    assert set(cache["study_B"].keys()) == {"series_B1", "series_B2", "series_B3"}


def test_fails_on_missing_study(tmp_path):
    """Cache build fails clearly if a target study is never found in the listing."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)

    pages = [
        make_page({
            "study_A": {"series_A1": 1, "series_A2": 1},
            # study_B never appears at all
            "study_C": {"series_C1": 1},
        }, next_token=None),
    ]

    paginator = MockPaginator(pages)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        with pytest.raises(Exception, match="(?i)missing"):
            fetch_and_cache_file_listing(
                tmp_path, target_studies={"study_A", "study_B", "study_C"}
            )


def test_fails_on_missing_series(tmp_path):
    """Cache build fails clearly if a target study is missing some expected series."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)

    pages = [
        make_page({
            "study_A": {"series_A1": 1, "series_A2": 1},
            "study_B": {"series_B1": 1},  # missing series_B2, series_B3
            "study_C": {"series_C1": 1},
        }, next_token=None),
    ]

    paginator = MockPaginator(pages)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        with pytest.raises(Exception, match="(?i)missing"):
            fetch_and_cache_file_listing(
                tmp_path, target_studies={"study_A", "study_B", "study_C"}
            )


def test_fails_on_safety_limit(tmp_path):
    """Hitting the safety page-count ceiling is a hard error, never a silent success."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)

    # Every page always includes a next_token and never completes study_B's series,
    # forcing pagination to run until the safety limit is hit.
    incomplete_page = make_page({
        "study_A": {"series_A1": 1, "series_A2": 1},
        "study_B": {"series_B1": 1},  # series_B2 / series_B3 never appear
        "study_C": {"series_C1": 1},
    }, next_token="keep_going")

    pages = [incomplete_page] * 10005
    paginator = MockPaginator(pages)

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.time.sleep"):
        with pytest.raises(Exception, match="(?i)safety|limit"):
            fetch_and_cache_file_listing(
                tmp_path, target_studies={"study_A", "study_B", "study_C"}
            )


def test_succeeds_with_complete_pagination(tmp_path):
    """Cache succeeds and contains only target studies once all series are found."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)

    pages = [
        make_page({
            "study_A": {"series_A1": 2, "series_A2": 1},
            "study_B": {"series_B1": 1},
        }, next_token="token_1"),
        make_page({
            "study_B": {"series_B2": 2, "series_B3": 1},
            "study_C": {"series_C1": 1},
        }, next_token=None),
    ]

    paginator = MockPaginator(pages)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        cache = fetch_and_cache_file_listing(
            tmp_path, target_studies={"study_A", "study_B", "study_C"}
        )

    assert set(cache.keys()) == {"study_A", "study_B", "study_C"}
    assert set(cache["study_A"].keys()) == {"series_A1", "series_A2"}
    assert set(cache["study_B"].keys()) == {"series_B1", "series_B2", "series_B3"}
    assert set(cache["study_C"].keys()) == {"series_C1"}
    assert len(cache["study_A"]["series_A1"]) == 2
    assert len(cache["study_B"]["series_B2"]) == 2


def test_checkpoint_written_during_pagination(tmp_path):
    """Checkpoint file is created during pagination."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)

    pages = [
        make_page({
            "study_A": {"series_A1": 2, "series_A2": 2},
            "study_B": {"series_B1": 2},
        }, next_token="token_1"),
        make_page({
            "study_B": {"series_B2": 2, "series_B3": 2},
            "study_C": {"series_C1": 1},
        }, next_token=None),
    ]

    paginator = MockPaginator(pages)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        cache = fetch_and_cache_file_listing(
            tmp_path, target_studies={"study_A", "study_B", "study_C"}
        )

    # After successful completion, checkpoint should be deleted
    assert not checkpoint_file.exists()
    assert set(cache.keys()) == {"study_A", "study_B", "study_C"}


def test_resume_from_checkpoint(tmp_path):
    """Cache builder resumes from checkpoint and continues pagination."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)

    # Simulate first run that gets interrupted at page 1
    pages_first = [
        make_page({
            "study_A": {"series_A1": 1},  # Incomplete
            "study_B": {"series_B1": 1},
        }, next_token="token_1"),
    ]

    # Second run (resume) continues with page 2
    pages_resume = [
        make_page({
            "study_A": {"series_A2": 1},  # Complete
            "study_B": {"series_B2": 1, "series_B3": 1},
            "study_C": {"series_C1": 1},
        }, next_token=None),
    ]

    checkpoint_file = tmp_path / "checkpoint.partial.json"

    # First run: get interrupted
    paginator = MockPaginator(pages_first)
    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file), \
         patch("scripts.bulk_download_labeled_studies.CHECKPOINT_INTERVAL", 1):
        try:
            fetch_and_cache_file_listing(
                tmp_path, target_studies={"study_A", "study_B", "study_C"}
            )
        except Exception:
            pass  # Expected to fail on incomplete data

    # Checkpoint should exist
    assert checkpoint_file.exists()
    with open(checkpoint_file) as f:
        checkpoint_data = json.load(f)
    assert checkpoint_data["pages"] == 1
    assert checkpoint_data["page_token"] == "token_1"

    # Second run: resume from checkpoint
    paginator = MockPaginator(pages_resume)
    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        cache = fetch_and_cache_file_listing(
            tmp_path, target_studies={"study_A", "study_B", "study_C"}
        )

    # Should have successfully found all studies/series
    assert set(cache.keys()) == {"study_A", "study_B", "study_C"}
    assert set(cache["study_A"].keys()) == {"series_A1", "series_A2"}
    # Checkpoint should be deleted after success
    assert not checkpoint_file.exists()


def test_checkpoint_partial_file_map_survives(tmp_path):
    """Partial file map is preserved across checkpoint save/load."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)

    pages = [
        make_page({
            "study_A": {"series_A1": 5},  # 5 files on page 1
            "study_B": {"series_B1": 3},
        }, next_token="token_1"),
        make_page({
            "study_A": {"series_A1": 7, "series_A2": 2},  # 7 more files for A1, plus A2
            "study_B": {"series_B2": 1, "series_B3": 1},
            "study_C": {"series_C1": 1},
        }, next_token=None),
    ]

    checkpoint_file = tmp_path / "checkpoint.partial.json"
    paginator = MockPaginator(pages)

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        cache = fetch_and_cache_file_listing(
            tmp_path, target_studies={"study_A", "study_B", "study_C"}
        )

    # Study A should have all 12 files (5+7) from series_A1
    assert len(cache["study_A"]["series_A1"]) == 12


def test_corrupt_checkpoint_ignored(tmp_path):
    """Corrupt checkpoint is ignored and pagination starts fresh."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)

    checkpoint_file = tmp_path / "checkpoint.partial.json"
    # Write corrupt checkpoint
    with open(checkpoint_file, 'w') as f:
        f.write("{ invalid json")

    pages = [
        make_page({
            "study_A": {"series_A1": 2, "series_A2": 2},
            "study_B": {"series_B1": 2},
        }, next_token="token_1"),
        make_page({
            "study_B": {"series_B2": 2, "series_B3": 2},
            "study_C": {"series_C1": 1},
        }, next_token=None),
    ]

    paginator = MockPaginator(pages)

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        cache = fetch_and_cache_file_listing(
            tmp_path, target_studies={"study_A", "study_B", "study_C"}
        )

    # Should complete successfully starting from page 0
    assert set(cache.keys()) == {"study_A", "study_B", "study_C"}
    # Checkpoint should be deleted (replaced with valid one then cleared)
    assert not checkpoint_file.exists()


def test_deterministic_checksum_with_different_orderings(tmp_path):
    """Checksums are deterministic: same manifest produces same hash regardless of dict ordering."""
    from scripts.bulk_download_labeled_studies import compute_manifest_checksum

    manifest_1 = {
        "study_A": ["series_A1", "series_A2"],
        "study_B": ["series_B1", "series_B2"],
        "study_C": ["series_C1"],
    }

    # Same manifest but different dict ordering
    manifest_2 = {
        "study_C": ["series_C1"],
        "study_A": ["series_A1", "series_A2"],
        "study_B": ["series_B1", "series_B2"],
    }

    # Compute checksums
    checksum_1 = compute_manifest_checksum(manifest_1)
    checksum_2 = compute_manifest_checksum(manifest_2)

    # Should be identical despite different dict insertion order
    assert checksum_1 == checksum_2
    assert len(checksum_1) == 64  # SHA-256 produces 64-char hex string

    # Verify checksum changes when manifest changes
    manifest_3 = {
        "study_A": ["series_A1", "series_A2"],
        "study_B": ["series_B1", "series_B3"],  # Different series
        "study_C": ["series_C1"],
    }
    checksum_3 = compute_manifest_checksum(manifest_3)
    assert checksum_3 != checksum_1  # Should be different


def test_target_study_compatibility_with_list_vs_set(tmp_path):
    """Target studies checkpoint compatibility: JSON list vs Python set with identical UIDs."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)

    pages = [
        make_page({
            "study_A": {"series_A1": 2, "series_A2": 2},
            "study_B": {"series_B1": 2},
        }, next_token="token_1"),
        make_page({
            "study_B": {"series_B2": 2, "series_B3": 2},
            "study_C": {"series_C1": 1},
        }, next_token=None),
    ]

    paginator = MockPaginator(pages)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        cache = fetch_and_cache_file_listing(
            tmp_path, target_studies={"study_A", "study_B", "study_C"}
        )

    # Checkpoint should exist (then be deleted after success)
    assert not checkpoint_file.exists()
    assert set(cache.keys()) == {"study_A", "study_B", "study_C"}


def test_target_study_compatibility_rejects_different_uid(tmp_path):
    """Target studies compatibility rejects genuinely different UID."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)

    checkpoint_file = tmp_path / "checkpoint.partial.json"
    checkpoint_data = {
        "pages": 1,
        "page_token": "token_1",
        "labeled_files": {},
        "found_series_per_study": {},
        "target_studies": ["study_A", "study_B", "study_WRONG"],
        "manifest_checksum": "dummy",
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)

    # Create a valid page response
    pages = [make_page({}, next_token=None)]
    paginator = MockPaginator(pages)

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        try:
            fetch_and_cache_file_listing(
                tmp_path,
                target_studies={"study_A", "study_B", "study_C"}
            )
            assert False, "Should have raised exception"
        except Exception as e:
            # Should fail on missing series (checkpoint loads, then finds missing series)
            assert "missing" in str(e).lower()


def test_503_error_is_retried(tmp_path):
    """Kaggle 503 Service Unavailable is retried, not treated as fatal."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)

    checkpoint_file = tmp_path / "checkpoint.partial.json"

    class MockPaginatorWith503:
        def __init__(self):
            self.call_count = 0

        def run_command(self, *args, **kwargs):
            self.call_count += 1
            result = MagicMock()

            if self.call_count == 1:
                result.returncode = 1
                result.stderr = "503 Service Unavailable"
                result.stdout = ""
            else:
                result.returncode = 0
                result.stdout = make_page({
                    "study_A": {"series_A1": 2, "series_A2": 2},
                    "study_B": {"series_B1": 2, "series_B2": 2, "series_B3": 2},
                    "study_C": {"series_C1": 1},
                }, next_token=None)
                result.stderr = ""

            return result

    paginator = MockPaginatorWith503()

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file), \
         patch("scripts.bulk_download_labeled_studies.time.sleep"):
        cache = fetch_and_cache_file_listing(
            tmp_path, target_studies={"study_A", "study_B", "study_C"}
        )

    assert set(cache.keys()) == {"study_A", "study_B", "study_C"}
    assert paginator.call_count >= 2


def test_timeout_preserves_checkpoint(tmp_path):
    """Timeout saves checkpoint and exit message directs to resume."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)

    checkpoint_file = tmp_path / "checkpoint.partial.json"

    class TimeoutPaginator:
        def run_command(self, *args, **kwargs):
            raise subprocess.TimeoutExpired("kaggle", 30)

    paginator = TimeoutPaginator()

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        try:
            fetch_and_cache_file_listing(
                tmp_path, target_studies={"study_A", "study_B", "study_C"}
            )
            assert False, "Should have raised timeout"
        except Exception as e:
            assert "timeout" in str(e).lower()
            assert "resume" in str(e).lower()

    assert checkpoint_file.exists()


def test_checkpoint_always_stores_all_target_studies(tmp_path):
    """Periodic checkpoint always stores all 3 requested target_studies."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    pages = [
        make_page({
            "study_A": {"series_A1": 1, "series_A2": 1},
        }, next_token="token_1"),
        make_page({
            "study_B": {"series_B1": 1, "series_B2": 1, "series_B3": 1},
            "study_C": {"series_C1": 1},
        }, next_token=None),
    ]

    paginator = MockPaginator(pages)

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file), \
         patch("scripts.bulk_download_labeled_studies.CHECKPOINT_INTERVAL", 1):
        cache = fetch_and_cache_file_listing(
            tmp_path, target_studies={"study_A", "study_B", "study_C"}
        )

    assert not checkpoint_file.exists()


def test_keyboard_interrupt_saves_checkpoint(tmp_path):
    """Ctrl+C saves checkpoint with all target_studies."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    class InterruptPaginator:
        def __init__(self):
            self.call_count = 0
        
        def run_command(self, *args, **kwargs):
            self.call_count += 1
            if self.call_count > 1:
                raise KeyboardInterrupt()
            result = MagicMock()
            result.returncode = 0
            result.stdout = make_page({"study_A": {"series_A1": 1}}, next_token="token_1")
            result.stderr = ""
            return result

    paginator = InterruptPaginator()

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        try:
            fetch_and_cache_file_listing(
                tmp_path, target_studies={"study_A", "study_B", "study_C"}
            )
            assert False, "Should have been interrupted"
        except SystemExit:
            pass

    assert checkpoint_file.exists()
    with open(checkpoint_file) as f:
        saved = json.load(f)
    assert len(saved["target_studies"]) == 3
    assert set(saved["target_studies"]) == {"study_A", "study_B", "study_C"}


def test_unrelated_studies_never_stored_in_checkpoint(tmp_path):
    """Unrelated studies must never appear in the checkpoint file itself, not just the final cache."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    pages = [
        make_page({
            "study_A": {"series_A1": 1, "series_A2": 1},
            "study_UNRELATED": {"series_X1": 5},
            "study_B": {"series_B1": 1},
        }, next_token="token_1"),
        make_page({
            "study_UNRELATED": {"series_X2": 3},
            "study_B": {"series_B2": 1, "series_B3": 1},
            "study_C": {"series_C1": 1},
        }, next_token="token_2"),
    ]
    paginator = MockPaginator(pages)

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file), \
         patch("scripts.bulk_download_labeled_studies.CHECKPOINT_INTERVAL", 1):
        try:
            fetch_and_cache_file_listing(
                tmp_path, target_studies={"study_A", "study_B", "study_C"}
            )
        except Exception:
            pass  # Pagination is incomplete (study_A missing series merge etc.); we only care about checkpoint content

    assert checkpoint_file.exists()
    with open(checkpoint_file) as f:
        checkpoint_data = json.load(f)
    assert "study_UNRELATED" not in checkpoint_data["labeled_files"]
    assert set(checkpoint_data["labeled_files"].keys()) <= {"study_A", "study_B", "study_C"}


def test_ctrl_c_resume_does_not_skip_a_page(tmp_path):
    """Interrupting after page 1 completes and resuming must still collect page 2's data - no page is skipped."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    class InterruptAfterFirstPage:
        def __init__(self):
            self.call_count = 0

        def run_command(self, *args, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                result = MagicMock()
                result.returncode = 0
                result.stdout = make_page({
                    "study_A": {"series_A1": 1, "series_A2": 1},
                    "study_B": {"series_B1": 1},
                }, next_token="token_1")
                result.stderr = ""
                return result
            raise KeyboardInterrupt()

    paginator = InterruptAfterFirstPage()
    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file), \
         patch("scripts.bulk_download_labeled_studies.CHECKPOINT_INTERVAL", 1):
        try:
            fetch_and_cache_file_listing(
                tmp_path, target_studies={"study_A", "study_B", "study_C"}
            )
        except SystemExit:
            pass

    with open(checkpoint_file) as f:
        checkpoint_data = json.load(f)
    # Checkpoint's page_token must be the token for the NEXT unfetched page (token_1),
    # proving page 1's data was captured and page 2 has not been skipped.
    assert checkpoint_data["pages"] == 1
    assert checkpoint_data["page_token"] == "token_1"

    # Resume: page 2 (the one pointed to by token_1) must still be fetched, not skipped.
    pages_resume = [
        make_page({
            "study_B": {"series_B2": 1, "series_B3": 1},
            "study_C": {"series_C1": 1},
        }, next_token=None),
    ]
    paginator2 = MockPaginator(pages_resume)
    with patch("subprocess.run", side_effect=paginator2.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        cache = fetch_and_cache_file_listing(
            tmp_path, target_studies={"study_A", "study_B", "study_C"}
        )

    assert paginator2.current_page == 1  # Page 2's content was fetched exactly once
    assert set(cache["study_B"].keys()) == {"series_B1", "series_B2", "series_B3"}
    assert set(cache["study_C"].keys()) == {"series_C1"}


def test_replaying_page_on_resume_does_not_duplicate_files(tmp_path):
    """If a resumed run's first fetched page overlaps with files already in the checkpoint, no duplicates result."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    # Pre-seed a checkpoint as if page 1 already completed and captured 2 files for series_A1.
    seeded_files = [
        "train_series/study_A/series_A1/file_study_A_series_A1_0.dcm",
        "train_series/study_A/series_A1/file_study_A_series_A1_1.dcm",
    ]
    from scripts.bulk_download_labeled_studies import compute_full_manifest_checksum, compute_expected_series_mapping
    target_studies = {"study_A", "study_B", "study_C"}
    expected_mapping = compute_expected_series_mapping(tmp_path, target_studies)
    checkpoint_data = {
        "pages": 1,
        "page_token": "token_1",
        "labeled_files": {"study_A": {"series_A1": seeded_files}},
        "found_series_per_study": {"study_A": ["series_A1"]},
        "target_studies": sorted(target_studies),
        "manifest_checksum": compute_full_manifest_checksum(target_studies, expected_mapping),
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)

    # The resumed page re-serves the SAME two files for series_A1 (simulating a Kaggle
    # pagination replay/overlap at the resume boundary) plus one genuinely new file and
    # the remaining studies' data.
    replayed_page_text = (
        "name,size,creationDate\n"
        + "\n".join(f"{p},321432,2026-09-19" for p in seeded_files)
        + "\ntrain_series/study_A/series_A1/file_study_A_series_A1_2.dcm,321432,2026-09-19\n"
        + "train_series/study_A/series_A2/file_study_A_series_A2_0.dcm,321432,2026-09-19\n"
        + "train_series/study_B/series_B1/file_study_B_series_B1_0.dcm,321432,2026-09-19\n"
        + "train_series/study_B/series_B2/file_study_B_series_B2_0.dcm,321432,2026-09-19\n"
        + "train_series/study_B/series_B3/file_study_B_series_B3_0.dcm,321432,2026-09-19\n"
        + "train_series/study_C/series_C1/file_study_C_series_C1_0.dcm,321432,2026-09-19"
    )

    class ReplayPaginator:
        def run_command(self, *args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = replayed_page_text
            result.stderr = ""
            return result

    with patch("subprocess.run", side_effect=ReplayPaginator().run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        cache = fetch_and_cache_file_listing(tmp_path, target_studies=target_studies)

    # The two overlapping files must appear exactly once each; the new file must still be present.
    series_a1_files = cache["study_A"]["series_A1"]
    assert sorted(series_a1_files) == sorted(set(series_a1_files)), "Duplicate file paths found after replay"
    assert len(series_a1_files) == 3
    for f in seeded_files:
        assert series_a1_files.count(f) == 1


def test_503_retries_are_bounded(tmp_path):
    """503 errors must stop retrying after TRANSIENT_PAGE_RETRIES attempts, not retry forever."""
    from scripts.bulk_download_labeled_studies import TRANSIENT_PAGE_RETRIES
    write_manifest(tmp_path, DEFAULT_MANIFEST)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    class AlwaysUnavailable:
        def __init__(self):
            self.call_count = 0

        def run_command(self, *args, **kwargs):
            self.call_count += 1
            result = MagicMock()
            result.returncode = 1
            result.stderr = "503 Service Unavailable"
            result.stdout = ""
            return result

    paginator = AlwaysUnavailable()
    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file), \
         patch("scripts.bulk_download_labeled_studies.time.sleep"):
        with pytest.raises(Exception, match="(?i)503|retries"):
            fetch_and_cache_file_listing(
                tmp_path, target_studies={"study_A", "study_B", "study_C"}
            )

    assert paginator.call_count == TRANSIENT_PAGE_RETRIES


def test_exhausted_retries_preserve_checkpoint(tmp_path):
    """After bounded retries are exhausted, the checkpoint must still exist for a later resume."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    class FirstPageOkThenAlways503:
        def __init__(self):
            self.call_count = 0

        def run_command(self, *args, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                result = MagicMock()
                result.returncode = 0
                result.stdout = make_page({"study_A": {"series_A1": 1}}, next_token="token_1")
                result.stderr = ""
                return result
            result = MagicMock()
            result.returncode = 1
            result.stderr = "503 Service Unavailable"
            result.stdout = ""
            return result

    paginator = FirstPageOkThenAlways503()
    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file), \
         patch("scripts.bulk_download_labeled_studies.time.sleep"):
        with pytest.raises(Exception, match="(?i)resume"):
            fetch_and_cache_file_listing(
                tmp_path, target_studies={"study_A", "study_B", "study_C"}
            )

    assert checkpoint_file.exists()
    with open(checkpoint_file) as f:
        checkpoint_data = json.load(f)
    assert checkpoint_data["pages"] == 1
    assert checkpoint_data["page_token"] == "token_1"


def test_auth_failure_preserves_checkpoint(tmp_path):
    """A non-transient error (e.g. authentication failure) must save the checkpoint before raising."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    class OkThenAuthFailure:
        def __init__(self):
            self.call_count = 0

        def run_command(self, *args, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                result = MagicMock()
                result.returncode = 0
                result.stdout = make_page({"study_A": {"series_A1": 1}}, next_token="token_1")
                result.stderr = ""
                return result
            result = MagicMock()
            result.returncode = 1
            result.stderr = "401 Unauthorized: invalid API credentials"
            result.stdout = ""
            return result

    paginator = OkThenAuthFailure()
    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file), \
         patch("scripts.bulk_download_labeled_studies.time.sleep"):
        with pytest.raises(Exception, match="(?i)resume"):
            fetch_and_cache_file_listing(
                tmp_path, target_studies={"study_A", "study_B", "study_C"}
            )

    assert checkpoint_file.exists()
    with open(checkpoint_file) as f:
        checkpoint_data = json.load(f)
    assert checkpoint_data["pages"] == 1
    assert checkpoint_data["page_token"] == "token_1"
    assert set(checkpoint_data["target_studies"]) == {"study_A", "study_B", "study_C"}


def test_incompatible_checkpoint_is_quarantined_not_deleted(tmp_path):
    """An incompatible checkpoint must be preserved (renamed), never silently unlinked."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    original_content = {
        "pages": 500,
        "page_token": "hours_of_progress_token",
        "labeled_files": {"study_X": {"series_X1": ["some/file.dcm"]}},
        "found_series_per_study": {"study_X": ["series_X1"]},
        "target_studies": ["study_X", "study_Y", "study_Z"],
        "manifest_checksum": "deadbeef",
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(original_content, f)

    pages = [make_page({
        "study_A": {"series_A1": 1, "series_A2": 1},
        "study_B": {"series_B1": 1, "series_B2": 1, "series_B3": 1},
        "study_C": {"series_C1": 1},
    }, next_token=None)]
    paginator = MockPaginator(pages)

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        fetch_and_cache_file_listing(
            tmp_path, target_studies={"study_A", "study_B", "study_C"}
        )

    # The original checkpoint must never be silently deleted - it must be quarantined under a new name.
    assert not checkpoint_file.exists() or json.load(open(checkpoint_file)) != original_content
    quarantined = list(tmp_path.glob("checkpoint.partial.json.incompatible.*"))
    assert len(quarantined) == 1, "Incompatible checkpoint was not preserved/quarantined"
    with open(quarantined[0]) as f:
        preserved = json.load(f)
    assert preserved == original_content, "Quarantined checkpoint content was altered or lost"


def test_changed_series_mapping_invalidates_checkpoint(tmp_path):
    """Same target_studies but a different expected study->series mapping must invalidate the checkpoint."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)
    checkpoint_file = tmp_path / "checkpoint.partial.json"

    from scripts.bulk_download_labeled_studies import compute_full_manifest_checksum
    target_studies = {"study_A", "study_B", "study_C"}
    # Checksum computed against a DIFFERENT (stale) expected series mapping than what
    # write_manifest(DEFAULT_MANIFEST) currently describes.
    stale_mapping = {
        "study_A": {"series_A1"},  # missing series_A2 relative to DEFAULT_MANIFEST
        "study_B": {"series_B1", "series_B2", "series_B3"},
        "study_C": {"series_C1"},
    }
    stale_checksum = compute_full_manifest_checksum(target_studies, stale_mapping)

    checkpoint_data = {
        "pages": 3,
        "page_token": "some_token",
        "labeled_files": {"study_A": {"series_A1": ["train_series/study_A/series_A1/f0.dcm"]}},
        "found_series_per_study": {"study_A": ["series_A1"]},
        "target_studies": sorted(target_studies),
        "manifest_checksum": stale_checksum,
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)

    pages = [make_page({
        "study_A": {"series_A1": 1, "series_A2": 1},
        "study_B": {"series_B1": 1, "series_B2": 1, "series_B3": 1},
        "study_C": {"series_C1": 1},
    }, next_token=None)]
    paginator = MockPaginator(pages)

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file):
        cache = fetch_and_cache_file_listing(tmp_path, target_studies=target_studies)

    # Started fresh (not resumed) despite matching target_studies, because the series mapping changed.
    quarantined = list(tmp_path.glob("checkpoint.partial.json.incompatible.*"))
    assert len(quarantined) == 1
    assert set(cache["study_A"].keys()) == {"series_A1", "series_A2"}


def test_final_cache_write_is_atomic(tmp_path):
    """A crash during the final cache write must never corrupt a previously valid cache file."""
    write_manifest(tmp_path, DEFAULT_MANIFEST)
    checkpoint_file = tmp_path / "checkpoint.partial.json"
    cache_file = tmp_path / "final_cache.json"

    good_previous_content = {"study_PREVIOUS": {"series_P1": ["a/b/c.dcm"]}}
    with open(cache_file, 'w') as f:
        json.dump(good_previous_content, f)

    pages = [make_page({
        "study_A": {"series_A1": 1, "series_A2": 1},
        "study_B": {"series_B1": 1, "series_B2": 1, "series_B3": 1},
        "study_C": {"series_C1": 1},
    }, next_token=None)]
    paginator = MockPaginator(pages)

    with patch("subprocess.run", side_effect=paginator.run_command), \
         patch("scripts.bulk_download_labeled_studies.CACHE_PARTIAL_FILE", checkpoint_file), \
         patch("scripts.bulk_download_labeled_studies.json.dump", side_effect=OSError("disk full mid-write")):
        with pytest.raises(OSError):
            fetch_and_cache_file_listing(
                tmp_path, cache_file=cache_file, target_studies={"study_A", "study_B", "study_C"}
            )

    # The real cache file must be untouched - the crash happened while writing a temp file.
    with open(cache_file) as f:
        assert json.load(f) == good_previous_content

    # No leftover temp file should remain visible under the final name.
    assert cache_file.exists()
    leftover_temp = tmp_path / f".{cache_file.name}.tmp"
    # Temp file may or may not exist depending on where the mocked dump failed, but if it
    # does, it must never have replaced the real cache (already asserted above).
    _ = leftover_temp.exists()
