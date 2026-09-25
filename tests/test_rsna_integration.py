"""Tests for RSNA dataset integration and discovery.

Covers RSNA metadata loading, study discovery, plane availability,
and safe handling of partial/incomplete downloads.
"""

import json
from pathlib import Path
from typing import Dict

import pytest

from mri_core.rsna_integration import (
    discover_rsna_root,
    load_rsna_metadata_safe,
    discover_available_studies,
    get_available_planes,
    load_rsna_study_series,
    RSNADiscoveryError,
    RSNAStudyNotAvailable,
)


pytestmark = pytest.mark.rsna


ROOT = Path(__file__).resolve().parents[1]
RSNA_ROOT = ROOT / "data" / "rsna-knee"


class TestRSNADiscovery:
    """Test RSNA root discovery."""

    def test_discover_rsna_root_finds_default_location(self):
        """Discover RSNA root at data/rsna-knee."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        discovered = discover_rsna_root(explicit_root=RSNA_ROOT)
        assert discovered == RSNA_ROOT
        assert (discovered / "train.csv").is_file()

    def test_discover_rsna_root_fails_when_missing(self, tmp_path):
        """RSNADiscoveryError when root doesn't exist."""
        with pytest.raises(RSNADiscoveryError):
            discover_rsna_root(explicit_root=tmp_path / "nonexistent")

    def test_explicit_root_takes_precedence(self):
        """Explicit root parameter is used when provided."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        discovered = discover_rsna_root(explicit_root=RSNA_ROOT)
        assert discovered == RSNA_ROOT


class TestMetadataLoading:
    """Test RSNA metadata CSV loading."""

    def test_load_rsna_metadata_succeeds(self):
        """Load RSNA metadata CSVs successfully."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        assert metadata is not None
        assert len(metadata.train) > 0
        assert len(metadata.train_series) > 0

    def test_load_rsna_metadata_fails_on_missing_files(self, tmp_path):
        """RSNADiscoveryError when required CSV files are missing."""
        with pytest.raises(RSNADiscoveryError):
            load_rsna_metadata_safe(tmp_path)

    def test_rsna_metadata_has_expected_columns(self):
        """Loaded metadata has expected columns."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        assert "StudyInstanceUID" in metadata.train.columns
        assert "ACL" in metadata.train.columns  # One of 12 targets
        assert "Anatomical_Plane" in metadata.train_series.columns


class TestStudyDiscovery:
    """Test discovery of locally available studies."""

    def test_discover_available_studies_returns_list(self):
        """discover_available_studies returns list of UIDs."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        available = discover_available_studies(RSNA_ROOT, metadata)

        assert isinstance(available, list)
        # May be empty if no downloads yet
        if available:
            assert all(isinstance(uid, str) for uid in available)

    def test_discover_available_studies_skips_missing_dicom_dirs(self):
        """Studies without local DICOM directories are not listed."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        available = discover_available_studies(RSNA_ROOT, metadata)

        # Verify each returned study has at least one series directory
        train_series_root = RSNA_ROOT / "raw" / "train_series"
        for study_uid in available:
            study_dir = train_series_root / study_uid
            assert study_dir.is_dir(), f"Study {study_uid} missing {study_dir}"
            assert any(p.is_dir() for p in study_dir.iterdir()), (
                f"Study {study_uid} has no series directories"
            )

    def test_discover_returns_empty_when_no_downloads(self, tmp_path):
        """Returns empty list when no downloads present."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)

        # Create fake RSNA structure with no downloads
        fake_raw = tmp_path / "raw" / "train_series"
        fake_raw.mkdir(parents=True)

        available = discover_available_studies(tmp_path, metadata)
        assert available == []


class TestPlaneAvailability:
    """Test plane availability detection."""

    def test_get_available_planes_returns_list(self):
        """get_available_planes returns list of plane names."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        available_studies = discover_available_studies(RSNA_ROOT, metadata)

        if not available_studies:
            pytest.skip("No downloaded studies available")

        study_uid = available_studies[0]
        planes = get_available_planes(RSNA_ROOT, metadata, study_uid)

        assert isinstance(planes, list)
        if planes:
            assert all(p in ("Axial", "Coronal", "Sagittal") for p in planes)

    def test_get_available_planes_handles_missing_study(self):
        """Returns empty list for study with no series directories."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)

        # Pick a study from metadata that likely has no downloads
        all_studies = metadata.train["StudyInstanceUID"].astype(str).tolist()
        # Try the first few to find one that's not downloaded
        for study_uid in all_studies[:5]:
            study_dir = RSNA_ROOT / "raw" / "train_series" / study_uid
            if not study_dir.is_dir():
                planes = get_available_planes(RSNA_ROOT, metadata, study_uid)
                assert planes == []
                break


class TestSeriesLoading:
    """Test loading RSNA series into MRIVolume."""

    def test_load_series_succeeds_when_available(self):
        """Load series successfully when DICOM data is present."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        available_studies = discover_available_studies(RSNA_ROOT, metadata)

        if not available_studies:
            pytest.skip("No downloaded studies available")

        study_uid = available_studies[0]
        planes = get_available_planes(RSNA_ROOT, metadata, study_uid)

        if not planes:
            pytest.skip(f"Study {study_uid} has no available planes")

        plane = planes[0]
        volume, meta = load_rsna_study_series(RSNA_ROOT, metadata, study_uid, plane)

        assert volume is not None
        assert volume.num_slices > 0
        assert meta["study_uid"] == study_uid
        assert meta["plane"] == plane

    def test_load_series_fails_when_not_available(self):
        """RSNAStudyNotAvailable when series not downloaded."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)

        # Find a study from metadata that's definitely not downloaded
        all_studies = metadata.train["StudyInstanceUID"].astype(str).tolist()
        downloaded_studies = discover_available_studies(RSNA_ROOT, metadata)

        for study_uid in all_studies:
            if study_uid not in downloaded_studies:
                with pytest.raises(RSNAStudyNotAvailable):
                    load_rsna_study_series(RSNA_ROOT, metadata, study_uid, "Axial")
                break

    def test_load_series_deterministic(self):
        """Same series selected on repeated calls."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        available_studies = discover_available_studies(RSNA_ROOT, metadata)

        if not available_studies:
            pytest.skip("No downloaded studies available")

        study_uid = available_studies[0]
        planes = get_available_planes(RSNA_ROOT, metadata, study_uid)

        if not planes:
            pytest.skip(f"Study {study_uid} has no available planes")

        plane = planes[0]

        # Load same study/plane twice
        _, meta1 = load_rsna_study_series(RSNA_ROOT, metadata, study_uid, plane)
        _, meta2 = load_rsna_study_series(RSNA_ROOT, metadata, study_uid, plane)

        assert meta1["series_uid"] == meta2["series_uid"]
