"""End-to-end tests for RSNA DICOM processing pipeline.

Tests loading RSNA DICOM data and running it through the full pipeline:
DICOM → volume → preprocess → segment → features → decision report.
"""

from pathlib import Path

import pytest

from mri_core.rsna_integration import (
    discover_rsna_root,
    load_rsna_metadata_safe,
    discover_available_studies,
    get_available_planes,
    load_rsna_study_series,
    RSNADiscoveryError,
)
from mri_core.pipeline import process_mri_image
from mri_core.decision import generate_decision_report, QualityStatus


pytestmark = pytest.mark.rsna


ROOT = Path(__file__).resolve().parents[1]
RSNA_ROOT = ROOT / "data" / "rsna-knee"


class TestRSNAPipelineEndToEnd:
    """Test complete RSNA processing pipeline."""

    @pytest.fixture(scope="class")
    @staticmethod
    def rsna_data():
        """Load RSNA data for testing."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        try:
            metadata = load_rsna_metadata_safe(RSNA_ROOT)
            available_studies = discover_available_studies(RSNA_ROOT, metadata)

            if not available_studies:
                pytest.skip("No downloaded RSNA studies available")

            study_uid = available_studies[0]
            planes = get_available_planes(RSNA_ROOT, metadata, study_uid)

            if not planes:
                pytest.skip(f"Study {study_uid} has no available planes")

            return {
                "metadata": metadata,
                "study_uid": study_uid,
                "planes": planes,
                "rsna_root": RSNA_ROOT,
            }
        except RSNADiscoveryError as e:
            pytest.skip(f"RSNA discovery failed: {e}")

    def test_load_rsna_volume_shape(self, rsna_data):
        """Loaded RSNA volume has expected shape."""
        volume, meta = load_rsna_study_series(
            rsna_data["rsna_root"],
            rsna_data["metadata"],
            rsna_data["study_uid"],
            rsna_data["planes"][0],
        )

        # Volume should be 3D (slices, height, width)
        assert volume.raw_data.ndim == 3
        assert volume.num_slices > 0
        assert volume.raw_data.shape[1] > 0  # height
        assert volume.raw_data.shape[2] > 0  # width

    def test_load_rsna_volume_metadata(self, rsna_data):
        """Loaded RSNA volume has expected metadata."""
        volume, meta = load_rsna_study_series(
            rsna_data["rsna_root"],
            rsna_data["metadata"],
            rsna_data["study_uid"],
            rsna_data["planes"][0],
        )

        assert meta["study_uid"] == rsna_data["study_uid"]
        assert "series_uid" in meta
        assert meta["plane"] in ("Axial", "Coronal", "Sagittal")
        assert meta["num_slices"] == volume.num_slices
        assert volume.format_type == "DICOM"

    def test_process_rsna_slice(self, rsna_data):
        """Process single RSNA slice through pipeline."""
        volume, _ = load_rsna_study_series(
            rsna_data["rsna_root"],
            rsna_data["metadata"],
            rsna_data["study_uid"],
            rsna_data["planes"][0],
        )

        # Get middle slice
        slice_idx = volume.default_slice_index
        display_slice = volume.get_display_slice(slice_idx)

        # Process through pipeline
        results = process_mri_image(
            display_slice,
            segmentation_method="otsu",
            is_mri=True,
            is_inverted=volume.is_inverted,
        )

        # Verify all results present
        assert "original" in results
        assert "preprocessed" in results
        assert "mask" in results
        assert "overlay" in results
        assert "features" in results

        # Verify array types and shapes
        assert results["preprocessed"].ndim == 2  # Grayscale
        assert results["mask"].dtype == results["preprocessed"].dtype
        assert results["features"]["image_height"] > 0
        assert results["features"]["foreground_pixels"] >= 0

    def test_generate_report_from_rsna(self, rsna_data):
        """Generate decision report from RSNA data."""
        volume, meta = load_rsna_study_series(
            rsna_data["rsna_root"],
            rsna_data["metadata"],
            rsna_data["study_uid"],
            rsna_data["planes"][0],
        )

        # Process slice
        slice_idx = volume.default_slice_index
        display_slice = volume.get_display_slice(slice_idx)
        results = process_mri_image(
            display_slice,
            segmentation_method="otsu",
            is_mri=True,
            is_inverted=volume.is_inverted,
        )

        # Generate report
        report = generate_decision_report(
            study_uid=meta["study_uid"],
            series_uid=meta["series_uid"],
            plane=meta["plane"],
            volume_data=volume.raw_data,
            preprocessed_data=results["preprocessed"],
            mask=results["mask"],
            features=results["features"],
            num_slices=meta["num_slices"],
        )

        # Verify report structure
        assert report.study_uid == meta["study_uid"]
        assert report.plane == meta["plane"]
        assert report.num_slices == meta["num_slices"]
        assert report.quality_status in (
            QualityStatus.OK,
            QualityStatus.REVIEW,
            QualityStatus.INVALID,
        )
        assert report.model_status is not None
        assert "checkpoint" in report.model_status.lower() or "not available" in report.model_status.lower()

    def test_all_planes_processable(self, rsna_data):
        """All available planes can be processed."""
        for plane in rsna_data["planes"]:
            volume, meta = load_rsna_study_series(
                rsna_data["rsna_root"],
                rsna_data["metadata"],
                rsna_data["study_uid"],
                plane,
            )
            assert volume is not None
            assert meta["plane"] == plane

            # Process at least one slice
            display_slice = volume.get_display_slice(volume.default_slice_index)
            results = process_mri_image(
                display_slice,
                segmentation_method="otsu",
                is_mri=True,
                is_inverted=volume.is_inverted,
            )
            assert "features" in results
            assert results["features"]["foreground_pixels"] >= 0
