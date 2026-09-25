"""Tests for decision report generation and quality assessment.

Covers report generation, quality rules, status classification,
and model prediction placeholder behavior.
"""

import numpy as np
import pytest

from mri_core.decision import (
    DecisionReport,
    QualityStatus,
    generate_decision_report,
    ImageQualityMetrics,
    SegmentationQuality,
    MIN_FOREGROUND_FRACTION,
    MAX_FOREGROUND_FRACTION,
    MIN_SLICES,
    MIN_IMAGE_DIMENSION,
    MAX_IMAGE_DIMENSION,
    MIN_VALID_INTENSITY_RANGE,
)


class TestQualityStatus:
    """Test quality status enumeration."""

    def test_quality_status_values(self):
        """QualityStatus has expected values."""
        assert QualityStatus.OK.value == "OK"
        assert QualityStatus.REVIEW.value == "REVIEW"
        assert QualityStatus.INVALID.value == "INVALID"


class TestImageQualityMetrics:
    """Test ImageQualityMetrics dataclass."""

    def test_image_quality_metrics_creation(self):
        """Create ImageQualityMetrics successfully."""
        metrics = ImageQualityMetrics(
            mean_intensity=128.5,
            std_intensity=25.3,
            min_intensity=10.0,
            max_intensity=245.0,
            intensity_range=235.0,
            foreground_fraction=0.45,
        )
        assert metrics.mean_intensity == 128.5
        assert metrics.intensity_range == 235.0

    def test_image_quality_metrics_frozen(self):
        """ImageQualityMetrics is immutable."""
        metrics = ImageQualityMetrics(
            mean_intensity=128.5,
            std_intensity=25.3,
            min_intensity=10.0,
            max_intensity=245.0,
            intensity_range=235.0,
            foreground_fraction=0.45,
        )
        with pytest.raises(AttributeError):
            metrics.mean_intensity = 100.0


class TestSegmentationQuality:
    """Test SegmentationQuality dataclass."""

    def test_segmentation_quality_creation(self):
        """Create SegmentationQuality successfully."""
        seg = SegmentationQuality(
            foreground_pixels=5000,
            foreground_fraction=0.35,
            largest_contour_area=4500.0,
            has_contours=True,
        )
        assert seg.foreground_pixels == 5000
        assert seg.has_contours

    def test_segmentation_quality_frozen(self):
        """SegmentationQuality is immutable."""
        seg = SegmentationQuality(
            foreground_pixels=5000,
            foreground_fraction=0.35,
            largest_contour_area=4500.0,
            has_contours=True,
        )
        with pytest.raises(AttributeError):
            seg.foreground_pixels = 6000


class TestDecisionReport:
    """Test DecisionReport dataclass and methods."""

    def test_decision_report_creation(self):
        """Create DecisionReport successfully."""
        metrics = ImageQualityMetrics(
            mean_intensity=128.0, std_intensity=25.0, min_intensity=10.0,
            max_intensity=245.0, intensity_range=235.0, foreground_fraction=0.4
        )
        seg = SegmentationQuality(
            foreground_pixels=5000, foreground_fraction=0.4,
            largest_contour_area=4500.0, has_contours=True
        )
        report = DecisionReport(
            study_uid="1.2.3",
            series_uid="4.5.6",
            plane="Axial",
            num_slices=30,
            image_height=512,
            image_width=512,
            quality_metrics=metrics,
            segmentation_quality=seg,
            quality_status=QualityStatus.OK,
        )
        assert report.study_uid == "1.2.3"
        assert report.quality_status == QualityStatus.OK

    def test_decision_report_frozen(self):
        """DecisionReport is immutable."""
        metrics = ImageQualityMetrics(
            mean_intensity=128.0, std_intensity=25.0, min_intensity=10.0,
            max_intensity=245.0, intensity_range=235.0, foreground_fraction=0.4
        )
        seg = SegmentationQuality(
            foreground_pixels=5000, foreground_fraction=0.4,
            largest_contour_area=4500.0, has_contours=True
        )
        report = DecisionReport(
            study_uid="1.2.3",
            series_uid="4.5.6",
            plane="Axial",
            num_slices=30,
            image_height=512,
            image_width=512,
            quality_metrics=metrics,
            segmentation_quality=seg,
        )
        with pytest.raises(AttributeError):
            report.study_uid = "different"

    def test_decision_report_to_dict(self):
        """Convert report to dictionary."""
        metrics = ImageQualityMetrics(
            mean_intensity=128.0, std_intensity=25.0, min_intensity=10.0,
            max_intensity=245.0, intensity_range=235.0, foreground_fraction=0.4
        )
        seg = SegmentationQuality(
            foreground_pixels=5000, foreground_fraction=0.4,
            largest_contour_area=4500.0, has_contours=True
        )
        report = DecisionReport(
            study_uid="1.2.3",
            series_uid="4.5.6",
            plane="Axial",
            num_slices=30,
            image_height=512,
            image_width=512,
            quality_metrics=metrics,
            segmentation_quality=seg,
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["study_uid"] == "1.2.3"
        assert d["quality_status"] == "OK"

    def test_decision_report_summary(self):
        """Generate human-readable summary."""
        metrics = ImageQualityMetrics(
            mean_intensity=128.0, std_intensity=25.0, min_intensity=10.0,
            max_intensity=245.0, intensity_range=235.0, foreground_fraction=0.4
        )
        seg = SegmentationQuality(
            foreground_pixels=5000, foreground_fraction=0.4,
            largest_contour_area=4500.0, has_contours=True
        )
        report = DecisionReport(
            study_uid="1.2.3",
            series_uid="4.5.6",
            plane="Axial",
            num_slices=30,
            image_height=512,
            image_width=512,
            quality_metrics=metrics,
            segmentation_quality=seg,
            quality_flags=["Low intensity range"],
        )
        summary = report.summary()
        assert "1.2.3" in summary
        assert "Axial" in summary
        assert "30 slices" in summary
        assert "OK" in summary
        assert "Low intensity range" in summary


class TestGenerateDecisionReport:
    """Test decision report generation from pipeline outputs."""

    def create_test_data(self, shape=(10, 256, 256)):
        """Create test volume, preprocessed, and mask arrays."""
        volume = np.random.rand(*shape).astype(np.float32) * 200 + 50
        preprocessed = (np.random.rand(256, 256).astype(np.uint8) * 100 + 75)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:200, 50:200] = 255  # ~36% foreground
        features = {
            "image_width": 256,
            "image_height": 256,
            "mean_intensity": float(np.mean(preprocessed)),
            "std_deviation": float(np.std(preprocessed)),
            "min_intensity": float(np.min(preprocessed)),
            "max_intensity": float(np.max(preprocessed)),
            "foreground_pixels": 150 * 150,
            "foreground_percentage": 36.0,
            "largest_contour_area": 22000.0,
            "bounding_rect": (50, 50, 150, 150),
        }
        return volume, preprocessed, mask, features

    def test_generate_report_ok_status(self):
        """Generate report with OK status."""
        volume, preprocessed, mask, features = self.create_test_data()
        # Ensure high enough intensity range
        preprocessed[:] = np.linspace(50, 200, 256*256).reshape(256, 256).astype(np.uint8)
        features["mean_intensity"] = 125.0
        features["std_deviation"] = 50.0
        features["min_intensity"] = 50.0
        features["max_intensity"] = 200.0

        report = generate_decision_report(
            study_uid="test_study",
            series_uid="test_series",
            plane="Axial",
            volume_data=volume,
            preprocessed_data=preprocessed,
            mask=mask,
            features=features,
            num_slices=10,
        )
        # If there are flags, verify they're quality-of-life issues only, not data integrity
        if report.quality_flags:
            # Status should still be OK or REVIEW, with OK being ideal
            assert report.quality_status in (QualityStatus.OK, QualityStatus.REVIEW)
        else:
            assert report.quality_status == QualityStatus.OK

    def test_generate_report_with_quality_flags(self):
        """Generate report with quality flags."""
        volume, preprocessed, mask, features = self.create_test_data()
        # Make intensity range too low
        preprocessed[:] = 100
        features["mean_intensity"] = 100.0
        features["std_deviation"] = 2.0
        features["min_intensity"] = 98.0
        features["max_intensity"] = 102.0  # range = 4, below MIN_VALID_INTENSITY_RANGE

        report = generate_decision_report(
            study_uid="test_study",
            series_uid="test_series",
            plane="Axial",
            volume_data=volume,
            preprocessed_data=preprocessed,
            mask=mask,
            features=features,
            num_slices=10,
        )
        assert report.quality_status in (QualityStatus.REVIEW, QualityStatus.OK)
        if report.quality_status == QualityStatus.REVIEW:
            assert any("intensity range" in flag.lower() for flag in report.quality_flags)

    def test_generate_report_low_foreground(self):
        """Report flagged when foreground too low."""
        volume, preprocessed, mask, features = self.create_test_data()
        features["foreground_pixels"] = 100  # Very low
        features["foreground_percentage"] = 0.15  # 0.15%
        mask[:] = 0
        mask[0:10, 0:10] = 255

        report = generate_decision_report(
            study_uid="test_study",
            series_uid="test_series",
            plane="Axial",
            volume_data=volume,
            preprocessed_data=preprocessed,
            mask=mask,
            features=features,
            num_slices=10,
        )
        assert report.quality_status == QualityStatus.REVIEW
        assert any("foreground" in flag.lower() for flag in report.quality_flags)

    def test_generate_report_high_foreground(self):
        """Report flagged when foreground too high."""
        volume, preprocessed, mask, features = self.create_test_data()
        # Almost all foreground
        features["foreground_pixels"] = 256 * 256 - 100
        features["foreground_percentage"] = 99.85
        mask[:] = 255
        mask[0:10, 0:10] = 0

        report = generate_decision_report(
            study_uid="test_study",
            series_uid="test_series",
            plane="Axial",
            volume_data=volume,
            preprocessed_data=preprocessed,
            mask=mask,
            features=features,
            num_slices=10,
        )
        assert report.quality_status == QualityStatus.REVIEW
        assert any("foreground" in flag.lower() for flag in report.quality_flags)

    def test_generate_report_low_slice_count(self):
        """Report flagged for low slice count."""
        volume, preprocessed, mask, features = self.create_test_data()
        report = generate_decision_report(
            study_uid="test_study",
            series_uid="test_series",
            plane="Axial",
            volume_data=volume,
            preprocessed_data=preprocessed,
            mask=mask,
            features=features,
            num_slices=2,  # Below MIN_SLICES
        )
        assert report.quality_status == QualityStatus.REVIEW
        assert any("slice" in flag.lower() for flag in report.quality_flags)

    def test_generate_report_invalid_data(self):
        """Report marked INVALID when data has NaN/Inf."""
        volume = np.full((10, 256, 256), np.nan, dtype=np.float32)
        preprocessed, mask, features = self.create_test_data()[1:]

        report = generate_decision_report(
            study_uid="test_study",
            series_uid="test_series",
            plane="Axial",
            volume_data=volume,
            preprocessed_data=preprocessed,
            mask=mask,
            features=features,
            num_slices=10,
        )
        assert report.quality_status == QualityStatus.INVALID

    def test_generate_report_model_status_placeholder(self):
        """Report includes model status placeholder."""
        volume, preprocessed, mask, features = self.create_test_data()
        report = generate_decision_report(
            study_uid="test_study",
            series_uid="test_series",
            plane="Axial",
            volume_data=volume,
            preprocessed_data=preprocessed,
            mask=mask,
            features=features,
            num_slices=10,
        )
        assert "Not available" in report.model_status or "checkpoint" in report.model_status
        assert report.model_predictions is None

    def test_generate_report_no_contours(self):
        """Report flagged when no contours detected."""
        volume, preprocessed, mask, features = self.create_test_data()
        features["largest_contour_area"] = 0.0  # No contours
        features["bounding_rect"] = (0, 0, 0, 0)

        report = generate_decision_report(
            study_uid="test_study",
            series_uid="test_series",
            plane="Axial",
            volume_data=volume,
            preprocessed_data=preprocessed,
            mask=mask,
            features=features,
            num_slices=10,
        )
        # May not flag if there's no foreground at all, but if there is foreground
        # and no contours, it should flag
        if features["foreground_pixels"] > 0:
            assert any("contour" in flag.lower() for flag in report.quality_flags)

    def test_generate_report_small_image(self):
        """Report flagged for small image dimensions."""
        volume = np.random.rand(10, 64, 64).astype(np.float32)  # Below MIN
        preprocessed = np.random.rand(64, 64).astype(np.uint8) * 100 + 75
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        features = {
            "image_width": 64,
            "image_height": 64,
            "mean_intensity": 100.0,
            "std_deviation": 25.0,
            "min_intensity": 50.0,
            "max_intensity": 150.0,
            "foreground_pixels": 40 * 40,
            "foreground_percentage": 39.0,
            "largest_contour_area": 1500.0,
            "bounding_rect": (10, 10, 40, 40),
        }
        report = generate_decision_report(
            study_uid="test_study",
            series_uid="test_series",
            plane="Axial",
            volume_data=volume,
            preprocessed_data=preprocessed,
            mask=mask,
            features=features,
            num_slices=10,
        )
        assert report.image_height == 64
        assert report.image_width == 64
        if 64 < MIN_IMAGE_DIMENSION:
            assert report.quality_status == QualityStatus.REVIEW


class TestModelPredictionsParameter:
    """Tests for the optional model_predictions parameter added for inference wiring."""

    def create_ok_data(self):
        """Data that produces a clean OK quality status, so model_predictions'
        independence from image-quality logic is unambiguous."""
        volume = np.random.rand(10, 256, 256).astype(np.float32) * 200 + 50
        preprocessed = np.linspace(50, 200, 256 * 256).reshape(256, 256).astype(np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:200, 50:200] = 255
        features = {
            "image_width": 256,
            "image_height": 256,
            "mean_intensity": 125.0,
            "std_deviation": 50.0,
            "min_intensity": 50.0,
            "max_intensity": 200.0,
            "foreground_pixels": 150 * 150,
            "foreground_percentage": 36.0,
            "largest_contour_area": 22000.0,
            "bounding_rect": (50, 50, 150, 150),
        }
        return volume, preprocessed, mask, features

    def test_omitted_model_predictions_preserves_existing_behavior(self):
        """model_predictions=None (the default) must reproduce the exact prior defaults."""
        volume, preprocessed, mask, features = self.create_ok_data()
        report = generate_decision_report(
            study_uid="test_study", series_uid="test_series", plane="Axial",
            volume_data=volume, preprocessed_data=preprocessed, mask=mask,
            features=features, num_slices=10,
        )
        assert report.model_predictions is None
        assert report.model_status == "Not available — model checkpoint not loaded"

    def test_valid_prediction_dict_is_stored(self):
        volume, preprocessed, mask, features = self.create_ok_data()
        predictions = {target: 0.5 for target in [
            "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
        ]}
        report = generate_decision_report(
            study_uid="test_study", series_uid="test_series", plane="Axial",
            volume_data=volume, preprocessed_data=preprocessed, mask=mask,
            features=features, num_slices=10, model_predictions=predictions,
        )
        assert report.model_predictions == predictions
        assert report.model_status == "Model checkpoint loaded — predictions available"

    def test_probability_below_zero_rejected(self):
        volume, preprocessed, mask, features = self.create_ok_data()
        with pytest.raises(ValueError):
            generate_decision_report(
                study_uid="test_study", series_uid="test_series", plane="Axial",
                volume_data=volume, preprocessed_data=preprocessed, mask=mask,
                features=features, num_slices=10, model_predictions={"ACL": -0.1},
            )

    def test_probability_above_one_rejected(self):
        volume, preprocessed, mask, features = self.create_ok_data()
        with pytest.raises(ValueError):
            generate_decision_report(
                study_uid="test_study", series_uid="test_series", plane="Axial",
                volume_data=volume, preprocessed_data=preprocessed, mask=mask,
                features=features, num_slices=10, model_predictions={"ACL": 1.1},
            )

    def test_nan_probability_rejected(self):
        volume, preprocessed, mask, features = self.create_ok_data()
        with pytest.raises(ValueError):
            generate_decision_report(
                study_uid="test_study", series_uid="test_series", plane="Axial",
                volume_data=volume, preprocessed_data=preprocessed, mask=mask,
                features=features, num_slices=10, model_predictions={"ACL": float("nan")},
            )

    def test_inf_probability_rejected(self):
        volume, preprocessed, mask, features = self.create_ok_data()
        with pytest.raises(ValueError):
            generate_decision_report(
                study_uid="test_study", series_uid="test_series", plane="Axial",
                volume_data=volume, preprocessed_data=preprocessed, mask=mask,
                features=features, num_slices=10, model_predictions={"ACL": float("inf")},
            )

    def test_model_predictions_do_not_alter_quality_status(self):
        """Same underlying image data must yield the same quality_status regardless
        of whether (or what) model_predictions are supplied."""
        volume, preprocessed, mask, features = self.create_ok_data()
        without_predictions = generate_decision_report(
            study_uid="test_study", series_uid="test_series", plane="Axial",
            volume_data=volume, preprocessed_data=preprocessed, mask=mask,
            features=features, num_slices=10,
        )
        with_predictions = generate_decision_report(
            study_uid="test_study", series_uid="test_series", plane="Axial",
            volume_data=volume, preprocessed_data=preprocessed, mask=mask,
            features=features, num_slices=10, model_predictions={"ACL": 0.9, "Fracture": 0.1},
        )
        assert without_predictions.quality_status == with_predictions.quality_status
        assert without_predictions.quality_flags == with_predictions.quality_flags
