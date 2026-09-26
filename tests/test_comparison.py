"""Synthetic tests for the ChangeGuard deterministic comparison layer.

All tests are self-contained; no real dataset, DICOM files, or network access
is required.  Fixtures build minimal DecisionReport objects from known values.
"""

import numpy as np
import pytest

from mri_core.comparison import (
    ComparisonReport,
    ComparisonStatus,
    FeatureDelta,
    FOREGROUND_FRACTION_REVIEW_THRESHOLD,
    MEAN_INTENSITY_REVIEW_THRESHOLD,
    generate_comparison_report,
)
from mri_core.decision import (
    DecisionReport,
    ImageQualityMetrics,
    QualityStatus,
    SegmentationQuality,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrics(
    mean_intensity: float = 128.0,
    std_intensity: float = 30.0,
    min_intensity: float = 10.0,
    max_intensity: float = 240.0,
    intensity_range: float = 230.0,
    foreground_fraction: float = 0.40,
) -> ImageQualityMetrics:
    return ImageQualityMetrics(
        mean_intensity=mean_intensity,
        std_intensity=std_intensity,
        min_intensity=min_intensity,
        max_intensity=max_intensity,
        intensity_range=intensity_range,
        foreground_fraction=foreground_fraction,
    )


def _seg(
    foreground_pixels: int = 26214,  # ~40% of 256x256
    foreground_fraction: float = 0.40,
    largest_contour_area: float = 25000.0,
    has_contours: bool = True,
) -> SegmentationQuality:
    return SegmentationQuality(
        foreground_pixels=foreground_pixels,
        foreground_fraction=foreground_fraction,
        largest_contour_area=largest_contour_area,
        has_contours=has_contours,
    )


def _report(
    study_uid: str = "1.2.3",
    series_uid: str = "4.5.6",
    plane: str = "Axial",
    num_slices: int = 20,
    quality_status: QualityStatus = QualityStatus.OK,
    quality_flags: list = None,
    quality_metrics: ImageQualityMetrics = None,
    segmentation_quality: SegmentationQuality = None,
) -> DecisionReport:
    return DecisionReport(
        study_uid=study_uid,
        series_uid=series_uid,
        plane=plane,
        num_slices=num_slices,
        image_height=256,
        image_width=256,
        quality_metrics=quality_metrics or _metrics(),
        segmentation_quality=segmentation_quality or _seg(),
        quality_status=quality_status,
        quality_flags=quality_flags or [],
    )


# ---------------------------------------------------------------------------
# ComparisonStatus enum
# ---------------------------------------------------------------------------

class TestComparisonStatus:
    def test_enum_values(self):
        assert ComparisonStatus.STABLE.value == "STABLE"
        assert ComparisonStatus.REVIEW.value == "REVIEW"


# ---------------------------------------------------------------------------
# FeatureDelta immutability
# ---------------------------------------------------------------------------

class TestFeatureDelta:
    def test_frozen(self):
        delta = FeatureDelta(
            feature="mean_intensity",
            prior_value=100.0,
            current_value=120.0,
            absolute_change=20.0,
            relative_change=0.2,
        )
        with pytest.raises(AttributeError):
            delta.prior_value = 0.0  # type: ignore[misc]

    def test_values(self):
        delta = FeatureDelta(
            feature="foreground_fraction",
            prior_value=0.4,
            current_value=0.6,
            absolute_change=0.2,
            relative_change=0.5,
        )
        assert delta.absolute_change == pytest.approx(0.2)
        assert delta.relative_change == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ComparisonReport immutability
# ---------------------------------------------------------------------------

class TestComparisonReport:
    def test_frozen(self):
        report = ComparisonReport(
            prior_study_uid="A",
            current_study_uid="B",
            prior_quality_status=QualityStatus.OK,
            current_quality_status=QualityStatus.OK,
        )
        with pytest.raises(AttributeError):
            report.status = ComparisonStatus.REVIEW  # type: ignore[misc]

    def test_defaults(self):
        report = ComparisonReport(
            prior_study_uid="A",
            current_study_uid="B",
            prior_quality_status=QualityStatus.OK,
            current_quality_status=QualityStatus.OK,
        )
        assert report.status == ComparisonStatus.STABLE
        assert report.reasons == []
        assert report.feature_deltas == []


# ---------------------------------------------------------------------------
# generate_comparison_report — STABLE cases
# ---------------------------------------------------------------------------

class TestStable:
    def test_stable_both_ok_similar_features(self):
        """Two clean, similar studies produce STABLE with no reasons."""
        prior = _report(study_uid="prior-1")
        current = _report(study_uid="current-1")
        result = generate_comparison_report(prior, current)
        assert result.status == ComparisonStatus.STABLE
        assert result.reasons == []

    def test_stable_carries_both_uids(self):
        prior = _report(study_uid="prior-uid")
        current = _report(study_uid="current-uid")
        result = generate_comparison_report(prior, current)
        assert result.prior_study_uid == "prior-uid"
        assert result.current_study_uid == "current-uid"

    def test_stable_carries_quality_statuses(self):
        prior = _report(study_uid="P", quality_status=QualityStatus.OK)
        current = _report(study_uid="C", quality_status=QualityStatus.OK)
        result = generate_comparison_report(prior, current)
        assert result.prior_quality_status == QualityStatus.OK
        assert result.current_quality_status == QualityStatus.OK

    def test_stable_small_foreground_change_does_not_trigger(self):
        """Foreground change below the threshold must stay STABLE."""
        small_shift = FOREGROUND_FRACTION_REVIEW_THRESHOLD * 0.5  # half the threshold
        prior = _report(
            study_uid="P",
            segmentation_quality=_seg(foreground_fraction=0.40),
            quality_metrics=_metrics(foreground_fraction=0.40),
        )
        current = _report(
            study_uid="C",
            segmentation_quality=_seg(foreground_fraction=0.40 + small_shift),
            quality_metrics=_metrics(foreground_fraction=0.40 + small_shift),
        )
        result = generate_comparison_report(prior, current)
        assert result.status == ComparisonStatus.STABLE

    def test_stable_small_intensity_change_does_not_trigger(self):
        """Mean intensity change below the threshold must stay STABLE."""
        small_shift = MEAN_INTENSITY_REVIEW_THRESHOLD * 0.5
        prior = _report(study_uid="P", quality_metrics=_metrics(mean_intensity=128.0))
        current = _report(study_uid="C", quality_metrics=_metrics(mean_intensity=128.0 + small_shift))
        result = generate_comparison_report(prior, current)
        assert result.status == ComparisonStatus.STABLE


# ---------------------------------------------------------------------------
# generate_comparison_report — REVIEW cases
# ---------------------------------------------------------------------------

class TestReview:
    def test_review_on_large_foreground_shift(self):
        """Foreground fraction shift ≥ threshold triggers REVIEW."""
        shift = FOREGROUND_FRACTION_REVIEW_THRESHOLD + 0.05
        prior = _report(
            study_uid="P",
            segmentation_quality=_seg(foreground_fraction=0.30),
            quality_metrics=_metrics(foreground_fraction=0.30),
        )
        current = _report(
            study_uid="C",
            segmentation_quality=_seg(foreground_fraction=0.30 + shift),
            quality_metrics=_metrics(foreground_fraction=0.30 + shift),
        )
        result = generate_comparison_report(prior, current)
        assert result.status == ComparisonStatus.REVIEW
        assert any("foreground fraction" in r for r in result.reasons)

    def test_review_foreground_reason_is_not_clinical(self):
        """Foreground reason must not claim to be a clinical finding."""
        shift = FOREGROUND_FRACTION_REVIEW_THRESHOLD + 0.05
        prior = _report(
            study_uid="P",
            segmentation_quality=_seg(foreground_fraction=0.25),
            quality_metrics=_metrics(foreground_fraction=0.25),
        )
        current = _report(
            study_uid="C",
            segmentation_quality=_seg(foreground_fraction=0.25 + shift),
            quality_metrics=_metrics(foreground_fraction=0.25 + shift),
        )
        result = generate_comparison_report(prior, current)
        for reason in result.reasons:
            if "foreground fraction" in reason:
                assert "image-derived change candidate" in reason.lower()
                assert "not a clinical finding" in reason.lower()
                assert "clinician review required" in reason.lower()

    def test_review_on_large_mean_intensity_shift(self):
        """Mean intensity shift ≥ threshold triggers REVIEW."""
        shift = MEAN_INTENSITY_REVIEW_THRESHOLD + 5.0
        prior = _report(study_uid="P", quality_metrics=_metrics(mean_intensity=100.0))
        current = _report(study_uid="C", quality_metrics=_metrics(mean_intensity=100.0 + shift))
        result = generate_comparison_report(prior, current)
        assert result.status == ComparisonStatus.REVIEW
        assert any("mean intensity" in r for r in result.reasons)

    def test_review_intensity_reason_is_not_clinical(self):
        """Intensity reason must not claim to be a clinical finding."""
        shift = MEAN_INTENSITY_REVIEW_THRESHOLD + 5.0
        prior = _report(study_uid="P", quality_metrics=_metrics(mean_intensity=100.0))
        current = _report(study_uid="C", quality_metrics=_metrics(mean_intensity=100.0 + shift))
        result = generate_comparison_report(prior, current)
        for reason in result.reasons:
            if "mean intensity" in reason:
                assert "image-derived change candidate" in reason.lower()
                assert "not a clinical finding" in reason.lower()
                assert "clinician review required" in reason.lower()

    def test_review_when_current_quality_is_review(self):
        """Current study with REVIEW quality status triggers comparison REVIEW."""
        prior = _report(study_uid="P", quality_status=QualityStatus.OK)
        current = _report(
            study_uid="C",
            quality_status=QualityStatus.REVIEW,
            quality_flags=["Low intensity range (4.0)"],
        )
        result = generate_comparison_report(prior, current)
        assert result.status == ComparisonStatus.REVIEW
        assert any("current study quality status" in r for r in result.reasons)

    def test_review_when_prior_quality_is_review(self):
        """Prior study with REVIEW quality status triggers comparison REVIEW."""
        prior = _report(
            study_uid="P",
            quality_status=QualityStatus.REVIEW,
            quality_flags=["Low slice count (2)"],
        )
        current = _report(study_uid="C", quality_status=QualityStatus.OK)
        result = generate_comparison_report(prior, current)
        assert result.status == ComparisonStatus.REVIEW
        assert any("prior study quality status" in r for r in result.reasons)

    def test_review_quality_reason_is_not_clinical(self):
        """Quality-status reason must not claim to be a clinical finding."""
        prior = _report(study_uid="P", quality_status=QualityStatus.OK)
        current = _report(
            study_uid="C",
            quality_status=QualityStatus.REVIEW,
            quality_flags=["Low foreground fraction (2.0%)"],
        )
        result = generate_comparison_report(prior, current)
        for reason in result.reasons:
            if "current study quality status" in reason:
                assert "image-derived change candidate" in reason.lower()
                assert "clinician review required" in reason.lower()

    def test_multiple_rules_can_trigger_simultaneously(self):
        """Both foreground and intensity rules can fire at once."""
        ff_shift = FOREGROUND_FRACTION_REVIEW_THRESHOLD + 0.05
        mi_shift = MEAN_INTENSITY_REVIEW_THRESHOLD + 5.0
        prior = _report(
            study_uid="P",
            quality_metrics=_metrics(mean_intensity=100.0, foreground_fraction=0.30),
            segmentation_quality=_seg(foreground_fraction=0.30),
        )
        current = _report(
            study_uid="C",
            quality_metrics=_metrics(mean_intensity=100.0 + mi_shift, foreground_fraction=0.30 + ff_shift),
            segmentation_quality=_seg(foreground_fraction=0.30 + ff_shift),
        )
        result = generate_comparison_report(prior, current)
        assert result.status == ComparisonStatus.REVIEW
        assert len(result.reasons) >= 2


# ---------------------------------------------------------------------------
# generate_comparison_report — INVALID quality blocks comparison
# ---------------------------------------------------------------------------

class TestInvalidBlocked:
    def test_invalid_current_raises_value_error(self):
        prior = _report(study_uid="P", quality_status=QualityStatus.OK)
        current = _report(study_uid="C", quality_status=QualityStatus.INVALID)
        with pytest.raises(ValueError, match="INVALID"):
            generate_comparison_report(prior, current)

    def test_invalid_prior_raises_value_error(self):
        prior = _report(study_uid="P", quality_status=QualityStatus.INVALID)
        current = _report(study_uid="C", quality_status=QualityStatus.OK)
        with pytest.raises(ValueError, match="INVALID"):
            generate_comparison_report(prior, current)

    def test_invalid_error_message_names_the_study(self):
        prior = _report(study_uid="prior-bad", quality_status=QualityStatus.INVALID)
        current = _report(study_uid="current-ok", quality_status=QualityStatus.OK)
        with pytest.raises(ValueError) as exc_info:
            generate_comparison_report(prior, current)
        assert "prior-bad" in str(exc_info.value)

    def test_both_invalid_raises_for_prior_first(self):
        """When both are INVALID, the prior check fires first."""
        prior = _report(study_uid="P", quality_status=QualityStatus.INVALID)
        current = _report(study_uid="C", quality_status=QualityStatus.INVALID)
        with pytest.raises(ValueError):
            generate_comparison_report(prior, current)


# ---------------------------------------------------------------------------
# Auditable feature deltas
# ---------------------------------------------------------------------------

class TestFeatureDeltas:
    def test_deltas_are_present_and_named(self):
        """Result must contain deltas for the standard set of features."""
        prior = _report(study_uid="P")
        current = _report(study_uid="C")
        result = generate_comparison_report(prior, current)
        names = {d.feature for d in result.feature_deltas}
        assert "mean_intensity" in names
        assert "foreground_fraction" in names
        assert "intensity_range" in names
        assert "largest_contour_area" in names

    def test_delta_arithmetic_is_correct(self):
        """absolute_change = current − prior; relative_change = absolute / |prior|."""
        prior = _report(study_uid="P", quality_metrics=_metrics(mean_intensity=100.0))
        current = _report(study_uid="C", quality_metrics=_metrics(mean_intensity=130.0))
        result = generate_comparison_report(prior, current)
        mi_delta = next(d for d in result.feature_deltas if d.feature == "mean_intensity")
        assert mi_delta.prior_value == pytest.approx(100.0)
        assert mi_delta.current_value == pytest.approx(130.0)
        assert mi_delta.absolute_change == pytest.approx(30.0)
        assert mi_delta.relative_change == pytest.approx(0.30)

    def test_delta_relative_change_zero_when_prior_is_zero(self):
        """Relative change is 0.0 when prior value is 0 to avoid division by zero."""
        prior = _report(
            study_uid="P",
            segmentation_quality=_seg(largest_contour_area=0.0),
        )
        current = _report(
            study_uid="C",
            segmentation_quality=_seg(largest_contour_area=5000.0),
        )
        result = generate_comparison_report(prior, current)
        ca_delta = next(d for d in result.feature_deltas if d.feature == "largest_contour_area")
        assert ca_delta.relative_change == pytest.approx(0.0)

    def test_delta_sign_is_current_minus_prior(self):
        """absolute_change is negative when current < prior."""
        prior = _report(study_uid="P", quality_metrics=_metrics(mean_intensity=150.0))
        current = _report(study_uid="C", quality_metrics=_metrics(mean_intensity=100.0))
        result = generate_comparison_report(prior, current)
        mi_delta = next(d for d in result.feature_deltas if d.feature == "mean_intensity")
        assert mi_delta.absolute_change == pytest.approx(-50.0)
