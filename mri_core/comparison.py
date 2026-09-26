"""Deterministic ChangeGuard comparison layer.

Compares two DecisionReport objects (prior vs current) using only the
image-derived features and quality statuses already present in each report.

No CNN probabilities are read or modified.  No medical claims are made.
Every REVIEW reason string explicitly states it is an image-derived change
candidate requiring clinician review, not a clinical finding or diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from .decision import DecisionReport, QualityStatus


# ---------------------------------------------------------------------------
# Thresholds — named constants so they are easy to audit and adjust.
# ---------------------------------------------------------------------------

# Absolute change in foreground fraction (0-1 scale) that warrants REVIEW.
FOREGROUND_FRACTION_REVIEW_THRESHOLD = 0.15  # 15 percentage points

# Absolute change in mean intensity (0-255 uint8 scale) that warrants REVIEW.
MEAN_INTENSITY_REVIEW_THRESHOLD = 20.0

# Absolute change in foreground pixel count (fraction of total pixels) — secondary guard.
FOREGROUND_PIXEL_FRACTION_REVIEW_THRESHOLD = 0.15


class ComparisonStatus(str, Enum):
    """Outcome of a ChangeGuard comparison run."""
    STABLE = "STABLE"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class FeatureDelta:
    """Auditable numeric change for one image-derived feature between two studies.

    All values are image-pipeline outputs; none are clinical measurements.
    """
    feature: str
    prior_value: float
    current_value: float
    absolute_change: float   # current − prior
    relative_change: float   # (current − prior) / |prior|, or 0.0 when prior is 0


@dataclass(frozen=True)
class ComparisonReport:
    """Immutable result of a deterministic ChangeGuard comparison.

    This report describes image-derived feature changes between two studies.
    It does not constitute a clinical finding, diagnosis, or medical opinion.
    Clinician review is required before any clinical action is taken.
    """

    prior_study_uid: str
    current_study_uid: str
    prior_quality_status: QualityStatus
    current_quality_status: QualityStatus

    # Ordered list of auditable per-feature deltas.
    feature_deltas: List[FeatureDelta] = field(default_factory=list)

    # Comparison outcome.
    status: ComparisonStatus = ComparisonStatus.STABLE

    # Human-readable reasons for REVIEW status (empty when STABLE).
    # Every string explicitly states this is an image-derived change candidate,
    # not a clinical finding or diagnosis.
    reasons: List[str] = field(default_factory=list)


def _delta(feature: str, prior: float, current: float) -> FeatureDelta:
    """Build a FeatureDelta from two scalar values."""
    absolute = current - prior
    relative = absolute / abs(prior) if prior != 0.0 else 0.0
    return FeatureDelta(
        feature=feature,
        prior_value=prior,
        current_value=current,
        absolute_change=absolute,
        relative_change=relative,
    )


def generate_comparison_report(
    prior: DecisionReport,
    current: DecisionReport,
) -> ComparisonReport:
    """Produce a frozen ComparisonReport from two DecisionReport objects.

    Args:
        prior: DecisionReport from the earlier (reference) study.
        current: DecisionReport from the study being reviewed.

    Returns:
        ComparisonReport — always frozen, always auditable.

    Raises:
        ValueError: If either study has QualityStatus.INVALID.  Comparison of
            corrupt or non-finite data cannot produce a meaningful result.
    """
    if prior.quality_status == QualityStatus.INVALID:
        raise ValueError(
            f"Cannot compare: prior study {prior.study_uid!r} has INVALID quality status. "
            "Load a study with OK or REVIEW quality as the prior."
        )
    if current.quality_status == QualityStatus.INVALID:
        raise ValueError(
            f"Cannot compare: current study {current.study_uid!r} has INVALID quality status. "
            "The current study must have OK or REVIEW quality to run a comparison."
        )

    # ------------------------------------------------------------------
    # Build auditable per-feature deltas from image-pipeline measurements.
    # ------------------------------------------------------------------
    prior_qm = prior.quality_metrics
    current_qm = current.quality_metrics
    prior_sq = prior.segmentation_quality
    current_sq = current.segmentation_quality

    deltas: List[FeatureDelta] = [
        _delta("mean_intensity",        prior_qm.mean_intensity,        current_qm.mean_intensity),
        _delta("std_intensity",         prior_qm.std_intensity,         current_qm.std_intensity),
        _delta("intensity_range",       prior_qm.intensity_range,       current_qm.intensity_range),
        _delta("foreground_fraction",   prior_sq.foreground_fraction,   current_sq.foreground_fraction),
        _delta("foreground_pixels",     float(prior_sq.foreground_pixels), float(current_sq.foreground_pixels)),
        _delta("largest_contour_area",  prior_sq.largest_contour_area,  current_sq.largest_contour_area),
    ]

    # ------------------------------------------------------------------
    # Evaluate REVIEW rules — each rule appends a reason string that
    # explicitly labels it as an image-derived change candidate.
    # ------------------------------------------------------------------
    reasons: List[str] = []

    ff_delta = abs(current_sq.foreground_fraction - prior_sq.foreground_fraction)
    if ff_delta >= FOREGROUND_FRACTION_REVIEW_THRESHOLD:
        reasons.append(
            f"Image-derived change candidate: foreground fraction shifted by "
            f"{ff_delta * 100:.1f} percentage points "
            f"(prior {prior_sq.foreground_fraction * 100:.1f}% → "
            f"current {current_sq.foreground_fraction * 100:.1f}%). "
            "This is a segmentation measurement change, not a clinical finding or diagnosis. "
            "Clinician review required."
        )

    mi_delta = abs(current_qm.mean_intensity - prior_qm.mean_intensity)
    if mi_delta >= MEAN_INTENSITY_REVIEW_THRESHOLD:
        reasons.append(
            f"Image-derived change candidate: mean intensity shifted by "
            f"{mi_delta:.1f} units "
            f"(prior {prior_qm.mean_intensity:.1f} → current {current_qm.mean_intensity:.1f}). "
            "This is a pixel-level measurement change, not a clinical finding or diagnosis. "
            "Clinician review required."
        )

    if prior.quality_status != QualityStatus.OK:
        reasons.append(
            f"Image-derived change candidate: prior study quality status is "
            f"{prior.quality_status.value!r} (not OK) — image-quality flags: "
            f"{'; '.join(prior.quality_flags) or 'none recorded'}. "
            "Comparison reliability may be reduced. Clinician review required."
        )

    if current.quality_status != QualityStatus.OK:
        reasons.append(
            f"Image-derived change candidate: current study quality status is "
            f"{current.quality_status.value!r} (not OK) — image-quality flags: "
            f"{'; '.join(current.quality_flags) or 'none recorded'}. "
            "Comparison reliability may be reduced. Clinician review required."
        )

    status = ComparisonStatus.REVIEW if reasons else ComparisonStatus.STABLE

    return ComparisonReport(
        prior_study_uid=prior.study_uid,
        current_study_uid=current.study_uid,
        prior_quality_status=prior.quality_status,
        current_quality_status=current.quality_status,
        feature_deltas=deltas,
        status=status,
        reasons=reasons,
    )
