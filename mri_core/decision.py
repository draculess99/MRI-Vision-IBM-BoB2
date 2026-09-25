"""Deterministic decision/reporting layer for image quality and pipeline integrity.

This module generates structured reports about image quality and processing status.
Reports are based solely on image properties and pipeline integrity checks.
No medical diagnosis or clinical claims are made.

The report is designed so that model predictions (sigmoid probabilities 0-1 per target)
can later be supplied without architectural changes.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum
import numpy as np


class QualityStatus(str, Enum):
    """Image/pipeline quality classification."""
    OK = "OK"
    REVIEW = "REVIEW"
    INVALID = "INVALID"


# Quality thresholds (named constants, easily tunable)
MIN_FOREGROUND_FRACTION = 0.05  # At least 5% foreground
MAX_FOREGROUND_FRACTION = 0.95  # At most 95% foreground (likely segmentation artifact)
MIN_SLICES = 3  # Minimum sensible slice count for volumetric analysis
MIN_IMAGE_DIMENSION = 64  # Minimum pixel dimension
MAX_IMAGE_DIMENSION = 4096  # Maximum sensible pixel dimension
MIN_VALID_INTENSITY_RANGE = 10  # (max - min) should be at least this


@dataclass(frozen=True)
class ImageQualityMetrics:
    """Image-level quality indicators (pipeline integrity only)."""
    mean_intensity: float
    std_intensity: float
    min_intensity: float
    max_intensity: float
    intensity_range: float
    foreground_fraction: float


@dataclass(frozen=True)
class SegmentationQuality:
    """Segmentation/mask quality indicators."""
    foreground_pixels: int
    foreground_fraction: float
    largest_contour_area: float
    has_contours: bool


@dataclass(frozen=True)
class DecisionReport:
    """Immutable structured report on image quality and processing status.

    This report contains metadata, image quality checks, and pipeline status.
    It does NOT contain medical interpretations or clinical conclusions.
    """

    # Study/series identification
    study_uid: str
    series_uid: str
    plane: str

    # Volume information
    num_slices: int
    image_height: int
    image_width: int

    # Quality metrics
    quality_metrics: ImageQualityMetrics
    segmentation_quality: SegmentationQuality

    # Extracted features (raw data)
    extracted_features: Dict[str, Any] = field(default_factory=dict)

    # Quality assessment
    quality_status: QualityStatus = QualityStatus.OK
    quality_flags: List[str] = field(default_factory=list)
    quality_details: Dict[str, str] = field(default_factory=dict)

    # Model prediction status
    model_status: str = "Not available — model checkpoint not loaded"
    model_predictions: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        result["quality_status"] = str(self.quality_status.value)
        result["quality_metrics"] = asdict(self.quality_metrics)
        result["segmentation_quality"] = asdict(self.segmentation_quality)
        return result

    def summary(self) -> str:
        """Human-readable summary of report."""
        lines = [
            f"Study: {self.study_uid}",
            f"Plane: {self.plane}",
            f"Volume: {self.num_slices} slices, {self.image_height}×{self.image_width}px",
            f"Quality: {self.quality_status.value}",
        ]
        if self.quality_flags:
            lines.append(f"Flags: {'; '.join(self.quality_flags)}")
        lines.append(f"Model: {self.model_status}")
        return "\n".join(lines)


def generate_decision_report(
    study_uid: str,
    series_uid: str,
    plane: str,
    volume_data: np.ndarray,
    preprocessed_data: np.ndarray,
    mask: np.ndarray,
    features: Dict[str, Any],
    num_slices: int,
    model_predictions: Optional[Dict[str, float]] = None,
) -> DecisionReport:
    """Generate a decision report from processing pipeline outputs.

    Args:
        study_uid: Study UID
        series_uid: Series UID
        plane: Anatomical plane name
        volume_data: Original volume data (shape: slices, height, width or height, width)
        preprocessed_data: Preprocessed image (from pipeline)
        mask: Segmentation mask
        features: Features extracted by extract_features()
        num_slices: Number of slices in volume
        model_predictions: Optional {target_name: probability} dict, each value finite
            and in [0.0, 1.0]. When omitted, the report's model_status/model_predictions
            keep their "no checkpoint loaded" defaults. When supplied, it is stored
            as-is (target names unchanged) and model_status is updated to a neutral
            "predictions available" message. This never affects the OK/REVIEW/INVALID
            image-quality status below, which is computed independently of model output.

    Returns:
        DecisionReport with quality assessment and pipeline status

    Raises:
        ValueError: If model_predictions contains a non-finite value or one outside [0.0, 1.0].
    """
    if model_predictions is not None:
        for target, probability in model_predictions.items():
            if not np.isfinite(probability):
                raise ValueError(f"model_predictions[{target!r}] is not finite: {probability!r}")
            if not (0.0 <= probability <= 1.0):
                raise ValueError(f"model_predictions[{target!r}] must be in [0.0, 1.0], got {probability!r}")
    # Extract image dimensions
    if len(preprocessed_data.shape) == 3:
        _, height, width = preprocessed_data.shape
    else:
        height, width = preprocessed_data.shape

    # Calculate quality metrics from volume data
    # Use preprocessed for intensity stats (it's the canonical version)
    flat = preprocessed_data.flatten()
    valid = flat[np.isfinite(flat)]

    if len(valid) == 0:
        mean_intensity = 0.0
        std_intensity = 0.0
        min_intensity = 0.0
        max_intensity = 0.0
        intensity_range = 0.0
    else:
        mean_intensity = float(np.mean(valid))
        std_intensity = float(np.std(valid))
        min_intensity = float(np.min(valid))
        max_intensity = float(np.max(valid))
        intensity_range = max_intensity - min_intensity

    quality_metrics = ImageQualityMetrics(
        mean_intensity=mean_intensity,
        std_intensity=std_intensity,
        min_intensity=min_intensity,
        max_intensity=max_intensity,
        intensity_range=intensity_range,
        foreground_fraction=features.get("foreground_percentage", 0.0) / 100.0,
    )

    # Segmentation quality
    foreground_pixels = features.get("foreground_pixels", 0)
    total_pixels = height * width
    foreground_fraction = (
        (foreground_pixels / total_pixels) if total_pixels > 0 else 0.0
    )
    has_contours = features.get("largest_contour_area", 0.0) > 0

    segmentation_quality = SegmentationQuality(
        foreground_pixels=foreground_pixels,
        foreground_fraction=foreground_fraction,
        largest_contour_area=features.get("largest_contour_area", 0.0),
        has_contours=has_contours,
    )

    # Quality assessment
    status = QualityStatus.OK
    flags: List[str] = []
    details: Dict[str, str] = {}

    # Check for invalid/non-finite data
    if not np.isfinite(volume_data).any():
        status = QualityStatus.INVALID
        flags.append("No valid (finite) data in volume")
        details["data_validity"] = "All non-finite"

    # Check intensity range
    if intensity_range < MIN_VALID_INTENSITY_RANGE:
        if status == QualityStatus.OK:
            status = QualityStatus.REVIEW
        flags.append(f"Low intensity range ({intensity_range:.1f})")
        details["intensity_range"] = "Low dynamic range may indicate poor image quality"

    # Check foreground segmentation
    if foreground_fraction < MIN_FOREGROUND_FRACTION:
        if status == QualityStatus.OK:
            status = QualityStatus.REVIEW
        flags.append(f"Low foreground fraction ({foreground_fraction*100:.1f}%)")
        details["foreground"] = "Segmentation may have failed to detect anatomy"
    elif foreground_fraction > MAX_FOREGROUND_FRACTION:
        if status == QualityStatus.OK:
            status = QualityStatus.REVIEW
        flags.append(f"High foreground fraction ({foreground_fraction*100:.1f}%)")
        details["foreground"] = "Possible segmentation artifact or background not removed"

    # Check slice count
    if num_slices < MIN_SLICES:
        if status == QualityStatus.OK:
            status = QualityStatus.REVIEW
        flags.append(f"Low slice count ({num_slices})")
        details["slices"] = "Insufficient volumetric data for robust analysis"

    # Check image dimensions
    if height < MIN_IMAGE_DIMENSION or width < MIN_IMAGE_DIMENSION:
        if status == QualityStatus.OK:
            status = QualityStatus.REVIEW
        flags.append(f"Small image size ({height}×{width})")
        details["image_size"] = "Below recommended minimum"
    elif height > MAX_IMAGE_DIMENSION or width > MAX_IMAGE_DIMENSION:
        if status == QualityStatus.OK:
            status = QualityStatus.REVIEW
        flags.append(f"Large image size ({height}×{width})")
        details["image_size"] = "Unusually large; check for format issues"

    # No contours detected
    if not has_contours and foreground_pixels > 0:
        if status == QualityStatus.OK:
            status = QualityStatus.REVIEW
        flags.append("No contours detected in mask")
        details["contours"] = "Segmentation may be fragmented or noisy"

    report_kwargs = dict(
        study_uid=study_uid,
        series_uid=series_uid,
        plane=plane,
        num_slices=num_slices,
        image_height=height,
        image_width=width,
        quality_metrics=quality_metrics,
        segmentation_quality=segmentation_quality,
        extracted_features=features,
        quality_status=status,
        quality_flags=flags,
        quality_details=details,
    )
    if model_predictions is not None:
        report_kwargs["model_predictions"] = dict(model_predictions)
        report_kwargs["model_status"] = "Model checkpoint loaded — predictions available"

    return DecisionReport(**report_kwargs)
