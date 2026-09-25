"""Deterministic CPU features from nine sampled slices of each MRNet plane."""

from pathlib import Path

import numpy as np

from .manifest import PLANES, Manifest

SLICE_FEATURES = ("intensity_mean", "intensity_std", "intensity_p10", "intensity_p50",
                  "intensity_p90", "entropy", "gradient_mean", "gradient_std",
                  "gradient_p90", "edge_fraction")
FEATURE_NAMES = tuple(f"{plane}_{name}_{aggregation}" for plane in PLANES
                      for name in SLICE_FEATURES for aggregation in ("mean", "std"))


def sampled_indices(slice_count: int, sample_count: int = 9) -> np.ndarray:
    if slice_count < 1 or sample_count < 1:
        raise ValueError("Slice and sample counts must be positive")
    return np.unique(np.linspace(0, slice_count - 1, min(slice_count, sample_count), dtype=int))


def slice_features(image: np.ndarray) -> np.ndarray:
    """Per-slice 1st/99th percentile normalization; no dataset-fitted parameters."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 2 or not np.isfinite(image).all():
        raise ValueError("Expected a finite 2D slice with both dimensions >= 2")
    low, high = np.percentile(image, [1, 99])
    normalized = np.clip((image - low) / (high - low), 0, 1) if high > low else np.zeros_like(image)
    p10, p50, p90 = np.percentile(normalized, [10, 50, 90])
    histogram = np.histogram(normalized, bins=32, range=(0, 1))[0]
    probabilities = histogram[histogram > 0] / normalized.size
    entropy = -np.sum(probabilities * np.log2(probabilities))
    dy, dx = np.gradient(normalized)
    gradient = np.hypot(dx, dy)
    return np.array([normalized.mean(), normalized.std(), p10, p50, p90, entropy,
                     gradient.mean(), gradient.std(), np.percentile(gradient, 90),
                     np.mean(gradient > 0.1)], dtype=np.float64)


def volume_features(path: Path) -> np.ndarray:
    """Load read-only, keep MRNet axis 0 as slices, aggregate means and stds."""
    volume = np.load(path, mmap_mode="r", allow_pickle=False)
    if (volume.ndim != 3 or volume.shape[0] < 1 or min(volume.shape[1:]) < 2
            or volume.dtype.kind not in "uif"):
        raise ValueError(f"Expected numeric (slices, height, width) volume: {path}")
    features = np.vstack([slice_features(volume[index]) for index in sampled_indices(volume.shape[0])])
    return np.column_stack((features.mean(axis=0), features.std(axis=0))).ravel()


def extract_features(manifest: Manifest, progress=None) -> np.ndarray:
    if manifest.missing_images:
        raise ValueError(f"Cannot run complete-plane baseline: {len(manifest.missing_images)} missing images")
    if not manifest.rows:
        raise ValueError("Cannot extract features from an empty manifest")
    features = []
    for index, row in enumerate(manifest.rows, start=1):
        features.append(np.concatenate([volume_features(manifest.paths[(row["exam_id"], plane)])
                                        for plane in PLANES]))
        if progress is not None:
            progress(index, len(manifest.rows))
    return np.vstack(features)
