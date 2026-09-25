import numpy as np
import pytest

from mri_core.preprocessing import (
    validate_finite,
    robust_percentile_normalize,
    to_uint8_image,
    preprocess_mri_slice,
)
from mri_core.pipeline import process_mri_image


def test_nan_inf_handling():
    arr = np.array([
        [10.0, np.nan, 20.0],
        [np.inf, 30.0, -np.inf],
        [40.0, 50.0, 60.0]
    ], dtype=np.float32)

    clean = validate_finite(arr, default_val=0.0)

    # All values must now be finite
    assert np.all(np.isfinite(clean))
    # NaN was replaced with 0.0
    assert clean[0, 1] == 0.0
    # -inf replaced with minimum finite (10.0)
    assert clean[1, 2] == 10.0
    # +inf replaced with maximum finite (60.0)
    assert clean[1, 0] == 60.0


def test_nan_inf_all_nonfinite():
    arr = np.full((10, 10), np.nan, dtype=np.float32)
    clean = validate_finite(arr, default_val=0.0)
    assert np.all(np.isfinite(clean))
    assert np.all(clean == 0.0)


def test_mri_normalization_and_percentile_clipping():
    # Array with standard signal (100 to 200) and extreme spikes
    arr = np.linspace(100, 200, 1000).reshape((100, 10)).astype(np.float32)
    # Extreme outlier spike
    arr[0, 0] = 999999.0
    arr[0, 1] = -999999.0

    norm = robust_percentile_normalize(arr, clip_percentiles=(1.0, 99.0))

    assert norm.dtype == np.float32
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0

    u8 = to_uint8_image(norm)
    assert u8.dtype == np.uint8
    assert u8.min() >= 0
    assert u8.max() <= 255


def test_flat_intensity_slice():
    flat_arr = np.full((50, 50), 100.0, dtype=np.float32)
    norm = robust_percentile_normalize(flat_arr)
    assert norm.shape == flat_arr.shape
    assert np.all(norm == 0.0)

    processed = preprocess_mri_slice(flat_arr)
    assert processed.dtype == np.uint8


def test_complete_mri_slice_pipeline():
    # Synthetic MRI brain slice: dark background, gray matter, bright lesion
    slice_data = np.zeros((128, 128), dtype=np.float32)
    y, x = np.ogrid[:128, :128]
    # Elliptical "brain"
    brain_mask = ((x - 64) ** 2) / (50 ** 2) + ((y - 64) ** 2) / (55 ** 2) <= 1
    slice_data[brain_mask] = 400.0
    # Bright spot / hyperintensity
    spot_mask = (x - 70) ** 2 + (y - 60) ** 2 <= 10 ** 2
    slice_data[spot_mask] = 900.0
    # Add minor noise
    slice_data += np.random.normal(0, 10, slice_data.shape).astype(np.float32)

    preprocessed = preprocess_mri_slice(slice_data, target_width=256)

    assert preprocessed is not None
    assert preprocessed.dtype == np.uint8
    assert preprocessed.shape[1] == 256

    # Test end-to-end pipeline with is_mri=True
    results = process_mri_image(slice_data, segmentation_method="otsu", is_mri=True)

    assert "original" in results
    assert "preprocessed" in results
    assert "mask" in results
    assert "overlay" in results
    assert "features" in results

    assert results["original"].dtype == np.uint8
    assert results["preprocessed"].dtype == np.uint8
    assert results["mask"].dtype == np.uint8
    assert results["mask"].shape == results["preprocessed"].shape

    features = results["features"]
    assert "image_width" in features
    assert "mean_intensity" in features
    assert "foreground_pixels" in features
    assert features["foreground_pixels"] > 0
