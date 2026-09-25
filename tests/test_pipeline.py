import numpy as np
import cv2
from mri_core.pipeline import process_mri_image
from mri_core.segmentation import segment_image
from mri_core.features import extract_features

def test_segmentation_returns_binary_mask():
    synthetic_image = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(synthetic_image, (50, 50), 20, 255, -1)
    
    mask = segment_image(synthetic_image, method="otsu")
    
    assert mask is not None
    assert len(np.unique(mask)) <= 2
    assert mask.shape == synthetic_image.shape

def test_feature_dictionary():
    synthetic_image = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(synthetic_image, (50, 50), 20, 255, -1)
    mask = segment_image(synthetic_image, method="otsu")
    
    features = extract_features(synthetic_image, mask)
    
    assert "image_width" in features
    assert "mean_intensity" in features
    assert "foreground_pixels" in features
    assert "largest_contour_area" in features

def test_pipeline_runs_synthetic_image():
    synthetic_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(synthetic_image, (50, 50), 20, (255, 255, 255), -1)
    
    results = process_mri_image(synthetic_image, "adaptive")
    
    assert "original" in results
    assert "preprocessed" in results
    assert "mask" in results
    assert "overlay" in results
    assert "features" in results
    
    # Mask dimensions should match preprocessed image dimensions
    assert results["mask"].shape == results["preprocessed"].shape
