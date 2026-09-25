from typing import Dict, Any
import numpy as np

from .preprocessing import (
    preprocess_image,
    preprocess_mri_slice,
    resize_preserve_aspect,
    robust_percentile_normalize,
    to_uint8_image,
)
from .segmentation import segment_image
from .features import extract_features
from .visualization import create_overlay

def process_mri_image(
    image: np.ndarray,
    segmentation_method: str = "otsu",
    is_mri: bool = False,
    is_inverted: bool = False,
) -> Dict[str, Any]:
    """
    High level API to process MRI image or standard image.
    Returns dictionary with: original, preprocessed, mask, overlay, features.
    Preserves exact V0.1 behavior when is_mri=False.
    """
    if is_mri:
        preprocessed = preprocess_mri_slice(image, is_inverted=is_inverted)
        if image.dtype != np.uint8:
            vis_original = to_uint8_image(robust_percentile_normalize(image))
            if is_inverted:
                vis_original = 255 - vis_original
        else:
            vis_original = image.copy()
    else:
        preprocessed = preprocess_image(image)
        vis_original = image
        
    mask = segment_image(preprocessed, method=segmentation_method)
    features = extract_features(preprocessed, mask)
    
    h, w = preprocessed.shape[:2]
    # Keep aspect ratio same for original visualization to match
    vis_resized = resize_preserve_aspect(vis_original, target_width=w)
    overlay = create_overlay(vis_resized, mask)
    
    return {
        "original": vis_resized,
        "preprocessed": preprocessed,
        "mask": mask,
        "overlay": overlay,
        "features": features
    }

