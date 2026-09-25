import cv2
import numpy as np

def segment_otsu(image: np.ndarray) -> np.ndarray:
    """Otsu thresholding."""
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def segment_adaptive(image: np.ndarray) -> np.ndarray:
    """Adaptive thresholding."""
    mask = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return mask

def apply_morphology(mask: np.ndarray) -> np.ndarray:
    """Simple morphological cleanup."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed

def segment_image(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Perform segmentation on preprocessed grayscale image.
    Returns binary mask.
    """
    if method.lower() == "adaptive":
        mask = segment_adaptive(image)
    else:
        mask = segment_otsu(image)
        
    cleaned_mask = apply_morphology(mask)
    return cleaned_mask
