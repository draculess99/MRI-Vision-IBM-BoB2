import cv2
import numpy as np
from typing import Optional, Tuple

def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()

def resize_preserve_aspect(image: np.ndarray, target_width: int = 512) -> np.ndarray:
    h, w = image.shape[:2]
    ratio = target_width / w
    target_height = int(h * ratio)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

def normalize_intensity(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def apply_clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (5, 5), 0)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Combined preprocessing pipeline (Generic V0.1).
    Does not modify the original image array.
    """
    gray = to_grayscale(image)
    resized = resize_preserve_aspect(gray)
    normalized = normalize_intensity(resized)
    enhanced = apply_clahe(normalized)
    denoised = denoise(enhanced)
    return denoised

# --- MRI-Aware Preprocessing Pathway (V0.2) ---

def validate_finite(slice_data: np.ndarray, default_val: float = 0.0) -> np.ndarray:
    """
    Ensures all elements in slice_data are finite.
    Replaces NaN and +/- Inf with valid finite values.
    """
    if np.all(np.isfinite(slice_data)):
        return slice_data.copy()
    
    clean = slice_data.copy().astype(np.float32)
    finite_mask = np.isfinite(clean)
    if not np.any(finite_mask):
        return np.full_like(clean, default_val)
    
    f_min = float(np.min(clean[finite_mask]))
    f_max = float(np.max(clean[finite_mask]))
    
    clean[np.isnan(clean)] = default_val
    clean[clean == -np.inf] = f_min
    clean[clean == np.inf] = f_max
    return clean

def robust_percentile_normalize(
    slice_data: np.ndarray,
    clip_percentiles: Tuple[float, float] = (1.0, 99.0)
) -> np.ndarray:
    """
    Robust intensity normalization using percentile clipping.
    Does not assume calibrated physical units.
    Returns float32 array in [0.0, 1.0].
    """
    clean = validate_finite(slice_data)
    p_low = float(np.percentile(clean, clip_percentiles[0]))
    p_high = float(np.percentile(clean, clip_percentiles[1]))
    
    if p_high > p_low:
        clipped = np.clip(clean, p_low, p_high)
        norm = (clipped - p_low) / (p_high - p_low)
    else:
        norm = np.zeros_like(clean, dtype=np.float32)
    return norm.astype(np.float32)

def to_uint8_image(norm_data: np.ndarray) -> np.ndarray:
    """
    Converts [0.0, 1.0] normalized float data to uint8 [0, 255].
    """
    clipped = np.clip(norm_data, 0.0, 1.0)
    return np.round(clipped * 255.0).astype(np.uint8)

def preprocess_mri_slice(
    slice_data: np.ndarray,
    target_width: Optional[int] = 512,
    clip_percentiles: Tuple[float, float] = (1.0, 99.0),
    is_inverted: bool = False
) -> np.ndarray:
    """
    MRI-aware preprocessing pipeline:
    MRI slice
    -> finite-value validation
    -> robust intensity normalization
    -> percentile clipping
    -> uint8 conversion
    -> CLAHE
    -> conservative denoising
    -> optional resize
    """
    # 1. Finite-value validation & 2. Robust intensity normalization & 3. Percentile clipping
    norm = robust_percentile_normalize(slice_data, clip_percentiles=clip_percentiles)
    
    # 4. uint8 conversion
    u8 = to_uint8_image(norm)
    if is_inverted:
        u8 = 255 - u8
        
    # 5. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(u8)
    
    # 6. Conservative denoising (gentle Gaussian blur 3x3 to preserve delicate edges)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 7. Optional resize
    if target_width is not None and target_width > 0:
        denoised = resize_preserve_aspect(denoised, target_width=target_width)
        
    return denoised

