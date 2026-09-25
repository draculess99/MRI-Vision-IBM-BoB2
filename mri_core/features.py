import cv2
import numpy as np

def extract_features(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Calculate simple image/region measurements based on preprocessed image and mask.
    """
    h, w = image.shape[:2]
    
    mean_intensity = float(np.mean(image))
    std_intensity = float(np.std(image))
    min_intensity = float(np.min(image))
    max_intensity = float(np.max(image))
    
    fg_pixels = int(cv2.countNonZero(mask))
    total_pixels = w * h
    fg_percentage = (fg_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    
    features = {
        "image_width": w,
        "image_height": h,
        "mean_intensity": mean_intensity,
        "std_deviation": std_intensity,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "foreground_pixels": fg_pixels,
        "foreground_percentage": fg_percentage
    }
    
    # Check for contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        x, y, bw, bh = cv2.boundingRect(largest_contour)
        features["largest_contour_area"] = float(area)
        features["bounding_rect"] = (x, y, bw, bh)
    else:
        features["largest_contour_area"] = 0.0
        features["bounding_rect"] = (0, 0, 0, 0)
        
    return features
