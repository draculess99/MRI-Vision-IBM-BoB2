import cv2
import numpy as np

def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay mask on top of the original image without modifying original.
    """
    if len(image.shape) == 2:
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = image.copy()
        
    if mask.shape[:2] != vis_img.shape[:2]:
        mask = cv2.resize(mask, (vis_img.shape[1], vis_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    color_mask = np.zeros_like(vis_img)
    # Highlight segmentation boundaries/areas
    color_mask[mask == 255] = [0, 255, 0] # Green in BGR
    
    overlay = cv2.addWeighted(vis_img, 1.0, color_mask, alpha, 0)
    return overlay
