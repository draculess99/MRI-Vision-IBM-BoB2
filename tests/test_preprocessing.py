import numpy as np
from mri_core.preprocessing import preprocess_image

def test_preprocess_returns_valid_uint8():
    # Create synthetic noisy grayscale image
    synthetic_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    processed = preprocess_image(synthetic_image)
    
    assert processed is not None
    assert processed.dtype == np.uint8
    assert len(processed.shape) == 2

def test_preprocess_not_empty():
    synthetic_image = np.ones((50, 50, 3), dtype=np.uint8) * 128
    processed = preprocess_image(synthetic_image)
    
    assert processed.size > 0
    assert processed.shape[0] > 0
    assert processed.shape[1] > 0
