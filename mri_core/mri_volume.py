from typing import Dict, Any, Optional
import numpy as np
import cv2

from .preprocessing import robust_percentile_normalize, to_uint8_image, validate_finite

class MRIVolume:
    """
    Clean abstraction representing an MRI volume or 2D medical scan.
    Provides slice indexing, metadata access, and normalized display
    representations without unnecessarily processing the entire volume.
    """
    def __init__(
        self,
        data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        format_type: str = "MRI",
        is_inverted: bool = False,
        slice_axis: int = 2
    ):
        self.raw_data = data
        self.metadata = metadata or {}
        self.format_type = format_type
        self.is_inverted = is_inverted
        self.slice_axis = slice_axis
        self._calculate_slice_info()

    def _calculate_slice_info(self) -> None:
        """Determines the number of slices and slice orientation."""
        if self.format_type == "IMAGE":
            # Standard 2D or color image
            self._num_slices = 1
        elif self.raw_data.ndim == 2:
            self._num_slices = 1
        elif self.raw_data.ndim == 3:
            if self.slice_axis < self.raw_data.ndim:
                self._num_slices = self.raw_data.shape[self.slice_axis]
            else:
                self._num_slices = self.raw_data.shape[-1]
        else:
            self._num_slices = 1

    @property
    def num_slices(self) -> int:
        """Returns the total number of slices in the volume."""
        return self._num_slices

    @property
    def default_slice_index(self) -> int:
        """Defaults to the middle slice (zero-indexed)."""
        return self._num_slices // 2

    @property
    def shape(self) -> tuple:
        """Returns the raw data shape."""
        return self.raw_data.shape

    def get_slice(self, slice_idx: int) -> np.ndarray:
        """
        Retrieves the raw numerical slice data preserving numerical precision.
        """
        if slice_idx < 0 or slice_idx >= self._num_slices:
            raise IndexError(
                f"Slice index {slice_idx} out of range (0 to {self._num_slices - 1})"
            )

        if self.format_type == "IMAGE" or self.raw_data.ndim == 2:
            return self.raw_data

        if self.raw_data.ndim == 3:
            if self.slice_axis == 2:
                slice_arr = self.raw_data[:, :, slice_idx]
            elif self.slice_axis == 0:
                slice_arr = self.raw_data[slice_idx, :, :]
            else:
                slice_arr = np.take(self.raw_data, slice_idx, axis=self.slice_axis)
            return slice_arr

        return self.raw_data

    def get_display_slice(self, slice_idx: int) -> np.ndarray:
        """
        Returns a normalized uint8 2D array suitable for OpenCV visualization
        and downstream pipeline processing.
        """
        raw_slice = self.get_slice(slice_idx)

        # If already a 3-channel color image (e.g. standard PNG/JPG)
        if raw_slice.ndim == 3 and raw_slice.shape[2] in (3, 4):
            if raw_slice.dtype == np.uint8:
                return raw_slice
            clean = validate_finite(raw_slice)
            return np.clip(clean, 0, 255).astype(np.uint8)

        # 2D Grayscale / MRI slice
        # Validate finite values, robust percentile normalize, convert to uint8
        norm = robust_percentile_normalize(raw_slice)
        u8 = to_uint8_image(norm)

        if self.is_inverted:
            u8 = 255 - u8

        return u8
