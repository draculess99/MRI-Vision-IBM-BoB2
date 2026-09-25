from pathlib import Path
from typing import Optional, List, Union
import numpy as np

from .mri_volume import MRIVolume

DEFAULT_MRNET_ROOT = r"D:\MRI_DATASETS\MRNet"


def discover_mrnet_root(base_dir: Union[str, Path] = DEFAULT_MRNET_ROOT) -> Optional[Path]:
    """
    Dynamically locates the MRNet dataset container directory without
    depending on hardcoded folder names, spaces, or apostrophes.
    
    Searches for a directory containing the subfolders 'axial', 'coronal', or 'sagittal'.
    """
    base = Path(base_dir)
    if not base.exists():
        return None

    # Check if base itself contains the plane directories
    planes = {"axial", "coronal", "sagittal"}
    try:
        subdirs = {d.name.lower() for d in base.iterdir() if d.is_dir()}
        if planes.intersection(subdirs):
            return base
    except Exception:
        pass

    # Search recursively for any directory containing plane subdirectories
    try:
        for item in base.rglob("*"):
            if item.is_dir() and item.name.lower() in planes:
                return item.parent
    except Exception:
        pass

    return None


def list_mrnet_exams(plane: str = "axial", base_dir: Union[str, Path] = DEFAULT_MRNET_ROOT) -> List[Path]:
    """
    Discovers and lists all available exam paths for a given plane (axial, coronal, sagittal).
    """
    root = discover_mrnet_root(base_dir)
    if not root:
        return []

    plane_dir = root / plane.lower()
    if not plane_dir.exists() or not plane_dir.is_dir():
        matches = [d for d in root.iterdir() if d.is_dir() and d.name.lower() == plane.lower()]
        if not matches:
            return []
        plane_dir = matches[0]

    return sorted(plane_dir.glob("*.npy"))


def load_mrnet_exam(
    exam_id_or_path: Union[str, Path, int],
    plane: str = "axial",
    base_dir: Union[str, Path] = DEFAULT_MRNET_ROOT
) -> MRIVolume:
    """
    Loads an MRNet examination as an MRIVolume.
    Can accept a full Path, a filename (e.g. '0000.npy'), or an exam id (e.g. '0000' or 0).
    """
    if isinstance(exam_id_or_path, Path) and exam_id_or_path.exists():
        file_path = exam_id_or_path
    elif isinstance(exam_id_or_path, str) and Path(exam_id_or_path).exists():
        file_path = Path(exam_id_or_path)
    else:
        root = discover_mrnet_root(base_dir)
        if not root:
            raise FileNotFoundError(f"MRNet dataset root not found under: {base_dir}")

        # Format exam string
        if isinstance(exam_id_or_path, int):
            exam_str = f"{exam_id_or_path:04d}.npy"
        else:
            exam_str = str(exam_id_or_path)
            if not exam_str.endswith(".npy"):
                exam_str += ".npy"

        plane_dir = root / plane.lower()
        file_path = plane_dir / exam_str
        if not file_path.exists():
            raise FileNotFoundError(f"Exam file not found at: {file_path}")

    arr = np.load(file_path)

    # MRNet arrays have shape (num_slices, 256, 256)
    # axis 0 is the MRI slice dimension
    slice_axis = 0 if arr.ndim == 3 else 2
    num_slices = arr.shape[slice_axis] if arr.ndim >= 3 else 1

    metadata = {
        "Modality": "MRI (MRNet)",
        "Plane": plane.capitalize(),
        "Exam File": file_path.name,
        "Dimensions": list(arr.shape),
        "Data Type": str(arr.dtype),
        "Slice Count": num_slices,
        "Intensity Min": float(arr.min()),
        "Intensity Max": float(arr.max()),
        "Intensity Mean": round(float(arr.mean()), 3),
    }

    return MRIVolume(
        data=arr,
        metadata=metadata,
        format_type="NUMPY",
        is_inverted=False,
        slice_axis=slice_axis
    )
