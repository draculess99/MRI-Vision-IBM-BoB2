import io
import gzip
from typing import Optional, Dict, Any
import cv2
import numpy as np
from PIL import Image
import pydicom
import nibabel as nib

from .mri_volume import MRIVolume

def load_image(file_bytes: bytes) -> np.ndarray:
    """
    Loads an image from bytes (PNG, JPG, JPEG) into an OpenCV format.
    Provides useful errors for corrupt/unsupported images.
    (Preserved from V0.1).
    """
    try:
        # Use PIL to read image from bytes
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Handle different color modes
        if img_array.ndim == 3 and img_array.shape[2] == 4:
            # RGBA to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            # RGB to BGR for OpenCV
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
        return img_array
    except Exception as e:
        raise ValueError(f"Failed to load image. Ensure it is a valid PNG/JPG/JPEG. Error: {e}")

def detect_file_format(file_bytes: bytes, filename: Optional[str] = None) -> str:
    """
    Identifies file type as 'dicom', 'nifti', 'numpy', 'image', or 'unknown'.
    Uses file extension hints and file magic bytes.
    """
    name_lower = (filename or "").lower()
    
    if name_lower.endswith(".dcm"):
        return "dicom"
    if name_lower.endswith(".nii") or name_lower.endswith(".nii.gz"):
        return "nifti"
    if name_lower.endswith(".npy") or file_bytes.startswith(b"\x93NUMPY"):
        return "numpy"
    if any(name_lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]):
        return "image"
        
    # DICOM magic tag at byte 128
    if len(file_bytes) >= 132 and file_bytes[128:132] == b"DICM":
        return "dicom"
    
    # GZIP header (often .nii.gz)
    if file_bytes.startswith(b"\x1f\x8b"):
        try:
            decomp = gzip.decompress(file_bytes[:1000])
            if len(decomp) >= 348 and decomp[344:348] in (b"n+1\0", b"ni1\0", b"n+2\0"):
                return "nifti"
        except Exception:
            pass
        return "nifti"
        
    # NIfTI magic at offset 344
    if len(file_bytes) >= 348 and file_bytes[344:348] in (b"n+1\0", b"ni1\0", b"n+2\0"):
        return "nifti"
        
    # NumPy magic
    if file_bytes.startswith(b"\x93NUMPY"):
        return "numpy"

    # PNG magic
    if file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"
        
    # JPEG magic
    if file_bytes.startswith(b"\xff\xd8\xff"):
        return "image"
        
    return "unknown"

def load_dicom(file_bytes: bytes) -> MRIVolume:
    """
    Loads DICOM (.dcm) file bytes into an MRIVolume.
    Applies rescale slope/intercept and handles MONOCHROME1 vs MONOCHROME2.
    Strictly filters out any patient-identifying metadata (PII).
    """
    try:
        dcm = pydicom.dcmread(io.BytesIO(file_bytes))
        pixel_array = dcm.pixel_array.astype(np.float32)
        
        # Apply Rescale Slope and Intercept if present
        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        if slope != 1.0 or intercept != 0.0:
            pixel_array = pixel_array * slope + intercept
            
        # Photometric Interpretation
        photo = str(getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")).strip().upper()
        is_inverted = (photo == "MONOCHROME1")
        
        # Extract strictly non-identifying technical metadata
        safe_keys = [
            ("Modality", "Modality"),
            ("SeriesDescription", "Series Description"),
            ("ImageType", "Image Type"),
            ("Rows", "Rows"),
            ("Columns", "Columns"),
            ("PixelSpacing", "Pixel Spacing"),
            ("SliceThickness", "Slice Thickness"),
            ("SpacingBetweenSlices", "Spacing Between Slices"),
            ("PhotometricInterpretation", "Photometric Interpretation"),
            ("RescaleSlope", "Rescale Slope"),
            ("RescaleIntercept", "Rescale Intercept"),
            ("WindowCenter", "Window Center"),
            ("WindowWidth", "Window Width"),
            ("RepetitionTime", "Repetition Time (TR)"),
            ("EchoTime", "Echo Time (TE)"),
            ("MagneticFieldStrength", "Field Strength (T)"),
        ]
        
        metadata: Dict[str, Any] = {}
        for attr_name, label in safe_keys:
            if hasattr(dcm, attr_name):
                val = getattr(dcm, attr_name)
                if val is not None and str(val).strip() != "":
                    # Convert DICOM MultiValue or arrays to readable strings/lists
                    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                        metadata[label] = [float(x) if isinstance(x, (int, float)) else str(x) for x in val]
                    else:
                        metadata[label] = float(val) if isinstance(val, (int, float)) else str(val)
                        
        metadata["Slice Count"] = 1 if pixel_array.ndim == 2 else pixel_array.shape[0]
        
        # Multi-frame vs 2D
        slice_axis = 0 if pixel_array.ndim == 3 else 2
        
        return MRIVolume(
            data=pixel_array,
            metadata=metadata,
            format_type="DICOM",
            is_inverted=is_inverted,
            slice_axis=slice_axis
        )
    except Exception as e:
        raise ValueError(f"Failed to load DICOM image: {e}")

def load_nifti(file_bytes: bytes, filename: Optional[str] = None) -> MRIVolume:
    """
    Loads NIfTI (.nii, .nii.gz) file bytes into an MRIVolume.
    Determines number of slices and exposes non-identifying metadata.
    """
    try:
        if file_bytes.startswith(b"\x1f\x8b"):
            decompressed = gzip.decompress(file_bytes)
            fh = nib.FileHolder(fileobj=io.BytesIO(decompressed))
        else:
            fh = nib.FileHolder(fileobj=io.BytesIO(file_bytes))
            
        nii = nib.Nifti1Image.from_file_map({"image": fh})
        data = np.asarray(nii.dataobj, dtype=np.float32)
        
        # If 4D volume, take first volume
        if data.ndim == 4:
            data = data[:, :, :, 0]
            
        num_slices = data.shape[2] if data.ndim >= 3 else 1
        zooms = [round(float(z), 3) for z in nii.header.get_zooms()[:data.ndim]]
        
        metadata = {
            "Modality": "MRI",
            "Dimensions": list(data.shape),
            "Voxel Spacing (mm)": zooms,
            "Data Type": str(nii.header.get_data_dtype()),
            "Slice Count": num_slices
        }
        
        return MRIVolume(
            data=data,
            metadata=metadata,
            format_type="NIFTI",
            is_inverted=False,
            slice_axis=2 if data.ndim >= 3 else 0
        )
    except Exception as e:
        raise ValueError(f"Failed to load NIfTI image: {e}")

def load_numpy(file_bytes: bytes, filename: Optional[str] = None) -> MRIVolume:
    """
    Loads NumPy (.npy) array bytes (such as MRNet examinations) into an MRIVolume.
    Automatically handles (S, H, W) slice orientation.
    """
    try:
        arr = np.load(io.BytesIO(file_bytes))
        
        # In MRNet, shape is (num_slices, 256, 256) where axis 0 is the slice axis
        if arr.ndim == 3:
            if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
                slice_axis = 0
            else:
                slice_axis = 2
            num_slices = arr.shape[slice_axis]
        elif arr.ndim == 2:
            slice_axis = 0
            num_slices = 1
        else:
            slice_axis = 0
            num_slices = 1

        metadata = {
            "Modality": "MRI (NumPy / MRNet)",
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
    except Exception as e:
        raise ValueError(f"Failed to load NumPy (.npy) MRI file: {e}")

def load_mri(file_bytes: bytes, filename: Optional[str] = None) -> MRIVolume:
    """
    Unified loader for medical and generic images.
    Auto-detects format (DICOM, NIfTI, NumPy, PNG/JPG/JPEG).
    """
    fmt = detect_file_format(file_bytes, filename)
    
    if fmt == "dicom":
        return load_dicom(file_bytes)
    elif fmt == "nifti":
        return load_nifti(file_bytes, filename)
    elif fmt == "numpy":
        return load_numpy(file_bytes, filename)
    elif fmt == "image":
        img_array = load_image(file_bytes)
        metadata = {
            "Format": "Standard Image",
            "Dimensions": list(img_array.shape),
            "Slice Count": 1
        }
        return MRIVolume(
            data=img_array,
            metadata=metadata,
            format_type="IMAGE",
            is_inverted=False,
            slice_axis=0
        )
    else:
        # Fallback trial
        for loader_fn in [
            load_dicom,
            load_numpy,
            lambda b: load_nifti(b, filename),
            lambda b: MRIVolume(load_image(b), metadata={"Format": "Standard Image"}, format_type="IMAGE")
        ]:
            try:
                return loader_fn(file_bytes)
            except Exception:
                continue
                
        raise ValueError(
            "Unsupported or corrupted file format. Supported formats: PNG, JPG, JPEG, DICOM (.dcm), NIfTI (.nii, .nii.gz), NumPy (.npy)"
        )


