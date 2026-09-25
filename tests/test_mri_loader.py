import io
import gzip
import pytest
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
import nibabel as nib

from mri_core.loader import load_dicom, load_nifti, load_mri, detect_file_format
from mri_core.mri_volume import MRIVolume


def make_synthetic_dicom_bytes(
    shape=(32, 32),
    photometric="MONOCHROME2",
    rescale_slope=1.0,
    rescale_intercept=0.0,
    patient_name="DOE^JOHN",
    patient_id="SECRET_PATIENT_123"
) -> bytes:
    """Generates synthetic DICOM bytes with optional PII to test privacy."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset("synthetic.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)


    ds.Modality = "MR"
    ds.SeriesDescription = "Synthetic T1 MRI"
    ds.Rows = shape[0]
    ds.Columns = shape[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.RescaleSlope = rescale_slope
    ds.RescaleIntercept = rescale_intercept
    ds.PixelSpacing = [1.0, 1.0]

    # Add identifying patient fields to verify they are NOT exposed
    ds.PatientName = patient_name
    ds.PatientID = patient_id

    # Synthetic circle pattern
    arr = np.zeros(shape, dtype=np.uint16)
    y, x = np.ogrid[:shape[0], :shape[1]]
    mask = (x - shape[1] // 2) ** 2 + (y - shape[0] // 2) ** 2 <= (shape[0] // 4) ** 2
    arr[mask] = 800

    ds.PixelData = arr.tobytes()

    buf = io.BytesIO()
    ds.save_as(buf)
    return buf.getvalue()


def make_synthetic_nifti_bytes(shape=(32, 32, 11), gzipped=False) -> bytes:
    """Generates synthetic NIfTI-1 bytes (optionally gzipped)."""
    data = np.zeros(shape, dtype=np.float32)
    # Put unique intensity per slice to verify slice indexing
    for z in range(shape[2]):
        data[:, :, z] = float(z + 1) * 10.0

    img = nib.Nifti1Image(data, affine=np.eye(4))
    raw_bytes = img.to_bytes()

    if gzipped:
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            gz.write(raw_bytes)
        return buf.getvalue()
    return raw_bytes


def test_synthetic_dicom_loading_monochrome2():
    dcm_bytes = make_synthetic_dicom_bytes(photometric="MONOCHROME2")
    volume = load_dicom(dcm_bytes)

    assert isinstance(volume, MRIVolume)
    assert volume.format_type == "DICOM"
    assert volume.num_slices == 1
    assert volume.is_inverted is False

    display_slice = volume.get_display_slice(0)
    assert display_slice.dtype == np.uint8
    assert display_slice.shape == (32, 32)
    assert display_slice.max() > display_slice.min()


def test_synthetic_dicom_loading_monochrome1():
    # MONOCHROME1: low values are white, high values are black
    dcm_bytes = make_synthetic_dicom_bytes(photometric="MONOCHROME1")
    volume = load_dicom(dcm_bytes)

    assert volume.is_inverted is True
    display_slice = volume.get_display_slice(0)
    assert display_slice.dtype == np.uint8


def test_synthetic_dicom_rescale_slope_intercept():
    dcm_bytes = make_synthetic_dicom_bytes(rescale_slope=2.0, rescale_intercept=50.0)
    volume = load_dicom(dcm_bytes)

    raw_slice = volume.get_slice(0)
    # Background in arr was 0 -> 0 * 2.0 + 50.0 = 50.0
    assert np.isclose(raw_slice.min(), 50.0)
    # Foreground was 800 -> 800 * 2.0 + 50.0 = 1650.0
    assert np.isclose(raw_slice.max(), 1650.0)


def test_dicom_privacy_no_patient_identifying_metadata():
    dcm_bytes = make_synthetic_dicom_bytes(
        patient_name="DOE^JOHN",
        patient_id="SECRET_PATIENT_123"
    )
    volume = load_dicom(dcm_bytes)

    meta_str = str(volume.metadata).lower()
    assert "john" not in meta_str
    assert "doe" not in meta_str
    assert "secret_patient_123" not in meta_str
    assert "patientname" not in meta_str
    assert "patientid" not in meta_str

    # Safe technical fields should be present
    assert "Modality" in volume.metadata
    assert volume.metadata["Modality"] == "MR"
    assert "Rows" in volume.metadata


def test_synthetic_nifti_loading_uncompressed_and_gz():
    # Test uncompressed .nii
    nii_bytes = make_synthetic_nifti_bytes(shape=(32, 32, 7), gzipped=False)
    vol = load_nifti(nii_bytes)
    assert vol.format_type == "NIFTI"
    assert vol.num_slices == 7

    # Test gzipped .nii.gz
    gz_bytes = make_synthetic_nifti_bytes(shape=(32, 32, 9), gzipped=True)
    vol_gz = load_nifti(gz_bytes)
    assert vol_gz.format_type == "NIFTI"
    assert vol_gz.num_slices == 9


def test_middle_slice_selection():
    # 11 slices -> middle slice index is 11 // 2 = 5
    nii_bytes = make_synthetic_nifti_bytes(shape=(32, 32, 11))
    vol = load_nifti(nii_bytes)

    assert vol.num_slices == 11
    assert vol.default_slice_index == 5

    mid_slice = vol.get_slice(vol.default_slice_index)
    # Per our generator, slice 5 has intensity (5 + 1) * 10 = 60
    assert np.isclose(mid_slice[0, 0], 60.0)


def test_explicit_slice_selection():
    nii_bytes = make_synthetic_nifti_bytes(shape=(32, 32, 8))
    vol = load_nifti(nii_bytes)

    # Slice 0
    s0 = vol.get_slice(0)
    assert np.isclose(s0[0, 0], 10.0)

    # Slice 7
    s7 = vol.get_slice(7)
    assert np.isclose(s7[0, 0], 80.0)

    # Out of range slice raises IndexError
    with pytest.raises(IndexError):
        vol.get_slice(8)

    with pytest.raises(IndexError):
        vol.get_slice(-1)


def test_unified_loader_and_format_detection():
    dcm_bytes = make_synthetic_dicom_bytes()
    assert detect_file_format(dcm_bytes, "scan.dcm") == "dicom"

    nii_bytes = make_synthetic_nifti_bytes()
    assert detect_file_format(nii_bytes, "brain.nii") == "nifti"

    gz_bytes = make_synthetic_nifti_bytes(gzipped=True)
    assert detect_file_format(gz_bytes, "brain.nii.gz") == "nifti"

    vol_dcm = load_mri(dcm_bytes, "scan.dcm")
    assert vol_dcm.format_type == "DICOM"

    vol_nii = load_mri(gz_bytes, "brain.nii.gz")
    assert vol_nii.format_type == "NIFTI"


def test_png_standard_image_backward_compatibility():
    import cv2
    from mri_core.loader import load_image

    # Create synthetic PNG bytes
    synth_img = np.zeros((64, 64, 3), dtype=np.uint8)
    synth_img[20:40, 20:40] = [0, 255, 0]
    _, png_encoded = cv2.imencode(".png", synth_img)
    png_bytes = png_encoded.tobytes()

    # Test legacy load_image
    loaded_img = load_image(png_bytes)
    assert loaded_img.shape == (64, 64, 3)

    # Test load_mri on standard image
    vol = load_mri(png_bytes, "sample.png")
    assert vol.format_type == "IMAGE"
    assert vol.num_slices == 1
    assert vol.get_display_slice(0).shape[:2] == (64, 64)


def test_synthetic_numpy_volume_loading():
    from mri_core.loader import load_numpy

    # Create synthetic MRNet-style volume: (slices, height, width) = (16, 64, 64)
    data = np.zeros((16, 64, 64), dtype=np.uint8)
    for z in range(16):
        data[z, 20:44, 20:44] = (z + 1) * 15

    buf = io.BytesIO()
    np.save(buf, data)
    npy_bytes = buf.getvalue()

    assert detect_file_format(npy_bytes, "exam.npy") == "numpy"

    vol = load_numpy(npy_bytes, "exam.npy")
    assert vol.format_type == "NUMPY"
    assert vol.num_slices == 16
    assert vol.default_slice_index == 8
    assert vol.slice_axis == 0

    s8 = vol.get_slice(8)
    assert s8.shape == (64, 64)
    assert np.isclose(s8.max(), (8 + 1) * 15)

    disp8 = vol.get_display_slice(8)
    assert disp8.shape == (64, 64)
    assert disp8.dtype == np.uint8

    # Also test via unified load_mri
    vol_unified = load_mri(npy_bytes, "exam.npy")
    assert vol_unified.format_type == "NUMPY"
    assert vol_unified.num_slices == 16


@pytest.mark.mrnet
def test_discover_mrnet_dataset_and_real_exam():
    from mri_core.dataset_discovery import discover_mrnet_root, list_mrnet_exams, load_mrnet_exam
    from pathlib import Path

    root = discover_mrnet_root()
    if not root:
        pytest.skip("MRNet dataset not found at default path D:\\MRI_DATASETS\\MRNet")

    # Verify discovered path exists and contains planes without hardcoded apostrophe
    assert root.exists()
    assert (root / "axial").exists()

    axial_exams = list_mrnet_exams(plane="axial")
    assert len(axial_exams) > 0

    # Load real 0000.npy axial exam
    vol = load_mrnet_exam("0000", plane="axial")
    assert vol.format_type == "NUMPY"
    assert vol.num_slices == 44
    assert vol.default_slice_index == 22
    assert vol.shape == (44, 256, 256)
    assert vol.raw_data.dtype == np.uint8
    assert vol.raw_data.min() == 0
    assert vol.raw_data.max() == 255
    assert np.isclose(vol.raw_data.mean(), 63.245, atol=0.01)

    # Middle slice inspection
    mid_slice = vol.get_slice(22)
    assert mid_slice.shape == (256, 256)
    assert mid_slice.dtype == np.uint8

    disp_slice = vol.get_display_slice(22)
    assert disp_slice.shape == (256, 256)
    assert disp_slice.dtype == np.uint8

