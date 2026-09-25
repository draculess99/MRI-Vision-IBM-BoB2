import warnings

import numpy as np
import pytest

from mri_core import DicomSeriesError, DicomSeriesWarning, MRIVolume, load_dicom_series

from .dicom_factory import AXIAL, CORONAL, SAGITTAL, SERIES_UID, STUDY_UID, at, corrupt_element, write_series


def pixel_values(volume):
    return [float(volume.get_slice(i)[0, 0]) for i in range(volume.num_slices)]


def without_position_tags(ds):
    del ds.ImagePositionPatient
    del ds.ImageOrientationPatient


# ----------------------------------------------------------------------------- ordering

def test_orders_by_instance_number_not_filename(tmp_path):
    paths = write_series(tmp_path, count=5)
    assert [p.name for p in paths] == sorted((p.name for p in paths), reverse=True)  # filename order is backwards
    assert pixel_values(load_dicom_series(tmp_path)) == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_orders_shuffled_instance_numbers_written_in_arbitrary_file_order(tmp_path):
    write_series(tmp_path, instance_numbers=[3, 1, 4, 2], filenames=["a.dcm", "b.dcm", "c.dcm", "d.dcm"])
    volume = load_dicom_series(tmp_path)
    assert pixel_values(volume) == [10.0, 20.0, 30.0, 40.0]
    assert volume.metadata["Instance Number Range"] == [1, 4]


def test_decreasing_physical_positions_are_valid(tmp_path):
    write_series(tmp_path, count=4, spacing=-5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        volume = load_dicom_series(tmp_path)
    assert pixel_values(volume) == [10.0, 20.0, 30.0, 40.0]
    assert volume.metadata["Measured Slice Spacing (mm)"] == 5.0


def test_instance_numbers_need_not_start_at_one(tmp_path):
    write_series(tmp_path, instance_numbers=[101, 102, 103])
    volume = load_dicom_series(tmp_path)
    assert pixel_values(volume) == [1010.0, 1020.0, 1030.0]
    assert volume.metadata["Instance Number Range"] == [101, 103]


# ----------------------------------------------------------------------------- assembly and metadata

def test_returns_mri_volume_with_slice_axis_zero(tmp_path):
    write_series(tmp_path, count=6, rows=8, columns=12)
    volume = load_dicom_series(tmp_path)
    assert isinstance(volume, MRIVolume)
    assert volume.format_type == "DICOM"
    assert volume.shape == (6, 8, 12)
    assert volume.raw_data.dtype == np.float32
    assert volume.slice_axis == 0
    assert volume.num_slices == 6 and volume.default_slice_index == 3
    assert volume.get_slice(2).shape == (8, 12)
    assert volume.get_display_slice(2).dtype == np.uint8
    assert volume.is_inverted is False


def test_preserves_series_metadata(tmp_path):
    write_series(tmp_path, count=4, orientation=SAGITTAL, spacing=5.0)
    volume = load_dicom_series(tmp_path, extra_metadata={"Anatomical Plane": "Sagittal"})
    meta = volume.metadata
    assert meta["Series Instance UID"] == SERIES_UID
    assert meta["Series Description"] == "SYNTH SERIES"
    assert meta["Modality"] == "MR"
    assert meta["Pixel Spacing"] == [0.5, 0.5]
    assert meta["Slice Thickness"] == 3.0
    assert meta["Spacing Between Slices"] == 4.0
    assert meta["Slice Count"] == 4
    assert meta["Rows"] == 8 and meta["Columns"] == 8
    assert meta["Measured Slice Spacing (mm)"] == 5.0
    assert meta["Slice Position Range (mm)"] == [0.0, 15.0]
    assert meta["Slice Order Validation"].startswith("position")
    assert meta["Anatomical Plane"] == "Sagittal"
    assert not any("Patient" in key for key in meta)


@pytest.mark.parametrize("orientation,plane", [(AXIAL, "Axial"), (CORONAL, "Coronal"), (SAGITTAL, "Sagittal")])
def test_derives_acquisition_plane_from_orientation(tmp_path, orientation, plane):
    write_series(tmp_path, count=3, orientation=orientation)
    assert load_dicom_series(tmp_path).metadata["Acquisition Plane (DICOM orientation)"] == plane


def test_applies_rescale_and_flags_monochrome1(tmp_path):
    def edit(ds):
        ds.RescaleSlope, ds.RescaleIntercept = 2.0, -5.0
        ds.PhotometricInterpretation = "MONOCHROME1"
    write_series(tmp_path, count=3, mutate=lambda i, ds: edit(ds))
    volume = load_dicom_series(tmp_path)
    assert pixel_values(volume) == [15.0, 35.0, 55.0]
    assert volume.is_inverted is True


def test_single_slice_series_loads(tmp_path):
    write_series(tmp_path, count=1)
    volume = load_dicom_series(tmp_path)
    assert volume.shape == (1, 8, 8)
    assert "Measured Slice Spacing (mm)" not in volume.metadata


def test_well_formed_series_emits_no_warnings(tmp_path):
    write_series(tmp_path, count=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_dicom_series(tmp_path, expected_series_uid=SERIES_UID, expected_study_uid=STUDY_UID)


def test_hidden_files_and_pattern_filter(tmp_path):
    write_series(tmp_path, count=3)
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    assert load_dicom_series(tmp_path).num_slices == 3
    (tmp_path / "notes.txt").write_text("not a slice")
    with pytest.raises(DicomSeriesError, match="notes.txt"):
        load_dicom_series(tmp_path)
    assert load_dicom_series(tmp_path, pattern="*.dcm").num_slices == 3


# ----------------------------------------------------------------------------- malformed series are rejected

def _bad_shape(ds):
    ds.Rows = ds.Columns = 4
    ds.PixelData = bytes(4 * 4 * 2)


MALFORMED = [
    ("duplicate-instance-number", at(2, lambda ds: setattr(ds, "InstanceNumber", 1)), "Duplicate InstanceNumber"),
    ("missing-instance-number", at(2, lambda ds: delattr(ds, "InstanceNumber")), "InstanceNumber is required"),
    ("inconsistent-orientation", at(2, lambda ds: setattr(ds, "ImageOrientationPatient", list(map(float, CORONAL)))),
     "Inconsistent ImageOrientationPatient"),
    ("non-monotonic-positions", at(1, lambda ds: setattr(ds, "ImagePositionPatient", [0.0, 0.0, 99.0])), "not monotonic"),
    ("duplicate-positions", at(2, lambda ds: setattr(ds, "ImagePositionPatient", [0.0, 0.0, 5.0])), "not monotonic"),
    ("partial-position-tags", at(2, lambda ds: delattr(ds, "ImagePositionPatient")), "Only some slices carry"),
    ("malformed-position", at(1, lambda ds: setattr(ds, "ImagePositionPatient", [0.0, 0.0])), "ImagePositionPatient"),
    ("degenerate-orientation", lambda i, ds: setattr(ds, "ImageOrientationPatient", [1.0, 0, 0, 1.0, 0, 0]),
     "does not define a slice plane"),
    ("mixed-series-uid", at(1, lambda ds: setattr(ds, "SeriesInstanceUID", "1.2.826.77")), "disagree on SeriesInstanceUID"),
    ("missing-series-uid", at(1, lambda ds: delattr(ds, "SeriesInstanceUID")), "no SeriesInstanceUID"),
    ("mixed-shapes", at(3, _bad_shape), "pixel array shape"),
    ("mixed-photometric", at(1, lambda ds: setattr(ds, "PhotometricInterpretation", "MONOCHROME1")),
     "PhotometricInterpretation"),
    ("multi-frame", at(1, lambda ds: (setattr(ds, "NumberOfFrames", 2), setattr(ds, "PixelData", bytes(2 * 8 * 8 * 2)))),
     "one 2D frame"),
]


@pytest.mark.parametrize("mutate,message", [m[1:] for m in MALFORMED], ids=[m[0] for m in MALFORMED])
def test_malformed_series_is_rejected(tmp_path, mutate, message):
    write_series(tmp_path, count=5, mutate=mutate)
    with pytest.raises(DicomSeriesError, match=message):
        load_dicom_series(tmp_path)


def _with_rescale(index, ds):
    ds.RescaleSlope, ds.RescaleIntercept = 2.0, 1.0


# (element, corrupt text, index of the slice to corrupt). Slice metadata is read from the first slice (index 0);
# spacing and rescale values are read from every slice.
MALFORMED_NUMERIC_HEADERS = [
    ("PixelSpacing", "a\\b", 1),
    ("PixelSpacing", "0.5", 2),
    ("RescaleSlope", "abc", 2),
    ("RescaleIntercept", "xyz", 1),
    ("SliceThickness", "abc", 0),
    ("SpacingBetweenSlices", "abc", 0),
    ("InstanceNumber", "x", 1),
]


@pytest.mark.filterwarnings("ignore:Invalid value for VR")  # pydicom itself warns when it reads these values
@pytest.mark.parametrize("keyword,bad,slice_index", MALFORMED_NUMERIC_HEADERS,
                         ids=[f"{k}-{b!r}-slice{i}" for k, b, i in MALFORMED_NUMERIC_HEADERS])
def test_malformed_numeric_header_is_a_dicom_series_error_naming_the_file(tmp_path, keyword, bad, slice_index):
    paths = write_series(tmp_path, count=3, mutate=_with_rescale)
    corrupt_element(paths[slice_index], keyword, bad)
    with pytest.raises(DicomSeriesError) as excinfo:
        load_dicom_series(tmp_path)
    message = str(excinfo.value)
    assert paths[slice_index].name in message
    assert keyword in message


def test_garbage_file_is_reported_by_name_not_skipped(tmp_path):
    write_series(tmp_path, count=4)
    (tmp_path / "corrupt_a.dcm").write_bytes(b"this is not dicom")
    (tmp_path / "corrupt_b.dcm").write_bytes(b"")
    with pytest.raises(DicomSeriesError) as excinfo:
        load_dicom_series(tmp_path)
    message = str(excinfo.value)
    assert "corrupt_a.dcm" in message and "corrupt_b.dcm" in message
    assert "2 of 6 files" in message


def test_truncated_dicom_is_reported_by_name(tmp_path):
    paths = write_series(tmp_path, count=4)
    data = paths[1].read_bytes()
    paths[1].write_bytes(data[: len(data) - 40])
    with pytest.raises(DicomSeriesError, match=paths[1].name):
        load_dicom_series(tmp_path)


def test_slice_without_pixel_data_is_reported(tmp_path):
    paths = write_series(tmp_path, count=3, mutate=at(0, lambda ds: delattr(ds, "PixelData")))
    with pytest.raises(DicomSeriesError, match=paths[0].name):
        load_dicom_series(tmp_path)


def test_empty_missing_and_non_directory_paths(tmp_path):
    with pytest.raises(DicomSeriesError, match="No files matching"):
        load_dicom_series(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_dicom_series(tmp_path / "absent")
    (tmp_path / "file.dcm").write_bytes(b"x")
    with pytest.raises(NotADirectoryError):
        load_dicom_series(tmp_path / "file.dcm")


def test_expected_uids_are_enforced(tmp_path):
    write_series(tmp_path, count=3)
    with pytest.raises(DicomSeriesError, match="does not match expected 9.9"):
        load_dicom_series(tmp_path, expected_series_uid="9.9")
    with pytest.raises(DicomSeriesError, match="StudyInstanceUID"):
        load_dicom_series(tmp_path, expected_study_uid="9.9")
    assert load_dicom_series(tmp_path, expected_series_uid=SERIES_UID, expected_study_uid=STUDY_UID).num_slices == 3


# ----------------------------------------------------------------------------- warnings for unverifiable series

def test_missing_position_tags_warn_and_are_reported(tmp_path):
    write_series(tmp_path, count=4, mutate=lambda i, ds: without_position_tags(ds))
    with pytest.warns(DicomSeriesWarning, match="InstanceNumber alone"):
        volume = load_dicom_series(tmp_path)
    assert pixel_values(volume) == [10.0, 20.0, 30.0, 40.0]
    assert volume.metadata["Slice Order Validation"].startswith("unvalidated")
    assert "Acquisition Plane (DICOM orientation)" not in volume.metadata


def test_missing_slice_warns_about_numbering_gap_and_uneven_spacing(tmp_path):
    write_series(tmp_path, instance_numbers=[1, 2, 4, 5])
    with pytest.warns(DicomSeriesWarning) as record:
        assert load_dicom_series(tmp_path).num_slices == 4
    messages = " | ".join(str(w.message) for w in record)
    assert "not contiguous" in messages and "not uniform" in messages


def test_uneven_slice_spacing_warns(tmp_path):
    write_series(tmp_path, count=4, mutate=at(2, lambda ds: setattr(ds, "ImagePositionPatient", [0.0, 0.0, 12.0])))
    with pytest.warns(DicomSeriesWarning, match="not uniform"):
        assert load_dicom_series(tmp_path).num_slices == 4


def test_pixel_spacing_disagreement_warns(tmp_path):
    write_series(tmp_path, count=3, mutate=at(1, lambda ds: setattr(ds, "PixelSpacing", [0.9, 0.9])))
    with pytest.warns(DicomSeriesWarning, match="PixelSpacing"):
        load_dicom_series(tmp_path)
