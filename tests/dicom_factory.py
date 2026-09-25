"""Synthetic DICOM series for the series-loader and RSNA adapter tests."""

import struct
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.tag import Tag
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

AXIAL = (1, 0, 0, 0, 1, 0)
CORONAL = (1, 0, 0, 0, 0, -1)
SAGITTAL = (0, 1, 0, 0, 0, -1)
STUDY_UID = "1.2.826.1"
SERIES_UID = "1.2.826.2"


def make_slice(number, position, *, orientation=AXIAL, series_uid=SERIES_UID, study_uid=STUDY_UID, rows=8, columns=8,
               textured=False):
    """One MR slice, positioned `position` mm along its slice normal.

    Pixels all equal 10 * InstanceNumber, or with textured=True form a deterministic non-constant pattern
    (constant slices normalize to all zeros, which would make numerical comparisons vacuous).
    """
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset("slice.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "MR"
    ds.SeriesDescription = "SYNTH SERIES"
    ds.InstanceNumber = number
    ds.ImageOrientationPatient = [float(v) for v in orientation]
    normal = np.cross(orientation[:3], orientation[3:])
    ds.ImagePositionPatient = [float(v) for v in normal * position]
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 3.0
    ds.SpacingBetweenSlices = 4.0
    ds.Rows, ds.Columns = rows, columns
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    if textured:
        pixels = (np.arange(rows * columns, dtype=np.int64).reshape(rows, columns) * (number + 3)) % 997 + number
    else:
        pixels = np.full((rows, columns), number * 10)
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    return ds


def write_series(directory, count=4, *, orientation=AXIAL, spacing=5.0, instance_numbers=None, filenames=None,
                 mutate=None, series_uid=SERIES_UID, study_uid=STUDY_UID, rows=8, columns=8, textured=False):
    """Write a series and return its paths in file-write order.

    Physical position follows InstanceNumber ((number - 1) * spacing). By default filenames descend as
    InstanceNumber ascends, so sorting by filename gives the wrong order. `mutate(index, ds)` may edit each
    dataset (index is the write position) before it is saved.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    numbers = list(instance_numbers) if instance_numbers is not None else list(range(1, count + 1))
    paths = []
    for index, number in enumerate(numbers):
        ds = make_slice(number, (number - 1) * spacing, orientation=orientation, series_uid=series_uid,
                        study_uid=study_uid, rows=rows, columns=columns, textured=textured)
        if mutate is not None:
            mutate(index, ds)
        name = filenames[index] if filenames else f"{len(numbers) - index:03d}.dcm"
        path = directory / name
        ds.save_as(path)
        paths.append(path)
    return paths


def corrupt_element(path, keyword, replacement):
    """Overwrite a saved DS/IS element's text in place, keeping its encoded length.

    pydicom refuses to write non-numeric numbers, so this is how a corrupt real file is simulated.
    """
    tag = Tag(keyword)
    data = bytearray(Path(path).read_bytes())
    start = bytes(data).find(struct.pack("<HH", tag.group, tag.element))
    assert start >= 0, f"{keyword} not found in {path}"
    assert bytes(data[start + 4:start + 6]) in (b"DS", b"IS"), f"{keyword} is not a DS/IS element"
    (length,) = struct.unpack("<H", data[start + 6:start + 8])
    assert len(replacement) <= length, f"{replacement!r} does not fit in {length} bytes"
    data[start + 8:start + 8 + length] = replacement.ljust(length).encode("ascii")
    Path(path).write_bytes(bytes(data))


def at(index, edit):
    """Build a mutate hook that applies edit(ds) to one slice only."""
    return lambda i, ds: edit(ds) if i == index else None
