"""Series-level DICOM loading: one directory of 2D slices becomes one ordered MRIVolume.

Slices are ordered by InstanceNumber, never by filename. When every slice carries
ImagePositionPatient and ImageOrientationPatient, that order is checked against the
physical slice positions. Structural problems raise DicomSeriesError; conditions that
only reduce confidence in the result emit DicomSeriesWarning.
"""

import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pydicom

from .mri_volume import MRIVolume

ORIENTATION_ATOL = 1e-3
SPACING_RTOL = 0.05
_PLANE_BY_AXIS = ("Sagittal", "Coronal", "Axial")  # patient x, y, z


class DicomSeriesError(ValueError):
    """The directory does not form a single, orderable, readable DICOM series."""


class DicomSeriesWarning(UserWarning):
    """The series loaded, but part of it could not be verified or looks irregular."""


def _warn(message: str) -> None:
    warnings.warn(message, DicomSeriesWarning, stacklevel=3)


def _listing(items: List[str], limit: int = 10) -> str:
    shown = "\n  ".join(items[:limit])
    return shown + (f"\n  ... and {len(items) - limit} more" if len(items) > limit else "")


def _vector(ds, name: str, length: int, filename: str) -> np.ndarray:
    try:
        values = np.array([float(v) for v in getattr(ds, name)], dtype=float)
    except (TypeError, ValueError) as exc:
        raise DicomSeriesError(f"{filename}: {name} is malformed: {exc}") from exc
    if values.shape != (length,) or not np.isfinite(values).all():
        raise DicomSeriesError(f"{filename}: {name} must be {length} finite numbers, got {values.tolist()}")
    return values


def _number(cast, value, name: str, filename: str):
    try:
        return cast(value)
    except (TypeError, ValueError) as exc:
        raise DicomSeriesError(f"{filename}: {name} {str(value)!r} is not a valid number: {exc}") from exc


def _read_slices(directory: Path, files: List[Path]):
    slices, failures = [], []
    for path in files:
        try:
            ds = pydicom.dcmread(path)
            pixels = ds.pixel_array
        except Exception as exc:
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if pixels.ndim != 2:
            failures.append(f"{path.name}: expected one 2D frame, got pixel array shape {pixels.shape}")
            continue
        slices.append((path, ds, pixels))
    if failures:
        raise DicomSeriesError(f"{len(failures)} of {len(files)} files in {directory} are not readable 2D DICOM slices:\n  "
                               + _listing(failures))
    return slices


def _require_unique_value(values, what: str, directory: Path):
    distinct = sorted({str(v) for v in values})
    if len(distinct) != 1:
        raise DicomSeriesError(f"{directory}: slices disagree on {what}: {distinct}")
    return distinct[0]


def _validate_positions(ordered) -> Dict[str, Any]:
    """Check InstanceNumber order against physical position. Returns metadata about the check."""
    complete = sum(hasattr(ds, "ImagePositionPatient") and hasattr(ds, "ImageOrientationPatient") for _, ds, _ in ordered)
    if complete == 0:
        _warn("ImagePositionPatient/ImageOrientationPatient are absent; slice order rests on InstanceNumber alone")
        return {"Slice Order Validation": "unvalidated (no position tags)"}
    if complete != len(ordered):
        missing = [p.name for p, ds, _ in ordered
                   if not (hasattr(ds, "ImagePositionPatient") and hasattr(ds, "ImageOrientationPatient"))]
        raise DicomSeriesError("Only some slices carry ImagePositionPatient/ImageOrientationPatient; missing on:\n  "
                               + _listing(missing))

    orientation = np.array([_vector(ds, "ImageOrientationPatient", 6, p.name) for p, ds, _ in ordered])
    deviating = [f"{ordered[i][0].name} (InstanceNumber {int(ordered[i][1].InstanceNumber)})"
                 for i in np.flatnonzero(~np.isclose(orientation, orientation[0], atol=ORIENTATION_ATOL).all(axis=1))]
    if deviating:
        raise DicomSeriesError("Inconsistent ImageOrientationPatient; these slices differ from the first:\n  " + _listing(deviating))
    normal = np.cross(orientation[0][:3], orientation[0][3:])
    length = float(np.linalg.norm(normal))
    if length < 0.5:
        raise DicomSeriesError(f"ImageOrientationPatient {orientation[0].tolist()} does not define a slice plane")
    normal /= length

    result: Dict[str, Any] = {"Slice Order Validation": "position (ImagePositionPatient)"}
    axis = int(np.argmax(np.abs(normal)))
    result["Acquisition Plane (DICOM orientation)"] = _PLANE_BY_AXIS[axis] if abs(normal[axis]) >= 0.9 else "Oblique"

    positions = np.array([_vector(ds, "ImagePositionPatient", 3, p.name) for p, ds, _ in ordered]) @ normal
    result["Slice Position Range (mm)"] = [round(float(positions.min()), 3), round(float(positions.max()), 3)]
    steps = np.diff(positions)
    if steps.size == 0:
        return result
    if not (np.all(steps > 0) or np.all(steps < 0)):
        direction = 1 if np.sum(steps > 0) >= np.sum(steps < 0) else -1
        bad = [f"InstanceNumber {int(ordered[i][1].InstanceNumber)} -> {int(ordered[i + 1][1].InstanceNumber)} "
               f"(step {steps[i]:+.3f} mm)" for i in range(steps.size) if steps[i] * direction <= 0]
        raise DicomSeriesError("InstanceNumber order is not monotonic in physical slice position:\n  " + _listing(bad))
    spacing = np.abs(steps)
    median = float(np.median(spacing))
    result["Measured Slice Spacing (mm)"] = round(median, 4)
    if np.any(np.abs(spacing - median) > SPACING_RTOL * median):
        _warn(f"Slice spacing is not uniform (min {spacing.min():.3f} mm, max {spacing.max():.3f} mm); slices may be missing")
    return result


def load_dicom_series(
    directory,
    *,
    pattern: str = "*",
    expected_series_uid: Optional[str] = None,
    expected_study_uid: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> MRIVolume:
    """Load one DICOM series directory (non-recursive) into an MRIVolume shaped (slices, rows, columns).

    Every non-hidden file matching ``pattern`` must be a readable single-frame DICOM slice;
    nothing is skipped silently. Raw values are float32 with RescaleSlope/Intercept applied.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"DICOM series directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    files = sorted(p for p in directory.glob(pattern) if p.is_file() and not p.name.startswith("."))
    if not files:
        raise DicomSeriesError(f"No files matching {pattern!r} in {directory}")

    slices = _read_slices(directory, files)
    names = [p.name for p, _, _ in slices]

    missing = [p.name for p, ds, _ in slices if getattr(ds, "InstanceNumber", None) is None]
    if missing:
        raise DicomSeriesError("InstanceNumber is required for ordering but missing on:\n  " + _listing(missing))
    numbers = [_number(int, ds.InstanceNumber, "InstanceNumber", p.name) for p, ds, _ in slices]
    duplicates = {n: c for n, c in Counter(numbers).items() if c > 1}
    if duplicates:
        detail = [f"InstanceNumber {n}: " + ", ".join(nm for nm, k in zip(names, numbers) if k == n) for n in sorted(duplicates)]
        raise DicomSeriesError("Duplicate InstanceNumber values; slice order is ambiguous:\n  " + _listing(detail))

    series_uids = [getattr(ds, "SeriesInstanceUID", None) for _, ds, _ in slices]
    if any(uid is None for uid in series_uids):
        raise DicomSeriesError(f"{directory}: some slices have no SeriesInstanceUID")
    series_uid = _require_unique_value(series_uids, "SeriesInstanceUID", directory)
    if expected_series_uid is not None and series_uid != str(expected_series_uid):
        raise DicomSeriesError(f"{directory}: SeriesInstanceUID {series_uid} does not match expected {expected_series_uid}")
    if expected_study_uid is not None:
        study_uid = _require_unique_value([getattr(ds, "StudyInstanceUID", None) for _, ds, _ in slices],
                                          "StudyInstanceUID", directory)
        if study_uid != str(expected_study_uid):
            raise DicomSeriesError(f"{directory}: StudyInstanceUID {study_uid} does not match expected {expected_study_uid}")
    _require_unique_value([p.shape for _, _, p in slices], "pixel array shape", directory)
    photometric = _require_unique_value([str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).strip().upper()
                                         for _, ds, _ in slices], "PhotometricInterpretation", directory)

    ordered = [slices[i] for i in sorted(range(len(slices)), key=lambda i: numbers[i])]
    ordered_numbers = sorted(numbers)
    if ordered_numbers[-1] - ordered_numbers[0] + 1 != len(ordered_numbers):
        _warn(f"InstanceNumber values are not contiguous ({ordered_numbers[0]}..{ordered_numbers[-1]} for {len(ordered)} slices)")

    metadata: Dict[str, Any] = {
        "Series Instance UID": series_uid,
        "Slice Count": len(ordered),
        "Instance Number Range": [ordered_numbers[0], ordered_numbers[-1]],
        "Photometric Interpretation": photometric,
    }
    metadata.update(_validate_positions(ordered))

    first = ordered[0][1]
    for attribute, label in (("Modality", "Modality"), ("SeriesDescription", "Series Description"),
                             ("Rows", "Rows"), ("Columns", "Columns")):
        value = getattr(first, attribute, None)
        if value is not None and str(value).strip() != "":
            metadata[label] = value if isinstance(value, int) else str(value)
    for attribute, label in (("SliceThickness", "Slice Thickness"), ("SpacingBetweenSlices", "Spacing Between Slices")):
        value = getattr(first, attribute, None)
        if value is not None and str(value).strip() != "":
            metadata[label] = _number(float, value, attribute, ordered[0][0].name)
    spacings = [_vector(ds, "PixelSpacing", 2, p.name).tolist() for p, ds, _ in ordered if hasattr(ds, "PixelSpacing")]
    if spacings:
        metadata["Pixel Spacing"] = spacings[0]
        if len(spacings) != len(ordered) or not np.allclose(spacings, spacings[0], rtol=1e-3):
            _warn("PixelSpacing is missing on some slices or differs between slices")
    metadata.update(extra_metadata or {})

    volume = np.empty((len(ordered), *ordered[0][2].shape), dtype=np.float32)
    for index, (path, ds, pixels) in enumerate(ordered):
        data = pixels.astype(np.float32)
        slope = _number(float, getattr(ds, "RescaleSlope", 1.0), "RescaleSlope", path.name)
        intercept = _number(float, getattr(ds, "RescaleIntercept", 0.0), "RescaleIntercept", path.name)
        volume[index] = data * slope + intercept if (slope != 1.0 or intercept != 0.0) else data

    return MRIVolume(data=volume, metadata=metadata, format_type="DICOM",
                     is_inverted=(photometric == "MONOCHROME1"), slice_axis=0)
