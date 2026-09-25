"""Frozen copy of RSNAKneeDicomDataset.__getitem__ from before it read series through MRIVolume.

Kept only as the regression reference: the MRIVolume-backed dataset must reproduce this output.
It deliberately preserves the old behaviour of silently skipping unreadable files.
"""

import numpy as np
import pydicom
import torch
import torch.nn.functional as F

from mri_core.rsna_knee_dataset import PLANE_NAMES, TARGET_COLUMNS
from mri_core.rsna_knee_train import _normalize


def legacy_series_arrays(directory):
    """Pixel arrays sorted by InstanceNumber, exactly as the previous reading block produced them."""
    files = sorted(directory.rglob("*")) if directory.is_dir() else []
    datasets = []
    for path in files:
        if not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(path, force=False)
            if hasattr(dataset, "pixel_array"):
                datasets.append((getattr(dataset, "InstanceNumber", 0), dataset.pixel_array))
        except Exception:
            continue
    datasets.sort(key=lambda item: item[0])
    return [array for _, array in datasets]


def legacy_item(metadata, row, config, split="train", select=None):
    """select(candidates) -> SeriesInstanceUID chooses the series per plane; defaults to the original
    first-CSV-row pick so old callers keep testing the pre-selection-change behaviour verbatim.
    """
    select = select or (lambda candidates: str(candidates.iloc[0]["SeriesInstanceUID"]))
    size = int(config["image_size"])
    slices = int(config["slices_per_series"])
    planes = tuple(config.get("planes", list(PLANE_NAMES)))
    output = torch.zeros(len(planes), slices, 1, size, size, dtype=torch.float32)
    mask = torch.zeros(len(planes), slices, dtype=torch.bool)
    series_rows = metadata.series_for(row["StudyInstanceUID"], split)
    for plane_index, plane in enumerate(planes):
        candidates = series_rows[series_rows["Anatomical_Plane"].astype(str).str.lower() == plane.lower()]
        if candidates.empty:
            continue
        series_uid = select(candidates)
        arrays = legacy_series_arrays(metadata.dicom_series_dir(row["StudyInstanceUID"], series_uid, split))
        if not arrays:
            continue
        chosen = np.linspace(0, len(arrays) - 1, min(slices, len(arrays)), dtype=int)
        for slice_index, source_index in enumerate(chosen):
            image = torch.from_numpy(_normalize(arrays[source_index])).unsqueeze(0).unsqueeze(0)
            output[plane_index, slice_index] = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)[0]
            mask[plane_index, slice_index] = True
    if not mask.any(dim=1).all():
        raise ValueError(f"No readable DICOM slices for {row['StudyInstanceUID']}")
    labels = torch.tensor(row[list(TARGET_COLUMNS)].to_numpy(dtype=np.float32)) if split == "train" else torch.zeros(len(TARGET_COLUMNS))
    return {"images": output, "mask": mask, "labels": labels, "study_uid": str(row["StudyInstanceUID"])}
