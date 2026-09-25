"""Read-only MRNet exam tensors; PyTorch is an optional dependency."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .baseline_features import sampled_indices
from .labels import TARGETS
from .manifest import PLANES


def normalize_slice(image):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 2 or not np.isfinite(image).all():
        raise ValueError("Expected a finite numeric 2D slice, dimensions >= 2")
    low, high = np.percentile(image, [1, 99])
    normalized = np.clip((image - low) / (high - low), 0, 1) if high > low else np.zeros_like(image)
    return torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)


def augment_plane(slices, generator, rotation=7.0, translation=0.05, contrast=0.1):
    """One affine and contrast transform shared by all valid slices in a plane."""
    draws = torch.rand(4, generator=generator) * 2 - 1
    angle = float(draws[0]) * math.radians(rotation)
    c, s = math.cos(angle), math.sin(angle)
    theta = slices.new_tensor([[c, -s, float(draws[1]) * 2 * translation],
                               [s, c, float(draws[2]) * 2 * translation]])
    theta = theta.unsqueeze(0).expand(slices.shape[0], -1, -1)
    grid = F.affine_grid(theta, slices.shape, align_corners=False)
    transformed = F.grid_sample(slices, grid, align_corners=False, padding_mode="zeros")
    return ((transformed - 0.5) * (1 + float(draws[3]) * contrast) + 0.5).clamp(0, 1)


class MRNetDataset(Dataset):
    def __init__(self, manifest, split, config, *, augment=False, seed=0):
        if split not in ("train", "valid") or (augment and split != "train"):
            raise ValueError("Augmentation is permitted only for the training split")
        self.rows = tuple(row for row in manifest.rows if row["split"] == split)
        if not self.rows:
            raise ValueError(f"Empty {split} split")
        ids = [row["exam_id"] for row in manifest.rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate IDs or split overlap")
        self.paths, self.config = manifest.paths, dict(config)
        self.planes = tuple(config["planes"])
        if not self.planes or len(set(self.planes)) != len(self.planes) or not set(self.planes) <= set(PLANES):
            raise ValueError("Invalid imaging planes")
        if config["slices"] < 1 or config["resolution"] < 16:
            raise ValueError("Require positive slice count and resolution >= 16")
        for row in self.rows:
            if any((row["exam_id"], plane) not in self.paths for plane in self.planes):
                raise ValueError(f"Missing plane for {row['exam_id']}")
            if any(row[target] not in (0, 1) for target in TARGETS):
                raise ValueError("Nonbinary label")
        self.augment, self.seed, self.epoch = augment, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        count, size = self.config["slices"], self.config["resolution"]
        images = torch.zeros(len(self.planes), count, 1, size, size)
        masks = torch.zeros(len(self.planes), count, dtype=torch.bool)
        indices = torch.full((len(self.planes), count), -1, dtype=torch.int64)
        for plane_index, plane in enumerate(self.planes):
            volume = np.load(self.paths[(row["exam_id"], plane)], mmap_mode="r", allow_pickle=False)
            if volume.ndim != 3 or volume.dtype.kind not in "uif" or min(volume.shape) < 1:
                raise ValueError("Expected numeric (slices, height, width) volume")
            selected = sampled_indices(volume.shape[0], count)
            samples = torch.stack([normalize_slice(volume[i]) for i in selected])
            samples = F.interpolate(samples, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
            if self.augment:
                # Independent of worker scheduling, batch order, and global RNG state.
                generator = torch.Generator().manual_seed(
                    self.seed + self.epoch * 1000003 + int(row["exam_id"]) * 17 + plane_index)
                samples = augment_plane(samples, generator, self.config["rotation_degrees"],
                                        self.config["translation_fraction"], self.config["contrast_fraction"])
            length = len(selected)
            images[plane_index, :length] = samples
            masks[plane_index, :length] = True
            indices[plane_index, :length] = torch.as_tensor(selected)
        return {"images": images, "mask": masks, "indices": indices,
                "labels": torch.tensor([row[target] for target in TARGETS], dtype=torch.float32),
                "exam_id": row["exam_id"]}


def training_class_weights(rows):
    """Never include validation labels when estimating BCE positive weights."""
    training = [row for row in rows if row["split"] == "train"]
    if not training:
        raise ValueError("No training rows")
    labels = torch.tensor([[row[t] for t in TARGETS] for row in training], dtype=torch.float32)
    if not torch.isin(labels, torch.tensor([0.0, 1.0])).all():
        raise ValueError("Nonbinary training labels")
    positive = labels.sum(dim=0)
    negative = len(training) - positive
    if (positive == 0).any() or (negative == 0).any():
        raise ValueError("Every training target requires both classes")
    return negative / positive
