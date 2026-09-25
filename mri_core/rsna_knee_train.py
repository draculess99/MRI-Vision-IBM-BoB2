"""Study-level RSNA Knee training; metadata-only datasets stop before image access."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .rsna_knee_dataset import (PLANE_NAMES, TARGET_COLUMNS, RSNAKneeMetadata,
                                load_series_volume, select_series, split_labeled_studies)
from .rsna_knee_model import RSNAKneeCNN


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_auc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def calculate_auc(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    per_label = {target: safe_auc(labels[:, index], probabilities[:, index])
                 for index, target in enumerate(TARGET_COLUMNS)}
    available = [value for value in per_label.values() if value is not None]
    return {"per_label": per_label, "macro_roc_auc": float(np.mean(available)) if available else None}


def _normalize(array):
    array = np.asarray(array, dtype=np.float32)
    low, high = np.percentile(array, [1, 99])
    if high <= low:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0, 1)


class RSNAKneeDicomDataset(Dataset):
    """Reads each study's series through load_series_volume; a malformed series raises instead of being skipped."""

    def __init__(self, metadata: RSNAKneeMetadata, studies, config, split="train"):
        self.metadata, self.studies, self.config, self.split = metadata, studies.reset_index(drop=True), config, split
        if not len(self.studies):
            raise ValueError(f"No studies in {split} split")
        if not metadata.dicom_split_dir(split).is_dir():
            raise FileNotFoundError(f"DICOM directory unavailable: {metadata.dicom_split_dir(split)}")

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, index):
        row = self.studies.iloc[index]
        study_uid = str(row["StudyInstanceUID"])
        size = int(self.config["image_size"])
        slices = int(self.config["slices_per_series"])
        planes = tuple(self.config.get("planes", list(PLANE_NAMES)))
        output = torch.zeros(len(planes), slices, 1, size, size, dtype=torch.float32)
        mask = torch.zeros(len(planes), slices, dtype=torch.bool)
        series_rows = self.metadata.series_for(study_uid, self.split)
        for plane_index, plane in enumerate(planes):
            candidates = series_rows[series_rows["Anatomical_Plane"].astype(str).str.lower() == plane.lower()]
            if candidates.empty:
                raise ValueError(f"StudyInstanceUID={study_uid}: no {plane} series listed in the {self.split} metadata")
            series_uid = select_series(candidates)
            # Malformed or unreadable series raise here (with both UIDs); nothing is skipped.
            volume = load_series_volume(self.metadata, study_uid, series_uid, self.split)
            chosen = np.linspace(0, volume.num_slices - 1, min(slices, volume.num_slices), dtype=int)
            for slice_index, source_index in enumerate(chosen):
                image = torch.from_numpy(_normalize(volume.get_slice(int(source_index)))).unsqueeze(0).unsqueeze(0)
                output[plane_index, slice_index] = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)[0]
                mask[plane_index, slice_index] = True
        labels = torch.tensor(row[list(TARGET_COLUMNS)].to_numpy(dtype=np.float32)) if self.split == "train" else torch.zeros(len(TARGET_COLUMNS))
        return {"images": output, "mask": mask, "labels": labels, "study_uid": str(row["StudyInstanceUID"])}


def _macro_loss(losses):
    return float(np.mean(losses)) if losses else None


def train_rsna(metadata: RSNAKneeMetadata, config: dict, checkpoint_path: Path):
    seed_everything(int(config["seed"]))
    train_frame, valid_frame = split_labeled_studies(metadata, float(config["validation_fraction"]), int(config["seed"]))
    if not metadata.dicom_split_dir("train").is_dir():
        raise FileNotFoundError("RSNA DICOM files are not available; inspect supports metadata-only operation, training requires images")
    device_name = config.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    train_data = RSNAKneeDicomDataset(metadata, train_frame, config, "train")
    valid_data = RSNAKneeDicomDataset(metadata, valid_frame, config, "train")
    train_loader = DataLoader(train_data, batch_size=int(config["batch_size"]), shuffle=True,
                              num_workers=int(config["num_workers"]))
    valid_loader = DataLoader(valid_data, batch_size=int(config["batch_size"]), shuffle=False,
                              num_workers=int(config["num_workers"]))
    model = RSNAKneeCNN(len(config.get("planes", PLANE_NAMES)), float(config["dropout"])).to(device)
    labels = train_frame[list(TARGET_COLUMNS)].to_numpy(dtype=np.float32)
    positive = labels.sum(axis=0)
    pos_weight = torch.tensor((len(labels) - positive) / np.maximum(positive, 1), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    best_auc = -np.inf
    history, started = [], time.perf_counter()
    for epoch in range(int(config["epochs"])):
        model.train(); losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["images"].to(device), batch["mask"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); valid_losses, valid_labels, valid_scores = [], [], []
        with torch.no_grad():
            for batch in valid_loader:
                logits = model(batch["images"].to(device), batch["mask"].to(device))
                valid_losses.append(float(criterion(logits, batch["labels"].to(device)).cpu()))
                valid_labels.append(batch["labels"].numpy()); valid_scores.append(logits.sigmoid().cpu().numpy())
        labels_array, scores_array = np.vstack(valid_labels), np.vstack(valid_scores)
        auc = calculate_auc(labels_array, scores_array)
        score = auc["macro_roc_auc"] if auc["macro_roc_auc"] is not None else -np.inf
        record = {"epoch": epoch + 1, "loss": _macro_loss(losses), "validation_loss": _macro_loss(valid_losses), **auc}
        history.append(record)
        if epoch == 0 or score > best_auc:
            best_auc = score
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": config, "target_columns": list(TARGET_COLUMNS),
                        "epoch": epoch + 1, "validation": record, "device": str(device)}, checkpoint_path)
    return {"checkpoint": str(checkpoint_path), "device": str(device), "train_studies": len(train_frame),
            "validation_studies": len(valid_frame), "seconds": time.perf_counter() - started, "history": history}
