"""Single-study RSNA inference: checkpoint -> per-target probabilities.

Reuses the existing dataset (RSNAKneeDicomDataset), model (RSNAKneeCNN), and metadata
infrastructure. No DICOM parsing, normalization, slice sampling, or plane-selection
logic is duplicated here. No trained RSNA checkpoint is created or assumed to exist;
callers must supply a real checkpoint path, and every failure mode is reported through
a specific, catchable exception rather than a crash.
"""

from pathlib import Path
from typing import Dict, Optional

import torch

from .dicom_series import DicomSeriesError
from .rsna_knee_dataset import TARGET_COLUMNS, RSNAKneeMetadata
from .rsna_knee_model import RSNAKneeCNN
from .rsna_knee_train import RSNAKneeDicomDataset

REQUIRED_CHECKPOINT_KEYS = ("model", "config", "target_columns")


class RSNAInferenceError(Exception):
    """Base class for all single-study RSNA inference failures."""


class CheckpointNotFoundError(RSNAInferenceError):
    """No file exists at the given checkpoint path."""


class MalformedCheckpointError(RSNAInferenceError):
    """Checkpoint could not be loaded, or is missing required fields."""


class TargetMismatchError(RSNAInferenceError):
    """Checkpoint's stored target_columns do not exactly match TARGET_COLUMNS."""


class ArchitectureMismatchError(RSNAInferenceError):
    """Checkpoint weights are not compatible with the current RSNAKneeCNN architecture."""


class StudyUnavailableError(RSNAInferenceError):
    """The requested study is missing from metadata, or its local DICOM data is
    missing/incomplete/malformed."""


def run_inference(
    metadata: RSNAKneeMetadata,
    checkpoint_path: Path | str,
    study_uid: str,
    config: Optional[dict] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """Run one study through a checkpointed RSNAKneeCNN and return per-target probabilities.

    Args:
        metadata: Loaded RSNA metadata (from load_rsna_metadata / load_rsna_metadata_safe).
        checkpoint_path: Path to a checkpoint saved by rsna_knee_train.train_rsna.
        study_uid: StudyInstanceUID to run inference on. Must have a labeled row in
            metadata.train (the dataset's label-lookup path) and locally available DICOMs.
        config: Optional override for dataset preparation (image_size, slices_per_series,
            planes). Defaults to the checkpoint's own stored config, keeping the input
            shape consistent with what the checkpoint's weights were trained for.
        device: Torch device string. Inference always loads the checkpoint on CPU first;
            this only controls where the forward pass runs.

    Returns:
        Dict mapping each of the 12 TARGET_COLUMNS (in that exact order) to a sigmoid
        probability in [0.0, 1.0].

    Raises:
        CheckpointNotFoundError, MalformedCheckpointError, TargetMismatchError,
        ArchitectureMismatchError, StudyUnavailableError.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise CheckpointNotFoundError(f"No checkpoint file at {checkpoint_path}")

    try:
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise MalformedCheckpointError(f"Failed to load checkpoint {checkpoint_path}: {exc}") from exc

    if not isinstance(checkpoint_data, dict):
        raise MalformedCheckpointError(
            f"Checkpoint {checkpoint_path} did not contain a dict (got {type(checkpoint_data).__name__})"
        )
    missing_keys = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in checkpoint_data]
    if missing_keys:
        raise MalformedCheckpointError(f"Checkpoint {checkpoint_path} is missing required fields: {missing_keys}")

    target_columns = checkpoint_data["target_columns"]
    if list(target_columns) != list(TARGET_COLUMNS):
        raise TargetMismatchError(
            f"Checkpoint target_columns {list(target_columns)!r} do not match "
            f"expected TARGET_COLUMNS {list(TARGET_COLUMNS)!r}"
        )

    checkpoint_config = checkpoint_data["config"]
    if not isinstance(checkpoint_config, dict):
        raise MalformedCheckpointError(
            f"Checkpoint {checkpoint_path} 'config' field is not a dict (got {type(checkpoint_config).__name__})"
        )

    try:
        planes = tuple(checkpoint_config.get("planes", ("Axial", "Coronal", "Sagittal")))
        dropout = float(checkpoint_config["dropout"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MalformedCheckpointError(f"Checkpoint {checkpoint_path} 'config' is incomplete or invalid: {exc}") from exc

    try:
        model = RSNAKneeCNN(num_planes=len(planes), dropout=dropout)
    except ValueError as exc:
        raise MalformedCheckpointError(f"Checkpoint {checkpoint_path} config produced an invalid model: {exc}") from exc

    if not isinstance(checkpoint_data["model"], dict):
        raise MalformedCheckpointError(
            f"Checkpoint {checkpoint_path} 'model' field is not a state dict (got {type(checkpoint_data['model']).__name__})"
        )
    try:
        model.load_state_dict(checkpoint_data["model"])
    except (RuntimeError, ValueError) as exc:
        raise ArchitectureMismatchError(
            f"Checkpoint {checkpoint_path} weights are not compatible with the current RSNAKneeCNN: {exc}"
        ) from exc
    model.eval()

    try:
        torch_device = torch.device(device)
        model = model.to(torch_device)
    except (RuntimeError, ValueError) as exc:
        raise RSNAInferenceError(f"Could not move model to device {device!r}: {exc}") from exc

    study_rows = metadata.train[metadata.train["StudyInstanceUID"].astype(str) == str(study_uid)]
    if study_rows.empty:
        raise StudyUnavailableError(f"StudyInstanceUID={study_uid} has no labeled row in metadata.train")

    effective_config = config if config is not None else checkpoint_config
    try:
        dataset = RSNAKneeDicomDataset(metadata, study_rows, effective_config, split="train")
        item = dataset[0]
    except (FileNotFoundError, ValueError, KeyError, DicomSeriesError) as exc:
        raise StudyUnavailableError(
            f"StudyInstanceUID={study_uid}: could not prepare study for inference: {exc}"
        ) from exc

    images = item["images"].unsqueeze(0).to(torch_device)
    mask = item["mask"].unsqueeze(0).to(torch_device)

    try:
        with torch.inference_mode():
            logits = model(images, mask)
            probabilities = logits.sigmoid().squeeze(0).cpu().numpy()
    except (RuntimeError, ValueError) as exc:
        raise ArchitectureMismatchError(
            f"Forward pass failed for StudyInstanceUID={study_uid}; the checkpoint's "
            f"architecture may not match the prepared input: {exc}"
        ) from exc

    return {target: float(probabilities[index]) for index, target in enumerate(TARGET_COLUMNS)}
