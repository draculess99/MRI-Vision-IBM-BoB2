"""5-fold ensemble RSNA inference: average predictions across multiple checkpoints.

Reuses the existing dataset preparation, model architecture, and exception framework
from rsna_knee_inference. All five checkpoints are validated before any inference
occurs; a single invalid fold fails the entire ensemble.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .dicom_series import DicomSeriesError
from .rsna_knee_dataset import TARGET_COLUMNS, RSNAKneeMetadata
from .rsna_knee_model import RSNAKneeCNN
from .rsna_knee_train import RSNAKneeDicomDataset

# Reuse existing exception classes from rsna_knee_inference
from .rsna_knee_inference import (
    CheckpointNotFoundError,
    MalformedCheckpointError,
    TargetMismatchError,
    ArchitectureMismatchError,
    StudyUnavailableError,
    RSNAInferenceError,
    REQUIRED_CHECKPOINT_KEYS,
)


def run_ensemble_inference(
    metadata: RSNAKneeMetadata,
    fold_checkpoint_paths: Sequence[Path | str],
    study_uid: str,
    config: Optional[dict] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """Run one study through all five folds and return ensemble-averaged probabilities.

    Args:
        metadata: Loaded RSNA metadata (from load_rsna_metadata / load_rsna_metadata_safe).
        fold_checkpoint_paths: Sequence of exactly 5 checkpoint paths, one per fold.
            Each must be a file saved by rsna_knee_train.train_rsna.
        study_uid: StudyInstanceUID to run inference on. Must have a labeled row in
            metadata.train and locally available DICOMs.
        config: Optional override for dataset preparation. Defaults to the first
            fold checkpoint's config, keeping input shape consistent with training.
        device: Torch device string. Inference always loads checkpoints on CPU first;
            this only controls where the forward pass runs.

    Returns:
        Dict mapping each of the 12 TARGET_COLUMNS (in that exact order) to the
        arithmetic mean of the sigmoid probability across all five folds.

    Raises:
        CheckpointNotFoundError, MalformedCheckpointError, TargetMismatchError,
        ArchitectureMismatchError, StudyUnavailableError.

    Note:
        All five checkpoints are validated before any inference occurs. If any
        checkpoint is missing, invalid, or incompatible, an exception is raised
        without returning a partial ensemble.
    """
    fold_checkpoint_paths = [Path(p) for p in fold_checkpoint_paths]
    if len(fold_checkpoint_paths) != 5:
        raise ValueError(
            f"Expected exactly 5 fold checkpoint paths, got {len(fold_checkpoint_paths)}"
        )

    # ======================================================================
    # Phase 1: Validate all five checkpoints upfront (fail-fast)
    # ======================================================================
    loaded_checkpoints = []
    models = []

    for fold_idx, checkpoint_path in enumerate(fold_checkpoint_paths, start=1):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise CheckpointNotFoundError(
                f"Fold {fold_idx}: No checkpoint file at {checkpoint_path}"
            )

        try:
            checkpoint_data = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise MalformedCheckpointError(
                f"Fold {fold_idx}: Failed to load checkpoint {checkpoint_path}: {exc}"
            ) from exc

        if not isinstance(checkpoint_data, dict):
            raise MalformedCheckpointError(
                f"Fold {fold_idx}: Checkpoint did not contain a dict "
                f"(got {type(checkpoint_data).__name__})"
            )

        missing_keys = [
            key for key in REQUIRED_CHECKPOINT_KEYS if key not in checkpoint_data
        ]
        if missing_keys:
            raise MalformedCheckpointError(
                f"Fold {fold_idx}: Checkpoint is missing required fields: {missing_keys}"
            )

        # Validate target columns
        target_columns = checkpoint_data["target_columns"]
        if list(target_columns) != list(TARGET_COLUMNS):
            raise TargetMismatchError(
                f"Fold {fold_idx}: target_columns {list(target_columns)!r} "
                f"do not match expected TARGET_COLUMNS {list(TARGET_COLUMNS)!r}"
            )

        # Validate and extract config
        checkpoint_config = checkpoint_data["config"]
        if not isinstance(checkpoint_config, dict):
            raise MalformedCheckpointError(
                f"Fold {fold_idx}: 'config' field is not a dict "
                f"(got {type(checkpoint_config).__name__})"
            )

        try:
            # Support both Kaggle format (num_planes) and existing format (planes list)
            if "planes" in checkpoint_config:
                planes = tuple(checkpoint_config.get("planes", ("Axial", "Coronal", "Sagittal")))
                num_planes = len(planes)
            else:
                num_planes = int(checkpoint_config.get("num_planes", 3))
                planes = None
            dropout = float(checkpoint_config["dropout"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedCheckpointError(
                f"Fold {fold_idx}: 'config' is incomplete or invalid: {exc}"
            ) from exc

        # Instantiate model
        try:
            model = RSNAKneeCNN(num_planes=num_planes, dropout=dropout)
        except ValueError as exc:
            raise MalformedCheckpointError(
                f"Fold {fold_idx}: config produced an invalid model: {exc}"
            ) from exc

        # Load state dict
        if not isinstance(checkpoint_data["model"], dict):
            raise MalformedCheckpointError(
                f"Fold {fold_idx}: 'model' field is not a state dict "
                f"(got {type(checkpoint_data['model']).__name__})"
            )

        try:
            model.load_state_dict(checkpoint_data["model"])
        except (RuntimeError, ValueError) as exc:
            raise ArchitectureMismatchError(
                f"Fold {fold_idx}: weights are not compatible with the current "
                f"RSNAKneeCNN: {exc}"
            ) from exc

        model.eval()
        try:
            torch_device = torch.device(device)
            model = model.to(torch_device)
        except (RuntimeError, ValueError) as exc:
            raise RSNAInferenceError(
                f"Fold {fold_idx}: Could not move model to device {device!r}: {exc}"
            ) from exc

        loaded_checkpoints.append(checkpoint_data)
        models.append(model)

    # ======================================================================
    # Phase 2: Prepare the study (deterministic, once)
    # ======================================================================
    study_rows = metadata.train[
        metadata.train["StudyInstanceUID"].astype(str) == str(study_uid)
    ]
    if study_rows.empty:
        raise StudyUnavailableError(
            f"StudyInstanceUID={study_uid} has no labeled row in metadata.train"
        )

    # Use first checkpoint's config as default; allow override
    raw_config = (
        config if config is not None else loaded_checkpoints[0]["config"]
    )

    # Normalize config for dataset preparation:
    # - Kaggle checkpoints use 'slices_per_plane' but dataset expects 'slices_per_series'
    # - Support both 'planes' list and 'num_planes' int formats
    effective_config = dict(raw_config)
    if "slices_per_series" not in effective_config and "slices_per_plane" in effective_config:
        effective_config["slices_per_series"] = effective_config["slices_per_plane"]

    try:
        dataset = RSNAKneeDicomDataset(
            metadata, study_rows, effective_config, split="train"
        )
        item = dataset[0]
    except (FileNotFoundError, ValueError, KeyError, DicomSeriesError) as exc:
        raise StudyUnavailableError(
            f"StudyInstanceUID={study_uid}: could not prepare study for inference: {exc}"
        ) from exc

    images = item["images"].unsqueeze(0).to(torch_device)
    mask = item["mask"].unsqueeze(0).to(torch_device)

    # ======================================================================
    # Phase 3: Run through all five models and collect probabilities
    # ======================================================================
    fold_probabilities = []

    for fold_idx, model in enumerate(models, start=1):
        try:
            with torch.inference_mode():
                logits = model(images, mask)
                probabilities = logits.sigmoid().squeeze(0).cpu().numpy()
        except (RuntimeError, ValueError) as exc:
            raise ArchitectureMismatchError(
                f"Fold {fold_idx}: Forward pass failed for StudyInstanceUID={study_uid}; "
                f"the checkpoint's architecture may not match the prepared input: {exc}"
            ) from exc

        fold_probabilities.append(probabilities)

    # ======================================================================
    # Phase 4: Ensemble output (arithmetic mean)
    # ======================================================================
    ensemble_probabilities = np.mean(fold_probabilities, axis=0)

    return {
        target: float(ensemble_probabilities[index])
        for index, target in enumerate(TARGET_COLUMNS)
    }
