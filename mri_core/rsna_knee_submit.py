"""Load an RSNA checkpoint and write the exact sample-submission schema."""

from pathlib import Path

import pandas as pd
import torch

from .rsna_knee_dataset import SUBMISSION_COLUMNS, TARGET_COLUMNS, load_rsna_metadata
from .rsna_knee_model import RSNAKneeCNN
from .rsna_knee_train import RSNAKneeDicomDataset


def write_submission(data_dir, checkpoint, output, batch_size=8, num_workers=0, device="auto", dicom_root=None):
    metadata = load_rsna_metadata(data_dir, dicom_root)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_data.get("target_columns") != list(TARGET_COLUMNS):
        raise ValueError("Checkpoint target order does not match RSNA schema")
    config = checkpoint_data["config"]
    selected_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    model = RSNAKneeCNN(len(config.get("planes", ("Axial", "Coronal", "Sagittal"))), float(config["dropout"])).to(selected_device)
    model.load_state_dict(checkpoint_data["model"]); model.eval()
    dataset = RSNAKneeDicomDataset(metadata, metadata.test, config, split="test")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    predictions = {}
    with torch.inference_mode():
        for batch in loader:
            scores = model(batch["images"].to(selected_device), batch["mask"].to(selected_device)).sigmoid().cpu().numpy()
            predictions.update({uid: row for uid, row in zip(batch["study_uid"], scores)})
    template = metadata.sample_submission.copy()
    if tuple(template.columns) != SUBMISSION_COLUMNS:
        raise ValueError("Submission template column order changed")
    result = pd.DataFrame({"StudyInstanceUID": template["StudyInstanceUID"].astype("string")})
    for index, target in enumerate(TARGET_COLUMNS):
        result[target] = result["StudyInstanceUID"].map(lambda uid: float(predictions[str(uid)][index]))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
