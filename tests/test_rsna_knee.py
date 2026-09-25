import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from mri_core.rsna_knee_dataset import (PLANE_NAMES, SERIES_COLUMNS, SUBMISSION_COLUMNS,
    TARGET_COLUMNS, load_rsna_metadata, metadata_report, split_labeled_studies)
from mri_core.rsna_knee_model import RSNAKneeCNN
from mri_core.rsna_knee_train import calculate_auc
from mri_core.rsna_knee_submit import write_submission


def make_metadata(tmp_path, labeled_rows=6):
    root = tmp_path / "rsna"
    root.mkdir()
    ids = [f"study-{i:03d}" for i in range(labeled_rows)] + ["study-missing"]
    rows = []
    for i, uid in enumerate(ids):
        row = {"StudyInstanceUID": uid, "Report": f"report {i}"}
        row.update({target: (i + j) % 2 if i < labeled_rows else np.nan for j, target in enumerate(TARGET_COLUMNS)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    series = []
    for uid in ids:
        for plane in PLANE_NAMES:
            series.append({"StudyInstanceUID": uid, "SeriesInstanceUID": f"series-{uid}-{plane}",
                           "Fluid_Sensitive": 1, "Fat_Suppression": 0, "Anatomical_Plane": plane})
    pd.DataFrame(series, columns=SERIES_COLUMNS).to_csv(root / "train_series.csv", index=False)
    test_ids = ["test-001", "test-002"]
    pd.DataFrame({"StudyInstanceUID": test_ids}).to_csv(root / "test.csv", index=False)
    test_series = [{"StudyInstanceUID": uid, "SeriesInstanceUID": f"series-{uid}-Axial",
                    "Fluid_Sensitive": 1, "Fat_Suppression": 1, "Anatomical_Plane": "Axial"} for uid in test_ids]
    pd.DataFrame(test_series, columns=SERIES_COLUMNS).to_csv(root / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": test_ids, **{target: 0.5 for target in TARGET_COLUMNS}}).to_csv(root / "sample_submission.csv", index=False)
    return root, ids, test_ids


def test_exact_targets_metadata_grouping_and_missing_labels(tmp_path):
    root, ids, _ = make_metadata(tmp_path)
    metadata = load_rsna_metadata(root)
    assert tuple(metadata.train.columns) == ("StudyInstanceUID", "Report", *TARGET_COLUMNS)
    assert metadata.train["StudyInstanceUID"].dtype.name == "string"
    assert len(metadata.study_series_map()[ids[0]]) == 3
    assert metadata.target_vector(ids[0]).shape == (12,)
    assert metadata.complete_label_train.shape[0] == 6
    report = metadata_report(metadata)
    assert report["training_studies"] == 7
    assert report["complete_label_studies"] == 6
    assert report["missing_label_counts"]["ACL"] == 1
    assert report["anatomical_plane_counts"] == {"Axial": 7, "Coronal": 7, "Sagittal": 7}
    assert report["dicom_directories"]["train"] is False


def test_study_level_split_is_disjoint_and_only_complete_labels(tmp_path):
    root, _, _ = make_metadata(tmp_path)
    metadata = load_rsna_metadata(root)
    train, valid = split_labeled_studies(metadata, 0.33, 7)
    assert set(train.StudyInstanceUID).isdisjoint(valid.StudyInstanceUID)
    assert len(train) + len(valid) == 6
    assert not train[list(TARGET_COLUMNS)].isna().any().any()


def test_model_outputs_twelve_logits_and_auc_handles_single_class():
    model = RSNAKneeCNN(num_planes=3, dropout=0).eval()
    images = torch.rand(2, 3, 2, 1, 32, 32)
    mask = torch.ones(2, 3, 2, dtype=torch.bool)
    assert model(images, mask).shape == (2, 12)
    labels = np.zeros((3, 12)); scores = np.full((3, 12), 0.5)
    result = calculate_auc(labels, scores)
    assert result["macro_roc_auc"] is None
    assert all(value is None for value in result["per_label"].values())
    labels[0, 0] = 1
    result = calculate_auc(labels, scores)
    assert result["per_label"][TARGET_COLUMNS[0]] == 0.5
    assert result["macro_roc_auc"] == 0.5


def test_submission_template_schema_and_metadata_only_no_dicom(tmp_path):
    root, _, test_ids = make_metadata(tmp_path)
    metadata = load_rsna_metadata(root)
    assert tuple(metadata.sample_submission.columns) == SUBMISSION_COLUMNS
    assert list(metadata.sample_submission.StudyInstanceUID.astype(str)) == test_ids
    assert not (root / "test_series" / test_ids[0]).exists()


@pytest.mark.filterwarnings("ignore:Invalid value for VR UI")  # toy IDs such as 'test-001' are not DICOM UID syntax
def test_submission_generation_with_tiny_synthetic_dicom(tmp_path):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
    root, _, test_ids = make_metadata(tmp_path)
    for uid in test_ids:
        series_uid = f"series-{uid}-Axial"
        directory = root / "test_series" / uid / series_uid
        directory.mkdir(parents=True)
        meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(directory / "1.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.StudyInstanceUID = uid; ds.SeriesInstanceUID = series_uid
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; ds.ImagePositionPatient = [0.0, 0.0, 0.0]
        ds.Rows = 32; ds.Columns = 32; ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
        ds.InstanceNumber = 1; ds.PixelData = np.zeros((32, 32), dtype=np.uint16).tobytes(); ds.save_as(directory / "1.dcm")
    config = {"dropout": 0.0, "planes": ["Axial"], "image_size": 32, "slices_per_series": 1}
    model = RSNAKneeCNN(1, 0.0)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": config, "target_columns": list(TARGET_COLUMNS)}, checkpoint)
    output = tmp_path / "submission.csv"
    result = write_submission(root, checkpoint, output, batch_size=2)
    assert output.is_file()
    assert tuple(result.columns) == SUBMISSION_COLUMNS
    assert result.shape == (2, 13)
    assert list(result.StudyInstanceUID.astype(str)) == test_ids
    assert np.logical_and(result[list(TARGET_COLUMNS)].to_numpy() >= 0,
                          result[list(TARGET_COLUMNS)].to_numpy() <= 1).all()
