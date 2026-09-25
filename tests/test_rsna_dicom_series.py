import dataclasses
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")

from mri_core import DicomSeriesError, DicomSeriesWarning, MRIVolume
from mri_core.rsna_knee_dataset import (SERIES_COLUMNS, SUBMISSION_COLUMNS, TARGET_COLUMNS, load_rsna_metadata,
                                        load_series_volume, load_study_volumes, metadata_report)

from .dicom_factory import AXIAL, SAGITTAL, STUDY_UID, corrupt_element, write_series

STUDY = STUDY_UID
AXIAL_SERIES, SAGITTAL_SERIES = "1.2.826.2.1", "1.2.826.2.2"
TEST_STUDY, TEST_SERIES = "1.2.826.9", "1.2.826.9.1"
CLI = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_rsna_knee.py"


@pytest.fixture
def rsna(tmp_path):
    """Metadata CSVs in meta/, DICOMs in a separate raw/ root, mirroring data/rsna-knee and data/rsna-knee/raw."""
    meta, raw = tmp_path / "meta", tmp_path / "raw"
    meta.mkdir()
    pd.DataFrame([{"StudyInstanceUID": STUDY, "Report": "r", **{t: i % 2 for i, t in enumerate(TARGET_COLUMNS)}}]) \
        .to_csv(meta / "train.csv", index=False)
    pd.DataFrame([(STUDY, AXIAL_SERIES, 1, 0, "Axial"), (STUDY, SAGITTAL_SERIES, 0, 1, "Sagittal")],
                 columns=SERIES_COLUMNS).to_csv(meta / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": [TEST_STUDY]}).to_csv(meta / "test.csv", index=False)
    pd.DataFrame([(TEST_STUDY, TEST_SERIES, 1, 1, "Axial")], columns=SERIES_COLUMNS).to_csv(meta / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": [TEST_STUDY], **{t: 0.5 for t in TARGET_COLUMNS}}) \
        .to_csv(meta / "sample_submission.csv", index=False)
    write_series(raw / "train_series" / STUDY / AXIAL_SERIES, count=4, orientation=AXIAL, series_uid=AXIAL_SERIES)
    write_series(raw / "train_series" / STUDY / SAGITTAL_SERIES, count=3, orientation=SAGITTAL, series_uid=SAGITTAL_SERIES,
                 spacing=4.0, rows=8, columns=10)
    write_series(raw / "test_series" / TEST_STUDY / TEST_SERIES, count=3, orientation=AXIAL, series_uid=TEST_SERIES,
                 study_uid=TEST_STUDY)
    return SimpleNamespace(meta=meta, raw=raw)


# ----------------------------------------------------------------------------- DICOM root configuration

def test_dicom_root_defaults_to_data_dir(rsna):
    metadata = load_rsna_metadata(rsna.meta)
    assert metadata.dicom_root is None
    assert metadata.dicom_split_dir("train") == rsna.meta / "train_series"
    assert metadata.dicom_series_dir(STUDY, AXIAL_SERIES) == rsna.meta / "train_series" / STUDY / AXIAL_SERIES
    assert metadata_report(metadata)["dicom_directories"]["train"] is False
    with pytest.raises(ValueError, match="train or test"):
        metadata.dicom_split_dir("validation")


@pytest.mark.parametrize("as_type", [Path, str])
def test_configured_dicom_root_needs_no_metadata_replacement(rsna, as_type):
    metadata = load_rsna_metadata(rsna.meta, dicom_root=as_type(rsna.raw))
    assert metadata.data_dir == rsna.meta and metadata.dicom_root == rsna.raw
    assert metadata.dicom_split_dir("train") == rsna.raw / "train_series"
    assert metadata.dicom_split_dir("test") == rsna.raw / "test_series"
    assert metadata.dicom_series_dir(STUDY, SAGITTAL_SERIES) == rsna.raw / "train_series" / STUDY / SAGITTAL_SERIES
    directories = metadata_report(metadata)["dicom_directories"]
    assert directories["train"] is True and directories["test"] is True
    assert directories["train_path"] == str(rsna.raw / "train_series")


def test_cli_inspect_accepts_dicom_root(rsna, capsys):
    spec = importlib.util.spec_from_file_location("kaggle_rsna_knee", CLI)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(["inspect", "--data-dir", str(rsna.meta)]) == 0
    assert json.loads(capsys.readouterr().out)["dicom_directories"]["train"] is False
    assert cli.main(["inspect", "--data-dir", str(rsna.meta), "--dicom-root", str(rsna.raw)]) == 0
    directories = json.loads(capsys.readouterr().out)["dicom_directories"]
    assert directories["train"] is True and directories["train_path"] == str(rsna.raw / "train_series")


def test_existing_dicom_dataset_honours_dicom_root(rsna):
    pytest.importorskip("torch")
    from mri_core.rsna_knee_train import RSNAKneeDicomDataset
    config = {"image_size": 16, "slices_per_series": 2, "planes": ["Axial", "Sagittal"]}
    default = load_rsna_metadata(rsna.meta)
    with pytest.raises(FileNotFoundError, match="train_series"):
        RSNAKneeDicomDataset(default, default.train, config)
    configured = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    item = RSNAKneeDicomDataset(configured, configured.train, config)[0]
    assert tuple(item["images"].shape) == (2, 2, 1, 16, 16)
    assert bool(item["mask"].all())


def test_submission_uses_dicom_root(rsna, tmp_path):
    torch = pytest.importorskip("torch")
    from mri_core.rsna_knee_model import RSNAKneeCNN
    from mri_core.rsna_knee_submit import write_submission
    config = {"dropout": 0.0, "planes": ["Axial"], "image_size": 16, "slices_per_series": 2}
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": RSNAKneeCNN(1, 0.0).state_dict(), "config": config, "target_columns": list(TARGET_COLUMNS)}, checkpoint)
    with pytest.raises(FileNotFoundError, match="test_series"):
        write_submission(rsna.meta, checkpoint, tmp_path / "out.csv")
    result = write_submission(rsna.meta, checkpoint, tmp_path / "out.csv", dicom_root=rsna.raw)
    assert tuple(result.columns) == SUBMISSION_COLUMNS and list(result.StudyInstanceUID.astype(str)) == [TEST_STUDY]


# ----------------------------------------------------------------------------- series and study adapter

def test_load_series_volume_combines_dicom_and_csv_metadata(rsna):
    metadata = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    volume = load_series_volume(metadata, STUDY, SAGITTAL_SERIES)
    assert isinstance(volume, MRIVolume) and volume.shape == (3, 8, 10)
    assert volume.metadata["Series Instance UID"] == SAGITTAL_SERIES
    assert volume.metadata["Anatomical Plane"] == "Sagittal"
    assert volume.metadata["Acquisition Plane (DICOM orientation)"] == "Sagittal"
    assert volume.metadata["Fluid Sensitive"] is False and volume.metadata["Fat Suppression"] is True
    assert [float(volume.get_slice(i)[0, 0]) for i in range(3)] == [10.0, 20.0, 30.0]


def test_load_series_volume_supports_test_split(rsna):
    metadata = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    assert load_series_volume(metadata, TEST_STUDY, TEST_SERIES, split="test").shape == (3, 8, 8)


def test_load_study_volumes_keeps_metadata_order(rsna):
    metadata = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    volumes = load_study_volumes(metadata, STUDY)
    assert list(volumes) == [AXIAL_SERIES, SAGITTAL_SERIES]
    assert [v.shape for v in volumes.values()] == [(4, 8, 8), (3, 8, 10)]
    with pytest.raises(KeyError, match="No train series"):
        load_study_volumes(metadata, "1.2.826.404")


def test_missing_series_directory_explains_dicom_root(rsna):
    metadata = load_rsna_metadata(rsna.meta)
    with pytest.raises(FileNotFoundError) as excinfo:
        load_series_volume(metadata, STUDY, AXIAL_SERIES)
    assert "--dicom-root" in str(excinfo.value) and str(rsna.meta / "train_series") in str(excinfo.value)


def test_unknown_series_is_rejected(rsna):
    metadata = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    with pytest.raises(KeyError, match="found 0"):
        load_series_volume(metadata, STUDY, "1.2.826.404")


def test_plane_disagreement_between_csv_and_dicom_warns(rsna):
    metadata = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    frame = metadata.train_series.copy()
    frame.loc[frame.SeriesInstanceUID == AXIAL_SERIES, "Anatomical_Plane"] = "Coronal"
    with pytest.warns(DicomSeriesWarning, match="Coronal.*Axial"):
        volume = load_series_volume(dataclasses.replace(metadata, train_series=frame), STUDY, AXIAL_SERIES)
    assert volume.metadata["Anatomical Plane"] == "Coronal"


def test_series_directory_holding_another_series_is_rejected(rsna):
    metadata = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    alias = "1.2.826.2.9"
    shutil.copytree(rsna.raw / "train_series" / STUDY / AXIAL_SERIES, rsna.raw / "train_series" / STUDY / alias)
    frame = metadata.train_series.copy()
    frame.loc[frame.SeriesInstanceUID == AXIAL_SERIES, "SeriesInstanceUID"] = alias
    with pytest.raises(DicomSeriesError, match="does not match expected"):
        load_series_volume(dataclasses.replace(metadata, train_series=frame), STUDY, alias)


# ----------------------------------------------------------------------------- MRIVolume-backed dataset

DATASET_CONFIG = {"image_size": 16, "slices_per_series": 3, "planes": ["Axial", "Sagittal"]}


@pytest.fixture
def dataset(rsna):
    pytest.importorskip("torch")
    from mri_core.rsna_knee_train import RSNAKneeDicomDataset
    metadata = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    return RSNAKneeDicomDataset(metadata, metadata.train, DATASET_CONFIG)


def _rewrite(rsna, series, **kwargs):
    orientation, count, extra = (AXIAL, 4, {}) if series == AXIAL_SERIES else (SAGITTAL, 3, {"spacing": 4.0, "columns": 10})
    kwargs = {"orientation": orientation, "count": count, "series_uid": series, **extra, **kwargs}
    return write_series(rsna.raw / "train_series" / STUDY / series, **kwargs)


def _rescaled(index, ds):
    ds.RescaleSlope, ds.RescaleIntercept = 2.5, 0.0


@pytest.mark.parametrize("config", [DATASET_CONFIG, {"image_size": 12, "slices_per_series": 9, "planes": ["Sagittal", "Axial"]}],
                         ids=["subsampled", "more-slices-than-series"])
def test_dataset_matches_previous_reading_algorithm(rsna, config):
    """RescaleSlope is applied now but was ignored before; per-slice percentile normalization makes that immaterial."""
    pytest.importorskip("torch")
    from mri_core.rsna_knee_train import RSNAKneeDicomDataset
    from .legacy_rsna_reader import legacy_item
    for series in (AXIAL_SERIES, SAGITTAL_SERIES):  # 32 px slices, so resizing to <=16 px really downsamples
        _rewrite(rsna, series, textured=True, mutate=_rescaled, rows=32, columns=32)
    metadata = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    new = RSNAKneeDicomDataset(metadata, metadata.train, config)[0]
    old = legacy_item(metadata, metadata.train.iloc[0], config)
    assert float(new["images"].std()) > 0.05  # the comparison is not vacuous
    assert float((new["images"] - old["images"]).abs().max()) <= 1e-6
    assert new["images"].shape == old["images"].shape and bool((new["mask"] == old["mask"]).all())
    assert bool((new["labels"] == old["labels"]).all()) and new["study_uid"] == old["study_uid"] == STUDY


def test_dataset_item_shapes_and_mask(dataset):
    item = dataset[0]
    assert tuple(item["images"].shape) == (2, 3, 1, 16, 16)
    assert tuple(item["mask"].shape) == (2, 3) and bool(item["mask"].all())
    assert tuple(item["labels"].shape) == (12,)


def test_dataset_fails_immediately_on_unreadable_file_naming_study_series_and_file(rsna, dataset):
    (rsna.raw / "train_series" / STUDY / SAGITTAL_SERIES / "corrupt.dcm").write_bytes(b"not dicom")
    with pytest.raises(DicomSeriesError) as excinfo:
        dataset[0]
    message = str(excinfo.value)
    assert f"StudyInstanceUID={STUDY}" in message and f"SeriesInstanceUID={SAGITTAL_SERIES}" in message
    assert "corrupt.dcm" in message


def test_dataset_fails_immediately_on_duplicate_instance_numbers(rsna, dataset):
    _rewrite(rsna, AXIAL_SERIES, mutate=lambda i, ds: setattr(ds, "InstanceNumber", 1) if i == 2 else None)
    with pytest.raises(DicomSeriesError, match="Duplicate InstanceNumber") as excinfo:
        dataset[0]
    assert f"StudyInstanceUID={STUDY}" in str(excinfo.value) and f"SeriesInstanceUID={AXIAL_SERIES}" in str(excinfo.value)


def test_dataset_fails_immediately_when_dicom_series_id_disagrees_with_metadata(rsna, dataset):
    _rewrite(rsna, AXIAL_SERIES, series_uid="1.2.826.77")
    with pytest.raises(DicomSeriesError, match="does not match expected") as excinfo:
        dataset[0]
    assert f"SeriesInstanceUID={AXIAL_SERIES}" in str(excinfo.value)


@pytest.mark.parametrize("keyword,bad,slice_index", [("PixelSpacing", "a\\b", 1), ("RescaleSlope", "abc", 2),
                                                     ("SliceThickness", "abc", 0)])
def test_dataset_malformed_numeric_header_names_file_study_series_and_reason(rsna, dataset, keyword, bad, slice_index):
    paths = _rewrite(rsna, SAGITTAL_SERIES, mutate=_rescaled)  # guarantees RescaleSlope/Intercept elements exist
    corrupt_element(paths[slice_index], keyword, bad)
    with pytest.raises(DicomSeriesError) as excinfo:
        dataset[0]
    message = str(excinfo.value)
    assert paths[slice_index].name in message
    assert f"StudyInstanceUID={STUDY}" in message and f"SeriesInstanceUID={SAGITTAL_SERIES}" in message
    assert keyword in message and "could not convert string to float" in message


def test_dataset_fails_immediately_when_series_directory_is_missing(rsna, dataset):
    shutil.rmtree(rsna.raw / "train_series" / STUDY / SAGITTAL_SERIES)
    with pytest.raises(FileNotFoundError) as excinfo:
        dataset[0]
    assert f"StudyInstanceUID={STUDY}" in str(excinfo.value) and f"SeriesInstanceUID={SAGITTAL_SERIES}" in str(excinfo.value)


def test_dataset_fails_immediately_when_metadata_lists_no_series_for_a_plane(rsna):
    pytest.importorskip("torch")
    from mri_core.rsna_knee_train import RSNAKneeDicomDataset
    metadata = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    with pytest.raises(ValueError, match="no Coronal series") as excinfo:
        RSNAKneeDicomDataset(metadata, metadata.train, {**DATASET_CONFIG, "planes": ["Axial", "Coronal"]})[0]
    assert f"StudyInstanceUID={STUDY}" in str(excinfo.value)


def test_dataset_batch_feeds_the_twelve_output_model(dataset):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader
    from mri_core.rsna_knee_model import RSNAKneeCNN
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    with torch.inference_mode():
        logits = RSNAKneeCNN(num_planes=2, dropout=0.0).eval()(batch["images"], batch["mask"])
    assert tuple(logits.shape) == (1, 12) and bool(torch.isfinite(logits).all())
