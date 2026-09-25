"""Tests for single-study RSNA inference plumbing.

No real trained RSNA checkpoint is used or required. Every checkpoint here is a
synthetic, randomly-initialized RSNAKneeCNN saved to a temporary file, built purely
to exercise the loading/validation/inference code paths.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

from mri_core.rsna_knee_dataset import SERIES_COLUMNS, TARGET_COLUMNS, load_rsna_metadata
from mri_core.rsna_knee_model import RSNAKneeCNN
from mri_core.rsna_knee_inference import (
    run_inference,
    CheckpointNotFoundError,
    MalformedCheckpointError,
    TargetMismatchError,
    ArchitectureMismatchError,
    StudyUnavailableError,
)

from .dicom_factory import AXIAL, CORONAL, SAGITTAL, write_series

STUDY = "1.2.826.1"
AXIAL_SERIES, CORONAL_SERIES, SAGITTAL_SERIES = "1.2.826.2.1", "1.2.826.2.2", "1.2.826.2.3"
CONFIG = {"image_size": 16, "slices_per_series": 2, "planes": ["Axial", "Coronal", "Sagittal"], "dropout": 0.0}


@pytest.fixture
def rsna(tmp_path):
    """Metadata CSVs plus a 3-plane synthetic DICOM study, mirroring data/rsna-knee layout."""
    meta, raw = tmp_path / "meta", tmp_path / "raw"
    meta.mkdir()
    pd.DataFrame([{"StudyInstanceUID": STUDY, "Report": "r", **{t: i % 2 for i, t in enumerate(TARGET_COLUMNS)}}]) \
        .to_csv(meta / "train.csv", index=False)
    pd.DataFrame(
        [(STUDY, AXIAL_SERIES, 1, 0, "Axial"), (STUDY, CORONAL_SERIES, 0, 1, "Coronal"),
         (STUDY, SAGITTAL_SERIES, 0, 1, "Sagittal")],
        columns=SERIES_COLUMNS,
    ).to_csv(meta / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": []}).to_csv(meta / "test.csv", index=False)
    pd.DataFrame(columns=SERIES_COLUMNS).to_csv(meta / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": [], **{t: [] for t in TARGET_COLUMNS}}).to_csv(meta / "sample_submission.csv", index=False)
    write_series(raw / "train_series" / STUDY / AXIAL_SERIES, count=2, orientation=AXIAL, series_uid=AXIAL_SERIES, study_uid=STUDY)
    write_series(raw / "train_series" / STUDY / CORONAL_SERIES, count=2, orientation=CORONAL, series_uid=CORONAL_SERIES, study_uid=STUDY)
    write_series(raw / "train_series" / STUDY / SAGITTAL_SERIES, count=2, orientation=SAGITTAL, series_uid=SAGITTAL_SERIES, study_uid=STUDY)
    metadata = load_rsna_metadata(meta, dicom_root=raw)
    return SimpleNamespace(meta=meta, raw=raw, metadata=metadata)


def make_checkpoint(path, num_planes=3, dropout=0.0, planes=("Axial", "Coronal", "Sagittal"),
                    target_columns=None, extra_config=None):
    config = {"image_size": 16, "slices_per_series": 2, "planes": list(planes), "dropout": dropout}
    if extra_config:
        config.update(extra_config)
    model = RSNAKneeCNN(num_planes=num_planes, dropout=dropout)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "target_columns": list(target_columns) if target_columns is not None else list(TARGET_COLUMNS),
        },
        path,
    )
    return path


class TestValidInference:
    def test_valid_checkpoint_loads_and_runs(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
        result = run_inference(rsna.metadata, checkpoint, STUDY)
        assert isinstance(result, dict)

    def test_output_has_exactly_twelve_targets(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
        result = run_inference(rsna.metadata, checkpoint, STUDY)
        assert len(result) == 12

    def test_target_order_matches_target_columns(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
        result = run_inference(rsna.metadata, checkpoint, STUDY)
        assert list(result.keys()) == list(TARGET_COLUMNS)

    def test_every_value_is_python_float_in_unit_range(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
        result = run_inference(rsna.metadata, checkpoint, STUDY)
        for target, probability in result.items():
            assert isinstance(probability, float), f"{target}: {type(probability)}"
            assert 0.0 <= probability <= 1.0, f"{target}: {probability}"

    def test_repeated_inference_is_deterministic(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
        first = run_inference(rsna.metadata, checkpoint, STUDY)
        second = run_inference(rsna.metadata, checkpoint, STUDY)
        assert first == second


class TestCheckpointErrors:
    def test_missing_checkpoint_raises_checkpoint_not_found(self, rsna, tmp_path):
        with pytest.raises(CheckpointNotFoundError):
            run_inference(rsna.metadata, tmp_path / "does_not_exist.pt", STUDY)

    def test_malformed_checkpoint_missing_required_fields(self, rsna, tmp_path):
        checkpoint = tmp_path / "malformed.pt"
        torch.save({"model": RSNAKneeCNN(3, 0.0).state_dict()}, checkpoint)  # no config/target_columns
        with pytest.raises(MalformedCheckpointError):
            run_inference(rsna.metadata, checkpoint, STUDY)

    def test_malformed_checkpoint_not_a_dict(self, rsna, tmp_path):
        checkpoint = tmp_path / "not_a_dict.pt"
        torch.save([1, 2, 3], checkpoint)
        with pytest.raises(MalformedCheckpointError):
            run_inference(rsna.metadata, checkpoint, STUDY)

    def test_target_column_mismatch_raises(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt", target_columns=["abnormal", "acl", "meniscus"])
        with pytest.raises(TargetMismatchError):
            run_inference(rsna.metadata, checkpoint, STUDY)

    def test_incompatible_head_dimensions_raise_architecture_mismatch(self, rsna, tmp_path):
        # Config claims 3 planes (head expects 384 inputs) but the saved weights were
        # actually produced by a 1-plane model (head has 128 inputs) -- an inconsistent
        # checkpoint that must surface as an architecture mismatch, not a crash.
        checkpoint = tmp_path / "mismatched.pt"
        mismatched_model = RSNAKneeCNN(num_planes=1, dropout=0.0)
        torch.save(
            {
                "model": mismatched_model.state_dict(),
                "config": {"image_size": 16, "slices_per_series": 2, "planes": ["Axial", "Coronal", "Sagittal"], "dropout": 0.0},
                "target_columns": list(TARGET_COLUMNS),
            },
            checkpoint,
        )
        with pytest.raises(ArchitectureMismatchError):
            run_inference(rsna.metadata, checkpoint, STUDY)


class TestStudyErrors:
    def test_unknown_study_uid_raises_study_unavailable(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
        with pytest.raises(StudyUnavailableError):
            run_inference(rsna.metadata, checkpoint, "1.2.826.999.not.a.real.study")

    def test_incomplete_study_missing_plane_raises_study_unavailable(self, rsna, tmp_path):
        # Config requests a plane ("Sagittal") that this study's series metadata omits.
        checkpoint = tmp_path / "ckpt.pt"
        config = {"image_size": 16, "slices_per_series": 2, "planes": ["Axial", "Coronal"], "dropout": 0.0}
        model = RSNAKneeCNN(num_planes=2, dropout=0.0)
        torch.save({"model": model.state_dict(), "config": config, "target_columns": list(TARGET_COLUMNS)}, checkpoint)
        # Remove the axial series row so the dataset can't find a required plane.
        metadata = rsna.metadata
        metadata.train_series.drop(
            metadata.train_series[metadata.train_series["SeriesInstanceUID"] == AXIAL_SERIES].index,
            inplace=True,
        )
        with pytest.raises(StudyUnavailableError):
            run_inference(metadata, checkpoint, STUDY)
