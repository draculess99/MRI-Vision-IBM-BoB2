"""Tests for 5-fold ensemble RSNA inference.

All checkpoints are synthetic, randomly-initialized RSNAKneeCNN saved to temporary
files, built purely to exercise the ensemble loading/validation/averaging code paths.
No real trained RSNA checkpoints are required or used.
"""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

from mri_core.rsna_knee_dataset import SERIES_COLUMNS, TARGET_COLUMNS, load_rsna_metadata
from mri_core.rsna_knee_model import RSNAKneeCNN
from mri_core.rsna_knee_ensemble import run_ensemble_inference
from mri_core.rsna_knee_inference import (
    run_inference,
    CheckpointNotFoundError,
    MalformedCheckpointError,
    TargetMismatchError,
    ArchitectureMismatchError,
    StudyUnavailableError,
)

from .dicom_factory import AXIAL, CORONAL, SAGITTAL, write_series

pytestmark = pytest.mark.rsna

STUDY = "1.2.826.1"
AXIAL_SERIES, CORONAL_SERIES, SAGITTAL_SERIES = (
    "1.2.826.2.1",
    "1.2.826.2.2",
    "1.2.826.2.3",
)
CONFIG = {
    "image_size": 16,
    "slices_per_series": 2,
    "planes": ["Axial", "Coronal", "Sagittal"],
    "dropout": 0.0,
}


@pytest.fixture
def rsna(tmp_path):
    """Metadata CSVs plus a 3-plane synthetic DICOM study."""
    meta, raw = tmp_path / "meta", tmp_path / "raw"
    meta.mkdir()
    pd.DataFrame(
        [
            {
                "StudyInstanceUID": STUDY,
                "Report": "r",
                **{t: i % 2 for i, t in enumerate(TARGET_COLUMNS)},
            }
        ]
    ).to_csv(meta / "train.csv", index=False)
    pd.DataFrame(
        [
            (STUDY, AXIAL_SERIES, 1, 0, "Axial"),
            (STUDY, CORONAL_SERIES, 0, 1, "Coronal"),
            (STUDY, SAGITTAL_SERIES, 0, 1, "Sagittal"),
        ],
        columns=SERIES_COLUMNS,
    ).to_csv(meta / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": []}).to_csv(meta / "test.csv", index=False)
    pd.DataFrame(columns=SERIES_COLUMNS).to_csv(
        meta / "test_series.csv", index=False
    )
    pd.DataFrame(
        {"StudyInstanceUID": [], **{t: [] for t in TARGET_COLUMNS}}
    ).to_csv(meta / "sample_submission.csv", index=False)
    write_series(
        raw / "train_series" / STUDY / AXIAL_SERIES,
        count=2,
        orientation=AXIAL,
        series_uid=AXIAL_SERIES,
        study_uid=STUDY,
    )
    write_series(
        raw / "train_series" / STUDY / CORONAL_SERIES,
        count=2,
        orientation=CORONAL,
        series_uid=CORONAL_SERIES,
        study_uid=STUDY,
    )
    write_series(
        raw / "train_series" / STUDY / SAGITTAL_SERIES,
        count=2,
        orientation=SAGITTAL,
        series_uid=SAGITTAL_SERIES,
        study_uid=STUDY,
    )
    metadata = load_rsna_metadata(meta, dicom_root=raw)
    return SimpleNamespace(meta=meta, raw=raw, metadata=metadata)


def make_checkpoint(path, num_planes=3, dropout=0.0, seed=None, target_columns=None):
    """Create a synthetic checkpoint with optional seed for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
    model = RSNAKneeCNN(num_planes=num_planes, dropout=dropout)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "image_size": 16,
                "slices_per_series": 2,
                "planes": ["Axial", "Coronal", "Sagittal"],
                "dropout": dropout,
            },
            "target_columns": list(target_columns or TARGET_COLUMNS),
        },
        path,
    )
    return path


def make_ensemble_checkpoints(tmp_path, count=5, seed_base=0):
    """Create a set of numbered checkpoint files with deterministic seeds."""
    paths = []
    for i in range(count):
        ckpt_path = tmp_path / f"fold_{i + 1}.pt"
        make_checkpoint(ckpt_path, seed=seed_base + i)
        paths.append(ckpt_path)
    return paths


class TestEnsembleValidStructure:
    """Tests for ensemble output structure and format."""

    def test_ensemble_requires_exactly_five_folds(self, rsna, tmp_path):
        """Reject anything other than 5 checkpoints."""
        ckpts = make_ensemble_checkpoints(tmp_path, count=4)
        with pytest.raises(ValueError, match="exactly 5 fold checkpoint paths"):
            run_ensemble_inference(rsna.metadata, ckpts, STUDY)

        ckpts = make_ensemble_checkpoints(tmp_path, count=6)
        with pytest.raises(ValueError, match="exactly 5 fold checkpoint paths"):
            run_ensemble_inference(rsna.metadata, ckpts, STUDY)

    def test_ensemble_output_has_exactly_twelve_targets(self, rsna, tmp_path):
        """Ensemble result must have all 12 targets."""
        ckpts = make_ensemble_checkpoints(tmp_path, count=5)
        result = run_ensemble_inference(rsna.metadata, ckpts, STUDY)
        assert len(result) == 12

    def test_ensemble_target_order_matches_target_columns(self, rsna, tmp_path):
        """Output targets in same order as single-model inference."""
        ckpts = make_ensemble_checkpoints(tmp_path, count=5)
        result = run_ensemble_inference(rsna.metadata, ckpts, STUDY)
        assert list(result.keys()) == list(TARGET_COLUMNS)

    def test_ensemble_every_value_is_python_float_in_unit_range(self, rsna, tmp_path):
        """All probabilities must be float in [0.0, 1.0]."""
        ckpts = make_ensemble_checkpoints(tmp_path, count=5)
        result = run_ensemble_inference(rsna.metadata, ckpts, STUDY)
        for target, probability in result.items():
            assert isinstance(probability, float), f"{target}: {type(probability)}"
            assert 0.0 <= probability <= 1.0, f"{target}: {probability}"


class TestEnsembleAveraging:
    """Tests for ensemble averaging logic."""

    def test_ensemble_averages_five_model_predictions(self, rsna, tmp_path):
        """Verify ensemble computes arithmetic mean across folds."""
        # Create 5 checkpoints with known seeds for reproducibility
        ckpts = make_ensemble_checkpoints(tmp_path, count=5, seed_base=42)

        # Run ensemble
        result = run_ensemble_inference(rsna.metadata, ckpts, STUDY)

        # Run each model individually
        individual_results = []
        for ckpt in ckpts:
            individual = run_inference(rsna.metadata, ckpt, STUDY)
            individual_results.append(individual)

        # Verify that ensemble mean matches arithmetic mean of individuals
        for target in TARGET_COLUMNS:
            individual_probs = [r[target] for r in individual_results]
            expected_mean = np.mean(individual_probs)
            actual_ensemble = result[target]
            assert (
                actual_ensemble == pytest.approx(expected_mean, abs=1e-6)
            ), f"{target}: ensemble {actual_ensemble} != mean {expected_mean}"

    def test_ensemble_repeated_runs_are_deterministic(self, rsna, tmp_path):
        """Ensemble output must be identical on repeated runs."""
        ckpts = make_ensemble_checkpoints(tmp_path, count=5)
        first = run_ensemble_inference(rsna.metadata, ckpts, STUDY)
        second = run_ensemble_inference(rsna.metadata, ckpts, STUDY)
        assert first == second


class TestEnsembleCheckpointValidation:
    """Tests for checkpoint loading and validation."""

    def test_missing_fold_raises_checkpoint_not_found(self, rsna, tmp_path):
        """Missing fold file → CheckpointNotFoundError with fold info."""
        ckpts = make_ensemble_checkpoints(tmp_path, count=5)
        ckpts[2] = tmp_path / "does_not_exist_fold_3.pt"  # Remove fold 3
        with pytest.raises(
            CheckpointNotFoundError, match="Fold 3.*No checkpoint file"
        ):
            run_ensemble_inference(rsna.metadata, ckpts, STUDY)

    def test_malformed_checkpoint_fold_raises_error(self, rsna, tmp_path):
        """Corrupted checkpoint → MalformedCheckpointError with fold info."""
        ckpts = make_ensemble_checkpoints(tmp_path, count=5)
        # Make fold 2 not a dict
        bad_ckpt = ckpts[1]
        torch.save([1, 2, 3], bad_ckpt)
        with pytest.raises(MalformedCheckpointError, match="Fold 2.*did not contain a dict"):
            run_ensemble_inference(rsna.metadata, ckpts, STUDY)

    def test_fold_missing_required_fields_raises_error(self, rsna, tmp_path):
        """Checkpoint missing required keys → MalformedCheckpointError."""
        ckpts = make_ensemble_checkpoints(tmp_path, count=5)
        # Make fold 4 missing 'config'
        bad_ckpt = ckpts[3]
        model = RSNAKneeCNN(3, 0.0)
        torch.save({"model": model.state_dict(), "target_columns": list(TARGET_COLUMNS)}, bad_ckpt)
        with pytest.raises(MalformedCheckpointError, match="Fold 4.*missing required fields"):
            run_ensemble_inference(rsna.metadata, ckpts, STUDY)

    def test_fold_target_mismatch_raises_error(self, rsna, tmp_path):
        """Fold with wrong target_columns → TargetMismatchError."""
        ckpts = make_ensemble_checkpoints(tmp_path, count=5)
        bad_ckpt = ckpts[0]
        make_checkpoint(bad_ckpt, target_columns=["abnormal", "acl", "meniscus"])
        with pytest.raises(TargetMismatchError, match="Fold 1.*target_columns.*do not match"):
            run_ensemble_inference(rsna.metadata, ckpts, STUDY)

    def test_fold_architecture_mismatch_raises_error(self, rsna, tmp_path):
        """Fold with incompatible weights → ArchitectureMismatchError."""
        ckpts = make_ensemble_checkpoints(tmp_path, count=5)
        bad_ckpt = ckpts[2]
        # Mismatched: config says 3 planes but weights from 1-plane model
        mismatched_model = RSNAKneeCNN(num_planes=1, dropout=0.0)
        torch.save(
            {
                "model": mismatched_model.state_dict(),
                "config": {
                    "image_size": 16,
                    "slices_per_series": 2,
                    "planes": ["Axial", "Coronal", "Sagittal"],
                    "dropout": 0.0,
                },
                "target_columns": list(TARGET_COLUMNS),
            },
            bad_ckpt,
        )
        with pytest.raises(ArchitectureMismatchError, match="Fold 3.*weights are not compatible"):
            run_ensemble_inference(rsna.metadata, ckpts, STUDY)


class TestEnsembleConfigFormats:
    """Tests for different checkpoint config formats (existing vs Kaggle)."""

    def test_ensemble_supports_existing_planes_list_config(self, rsna, tmp_path):
        """Existing format with 'planes' list should work."""
        ckpts = []
        for i in range(5):
            ckpt_path = tmp_path / f"fold_{i + 1}.pt"
            model = RSNAKneeCNN(num_planes=3, dropout=0.0)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "image_size": 16,
                        "slices_per_series": 2,
                        "planes": ["Axial", "Coronal", "Sagittal"],
                        "dropout": 0.0,
                    },
                    "target_columns": list(TARGET_COLUMNS),
                },
                ckpt_path,
            )
            ckpts.append(ckpt_path)
        result = run_ensemble_inference(rsna.metadata, ckpts, STUDY)
        assert len(result) == 12
        assert all(0.0 <= p <= 1.0 for p in result.values())

    def test_ensemble_supports_kaggle_num_planes_config(self, rsna, tmp_path):
        """Kaggle format with 'num_planes' and 'slices_per_plane' (no slices_per_series)."""
        ckpts = []
        for i in range(5):
            ckpt_path = tmp_path / f"fold_{i + 1}.pt"
            model = RSNAKneeCNN(num_planes=3, dropout=0.2)
            # Real Kaggle schema: uses slices_per_plane, NOT slices_per_series
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "num_planes": 3,
                        "slices_per_plane": 2,  # Dataset expects slices_per_series; ensemble normalizes
                        "image_size": 16,
                        "dropout": 0.2,
                        "num_targets": 12,
                        "learning_rate": 0.0003,
                        "train_size": 46,
                        "val_size": 12,
                        "cv_fold": i + 1,
                    },
                    "target_columns": list(TARGET_COLUMNS),
                },
                ckpt_path,
            )
            ckpts.append(ckpt_path)
        # Should succeed because ensemble normalizes slices_per_plane -> slices_per_series
        result = run_ensemble_inference(rsna.metadata, ckpts, STUDY)
        assert len(result) == 12
        assert all(0.0 <= p <= 1.0 for p in result.values())

    def test_ensemble_normalizes_kaggle_slices_per_plane_to_slices_per_series(self, rsna, tmp_path):
        """Config normalization: slices_per_plane -> slices_per_series."""
        ckpts = []
        for i in range(5):
            ckpt_path = tmp_path / f"fold_{i + 1}.pt"
            model = RSNAKneeCNN(num_planes=3, dropout=0.0)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "num_planes": 3,
                        "slices_per_plane": 999,  # Unique value for detection
                        "image_size": 16,
                        "dropout": 0.0,
                    },
                    "target_columns": list(TARGET_COLUMNS),
                },
                ckpt_path,
            )
            ckpts.append(ckpt_path)

        # Should work: ensemble normalizes config so RSNAKneeDicomDataset gets slices_per_series
        result = run_ensemble_inference(rsna.metadata, ckpts, STUDY)
        assert len(result) == 12
        # The normalization allows the inference to succeed when slices_per_series is missing


class TestEnsembleStudyValidation:
    """Tests for study-level validation."""

    def test_unknown_study_uid_raises_study_unavailable(self, rsna, tmp_path):
        """Invalid study UID → StudyUnavailableError."""
        ckpts = make_ensemble_checkpoints(tmp_path, count=5)
        with pytest.raises(StudyUnavailableError):
            run_ensemble_inference(rsna.metadata, ckpts, "1.2.826.999.not.real")


class TestEnsembleBackwardCompatibility:
    """Tests ensuring existing single-model inference is unaffected."""

    def test_single_checkpoint_inference_unchanged(self, rsna, tmp_path):
        """run_inference() must still work and remain unaffected."""
        single_ckpt = make_checkpoint(tmp_path / "single.pt")
        result = run_inference(rsna.metadata, single_ckpt, STUDY)
        assert len(result) == 12
        assert list(result.keys()) == list(TARGET_COLUMNS)
        assert all(0.0 <= p <= 1.0 for p in result.values())

    def test_ensemble_and_single_inference_coexist(self, rsna, tmp_path):
        """Both ensemble and single inference can be called on same metadata/study."""
        single_ckpt = make_checkpoint(tmp_path / "single.pt", seed=99)
        ensemble_ckpts = make_ensemble_checkpoints(tmp_path, count=5, seed_base=100)

        single_result = run_inference(rsna.metadata, single_ckpt, STUDY)
        ensemble_result = run_ensemble_inference(rsna.metadata, ensemble_ckpts, STUDY)

        assert len(single_result) == 12
        assert len(ensemble_result) == 12
        assert list(single_result.keys()) == list(ensemble_result.keys())
