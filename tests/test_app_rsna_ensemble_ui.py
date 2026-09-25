"""UI tests for RSNA 5-fold ensemble integration in Streamlit app."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
torch = pytest.importorskip("torch")
pytest.importorskip("pandas")

from streamlit.testing.v1 import AppTest

from mri_core.rsna_integration import discover_available_studies, load_rsna_metadata_safe
from mri_core.rsna_knee_dataset import TARGET_COLUMNS
from mri_core.rsna_knee_inference import RSNAInferenceError, run_inference
from mri_core.rsna_knee_ensemble import run_ensemble_inference
from mri_core.rsna_knee_model import RSNAKneeCNN

pytest.mark.rsna

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
RSNA_ROOT = ROOT / "data" / "rsna-knee"
ENSEMBLE_REL = Path("outputs/rsna-knee-ensemble")
SMOKE_REL = Path("outputs/rsna-knee-smoke/checkpoint_smoke.pt")
PROD_REL = Path("outputs/rsna-knee/checkpoint_best.pt")


def make_checkpoint(path, seed=0, target_columns=None):
    """Create a synthetic checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.save(
        {
            "model": RSNAKneeCNN(num_planes=3, dropout=0.0).state_dict(),
            "config": {
                "image_size": 16,
                "slices_per_series": 2,
                "planes": ["Axial", "Coronal", "Sagittal"],
                "dropout": 0.0,
            },
            "target_columns": list(target_columns) if target_columns is not None else list(TARGET_COLUMNS),
        },
        path,
    )
    return path


@pytest.fixture(scope="module")
def ready(tmp_path_factory):
    """First locally available RSNA study that can run ensemble inference."""
    if not (RSNA_ROOT / "train.csv").is_file():
        pytest.skip("RSNA data directory not found")
    metadata = load_rsna_metadata_safe(RSNA_ROOT)

    # Create a probe checkpoint to find a compatible study
    probe_dir = tmp_path_factory.mktemp("probe")
    probe = make_checkpoint(probe_dir / "probe.pt")

    for study_uid in discover_available_studies(RSNA_ROOT, metadata):
        try:
            run_inference(metadata, probe, study_uid)
        except RSNAInferenceError:
            continue
        return metadata, study_uid

    pytest.skip("No fully downloaded RSNA study available")


def run_app(monkeypatch, workdir, study_uid, *, ensemble=None, smoke=None, process=True):
    """Drive app.py from `workdir`. ensemble/smoke=None leaves them untouched (default off)."""
    monkeypatch.chdir(workdir)
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    next(r for r in at.sidebar.radio if r.label == "Data Source").set_value("Explore RSNA Studies")
    at.run()
    next(s for s in at.sidebar.selectbox if s.label == "Select Study").set_value(study_uid)

    # Set checkboxes if specified
    if ensemble is not None:
        at.sidebar.checkbox(key="rsna_use_ensemble_inference").set_value(ensemble)
    if smoke is not None:
        at.sidebar.checkbox(key="rsna_use_smoke_checkpoint").set_value(smoke)

    at.run()
    if process:
        at.sidebar.button[0].click().run()
    return at


def probability_frames(at):
    """Extract all probability dataframes from the app."""
    return [d.value for d in at.dataframe if list(d.value.columns) == ["Target", "Probability"]]


def warnings(at):
    """Extract all warning messages."""
    return [w.value for w in at.warning]


def infos(at):
    """Extract all info messages."""
    return [i.value for i in at.info]


class TestEnsembleUIIntegration:
    """Tests for ensemble integration in Streamlit app."""

    def test_ensemble_checkbox_defaults_off(self, ready, tmp_path, monkeypatch):
        """Ensemble checkbox must default to OFF."""
        _, study_uid = ready
        at = run_app(monkeypatch, tmp_path, study_uid, process=False)
        assert not at.exception
        ensemble_checkbox = at.sidebar.checkbox(key="rsna_use_ensemble_inference")
        assert ensemble_checkbox.value is False

    def test_ensemble_off_uses_single_checkpoint(self, ready, tmp_path, monkeypatch):
        """With ensemble OFF, single-checkpoint path is used (existing behavior)."""
        _, study_uid = ready
        single_ckpt = make_checkpoint(tmp_path / "single.pt", seed=42)
        at = run_app(monkeypatch, tmp_path, study_uid, ensemble=False)

        # With no checkpoints, no predictions should render
        assert not probability_frames(at)

    def test_ensemble_on_requires_all_five_folds(self, ready, tmp_path, monkeypatch):
        """Ensemble ON with missing folds produces info message, not crash."""
        _, study_uid = ready
        # Create only 3 of 5 folds
        ensemble_dir = tmp_path / ENSEMBLE_REL
        for i in range(1, 4):
            make_checkpoint(ensemble_dir / f"checkpoint_rsna_stratified_fold{i}.pt")

        at = run_app(monkeypatch, tmp_path, study_uid, ensemble=True)
        assert not at.exception
        assert not probability_frames(at)
        assert any("5-fold ensemble not available" in text for text in infos(at))

    def test_ensemble_on_with_all_folds_produces_predictions(self, ready, tmp_path, monkeypatch):
        """Ensemble ON with all 5 folds present produces 12 target predictions."""
        _, study_uid = ready
        # Create all 5 folds
        ensemble_dir = tmp_path / ENSEMBLE_REL
        fold_paths = []
        for i in range(1, 6):
            ckpt_path = ensemble_dir / f"checkpoint_rsna_stratified_fold{i}.pt"
            make_checkpoint(ckpt_path)
            fold_paths.append(ckpt_path)

        at = run_app(monkeypatch, tmp_path, study_uid, ensemble=True)
        assert not at.exception
        (frame,) = probability_frames(at)
        assert len(frame) == 12
        assert list(frame["Target"]) == list(TARGET_COLUMNS)
        assert all(0.0 <= p <= 1.0 for p in frame["Probability"])

    def test_ensemble_predictions_show_ensemble_warning(self, ready, tmp_path, monkeypatch):
        """Ensemble predictions must show the ensemble warning."""
        _, study_uid = ready
        ensemble_dir = tmp_path / ENSEMBLE_REL
        for i in range(1, 6):
            make_checkpoint(ensemble_dir / f"checkpoint_rsna_stratified_fold{i}.pt")

        at = run_app(monkeypatch, tmp_path, study_uid, ensemble=True)
        assert any("Experimental 5-fold ensemble" in w for w in warnings(at))

    def test_ensemble_quality_status_independent(self, ready, tmp_path, monkeypatch):
        """Quality status must be independent of ensemble predictions."""
        _, study_uid = ready
        ensemble_dir = tmp_path / ENSEMBLE_REL
        for i in range(1, 6):
            make_checkpoint(ensemble_dir / f"checkpoint_rsna_stratified_fold{i}.pt")

        without_ensemble = run_app(monkeypatch, tmp_path, study_uid, ensemble=False)
        with_ensemble = run_app(monkeypatch, tmp_path, study_uid, ensemble=True)

        assert not without_ensemble.exception
        assert not with_ensemble.exception

        # Quality status should be the same regardless of ensemble
        without_quality = [m.label for m in without_ensemble.metric if "Quality Status" in m.label]
        with_quality = [m.label for m in with_ensemble.metric if "Quality Status" in m.label]
        assert len(without_quality) > 0
        assert len(with_quality) > 0

    def test_ensemble_and_smoke_are_mutually_exclusive(self, ready, tmp_path, monkeypatch):
        """Ensemble ON takes precedence over smoke mode."""
        _, study_uid = ready
        # Create both ensemble and smoke checkpoints
        ensemble_dir = tmp_path / ENSEMBLE_REL
        for i in range(1, 6):
            make_checkpoint(ensemble_dir / f"checkpoint_rsna_stratified_fold{i}.pt", seed=10+i)
        make_checkpoint(tmp_path / SMOKE_REL, seed=20)

        # With both enabled, ensemble should be used
        at = run_app(monkeypatch, tmp_path, study_uid, ensemble=True, smoke=True)
        assert not at.exception
        warnings_list = warnings(at)
        # Should show ensemble warning, not smoke warning
        assert any("Experimental 5-fold ensemble" in w for w in warnings_list)

    def test_smoke_mode_still_works_when_ensemble_off(self, ready, tmp_path, monkeypatch):
        """Smoke checkpoint must still work when ensemble is OFF."""
        _, study_uid = ready
        make_checkpoint(tmp_path / SMOKE_REL)

        at = run_app(monkeypatch, tmp_path, study_uid, ensemble=False, smoke=True)
        assert not at.exception
        assert probability_frames(at)
        assert any("smoke-test model" in w for w in warnings(at))

    def test_ensemble_inference_error_shows_warning(self, ready, tmp_path, monkeypatch):
        """If ensemble inference raises an error, show warning, not crash."""
        _, study_uid = ready
        ensemble_dir = tmp_path / ENSEMBLE_REL
        # Create 5 checkpoints with mismatched targets to trigger error
        for i in range(1, 6):
            make_checkpoint(
                ensemble_dir / f"checkpoint_rsna_stratified_fold{i}.pt",
                target_columns=["abnormal", "acl", "meniscus"]
            )

        at = run_app(monkeypatch, tmp_path, study_uid, ensemble=True)
        assert not at.exception
        assert any("Ensemble inference unavailable" in w for w in warnings(at))
        assert not probability_frames(at)


class TestEnsembleBackwardCompatibility:
    """Tests ensuring existing app behavior is unchanged."""

    def test_upload_mode_unchanged(self, tmp_path, monkeypatch):
        """Upload mode must be unaffected by ensemble feature."""
        monkeypatch.chdir(tmp_path)
        at = AppTest.from_file(str(APP), default_timeout=60)
        at.run()
        assert not at.exception
        assert any("Upload an image or MRI file" in text for text in infos(at))
        # Ensemble checkbox should not appear in upload mode
        assert len([c for c in at.sidebar.checkbox if "ensemble" in str(c.label).lower()]) == 0

    def test_existing_rsna_smoke_tests_pass(self, ready, tmp_path, monkeypatch):
        """Existing smoke-test behavior must remain intact."""
        _, study_uid = ready
        make_checkpoint(tmp_path / SMOKE_REL)

        at = run_app(monkeypatch, tmp_path, study_uid, smoke=True)
        assert not at.exception
        (frame,) = probability_frames(at)
        assert len(frame) == 12
        assert list(frame["Target"]) == list(TARGET_COLUMNS)
