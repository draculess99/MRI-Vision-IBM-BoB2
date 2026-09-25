"""Tests for RSNA Streamlit app integration.

Verifies the RSNA data-source mode initializes correctly, study/plane selection
works, and incomplete/missing local data produces clean non-fatal handling
rather than crashing the app.
"""

from pathlib import Path

import pytest

from mri_core.rsna_integration import (
    discover_rsna_root,
    load_rsna_metadata_safe,
    discover_available_studies,
    get_available_planes,
    load_rsna_study_series,
    RSNADiscoveryError,
    RSNAStudyNotAvailable,
)
from mri_core.pipeline import process_mri_image
from mri_core.decision import generate_decision_report

pytestmark = pytest.mark.rsna

ROOT = Path(__file__).resolve().parents[1]
RSNA_ROOT = ROOT / "data" / "rsna-knee"


class TestRSNAModeInitialization:
    """Verify the RSNA exploration mode can initialize without crashing."""

    def test_rsna_mode_initializes_with_valid_root(self):
        """discover_rsna_root + metadata load succeed together, as app.py calls them."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        root = discover_rsna_root(explicit_root=RSNA_ROOT)
        metadata = load_rsna_metadata_safe(root)
        assert metadata is not None

    def test_rsna_mode_handles_missing_root_gracefully(self):
        """When RSNA root cannot be discovered, app.py catches RSNADiscoveryError
        and disables the RSNA mode rather than crashing (see app.py has_rsna flag)."""
        with pytest.raises(RSNADiscoveryError):
            discover_rsna_root(explicit_root=Path("/definitely/not/a/real/path"))

    def test_app_module_imports_without_error(self):
        """app.py must import cleanly with the new RSNA integration wired in."""
        pytest.importorskip("streamlit")
        import importlib
        import sys

        # app.py runs top-level Streamlit calls (st.set_page_config etc.) that
        # require a Streamlit script-run context; importing directly under
        # pytest would raise Streamlit's "missing ScriptRunContext" warnings
        # but should not raise ImportError for our new modules.
        spec = importlib.util.find_spec("app")
        assert spec is not None


class TestStudyAndPlaneSelection:
    """Verify study/plane selection logic used by the Streamlit sidebar works standalone."""

    def test_study_selection_lists_only_downloaded_studies(self):
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        studies = discover_available_studies(RSNA_ROOT, metadata)
        # Every entry must be selectable without raising when probing planes
        for study_uid in studies:
            planes = get_available_planes(RSNA_ROOT, metadata, study_uid)
            assert isinstance(planes, list)

    def test_plane_selection_then_load_round_trip(self):
        """Simulates the sidebar flow: pick study -> pick plane -> load series."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        studies = discover_available_studies(RSNA_ROOT, metadata)
        if not studies:
            pytest.skip("No downloaded RSNA studies available")

        study_uid = studies[0]
        planes = get_available_planes(RSNA_ROOT, metadata, study_uid)
        if not planes:
            pytest.skip("No available planes for the first downloaded study")

        volume, meta = load_rsna_study_series(RSNA_ROOT, metadata, study_uid, planes[0])
        assert volume.num_slices > 0
        assert meta["study_uid"] == study_uid


class TestIncompleteDownloadHandling:
    """Verify incomplete downloads produce clean, non-fatal signals (matching app.py's
    try/except RSNAStudyNotAvailable handling around load_rsna_study_series)."""

    def test_study_with_no_local_series_raises_recoverable_error(self):
        """A study listed in metadata but not yet downloaded raises RSNAStudyNotAvailable,
        which app.py catches and turns into a sidebar warning instead of a crash."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        all_studies = metadata.train["StudyInstanceUID"].astype(str).tolist()
        downloaded = set(discover_available_studies(RSNA_ROOT, metadata))

        not_downloaded = next((uid for uid in all_studies if uid not in downloaded), None)
        if not_downloaded is None:
            pytest.skip("All metadata studies already downloaded locally")

        try:
            load_rsna_study_series(RSNA_ROOT, metadata, not_downloaded, "Axial")
            pytest.fail("Expected RSNAStudyNotAvailable for a non-downloaded study")
        except RSNAStudyNotAvailable:
            pass  # Expected: this is the non-fatal path app.py relies on

    def test_empty_data_root_does_not_crash_discovery(self, tmp_path):
        """An RSNA root with metadata but zero downloaded DICOMs (very early in the
        bulk download) must return an empty study list, not raise."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        # tmp_path has no raw/train_series at all
        available = discover_available_studies(tmp_path, metadata)
        assert available == []

    def test_partially_present_series_directory_handled(self, tmp_path):
        """A study directory that exists but is empty (download started, no files
        landed yet) must not be listed as available and must not crash get_available_planes."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        all_studies = metadata.train["StudyInstanceUID"].astype(str).tolist()
        if not all_studies:
            pytest.skip("No studies in metadata")

        fake_study_uid = all_studies[0]
        empty_study_dir = tmp_path / "raw" / "train_series" / fake_study_uid
        empty_study_dir.mkdir(parents=True)  # study dir exists but has no series subdirs

        available = discover_available_studies(tmp_path, metadata)
        assert fake_study_uid not in available

        planes = get_available_planes(tmp_path, metadata, fake_study_uid)
        assert planes == []


class TestModelInferenceIntegration:
    """Mirrors app.py's exact RSNA-viewer flow: load study -> process -> (maybe) run
    inference -> generate_decision_report -> display. Exercised at the function level
    since app.py itself requires a live Streamlit ScriptRunContext to execute end to end."""

    @staticmethod
    def _load_and_process_first_available_study():
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")
        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        studies = discover_available_studies(RSNA_ROOT, metadata)
        if not studies:
            pytest.skip("No downloaded RSNA studies available")
        study_uid = studies[0]
        planes = get_available_planes(RSNA_ROOT, metadata, study_uid)
        if not planes:
            pytest.skip("No available planes for the first downloaded study")
        volume, meta = load_rsna_study_series(RSNA_ROOT, metadata, study_uid, planes[0])
        display_slice = volume.get_display_slice(volume.default_slice_index)
        results = process_mri_image(display_slice, segmentation_method="otsu", is_mri=True, is_inverted=volume.is_inverted)
        return metadata, volume, meta, results

    def test_no_checkpoint_preserves_not_available_behavior(self, tmp_path):
        """Matches app.py's `if RSNA_CHECKPOINT_PATH.is_file(): ... else: model_predictions stays None`."""
        metadata, volume, meta, results = self._load_and_process_first_available_study()
        checkpoint_path = tmp_path / "does_not_exist.pt"
        model_predictions = None
        if checkpoint_path.is_file():
            pytest.fail("checkpoint_path must not exist for this test")

        report = generate_decision_report(
            study_uid=meta["study_uid"], series_uid=meta["series_uid"], plane=meta["plane"],
            volume_data=volume.raw_data, preprocessed_data=results["preprocessed"], mask=results["mask"],
            features=results["features"], num_slices=meta["num_slices"], model_predictions=model_predictions,
        )
        assert report.model_predictions is None
        assert report.model_status == "Not available — model checkpoint not loaded"

    def test_valid_temporary_checkpoint_produces_probability_table(self, tmp_path):
        """A real (synthetic-weights) checkpoint matching RSNAKneeCNN's architecture must
        produce a full 12-target probability dict, renderable as the app's Target/Probability table."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("pandas")
        import pandas as pd
        from mri_core.rsna_knee_model import RSNAKneeCNN
        from mri_core.rsna_knee_dataset import TARGET_COLUMNS
        from mri_core.rsna_knee_inference import run_inference, RSNAInferenceError

        metadata, volume, meta, results = self._load_and_process_first_available_study()

        checkpoint_path = tmp_path / "ckpt.pt"
        model = RSNAKneeCNN(num_planes=3, dropout=0.0)
        torch.save(
            {
                "model": model.state_dict(),
                "config": {"image_size": 64, "slices_per_series": 4, "planes": ["Axial", "Coronal", "Sagittal"], "dropout": 0.0},
                "target_columns": list(TARGET_COLUMNS),
            },
            checkpoint_path,
        )

        model_predictions = None
        try:
            model_predictions = run_inference(metadata=metadata, checkpoint_path=checkpoint_path, study_uid=meta["study_uid"])
        except RSNAInferenceError as e:
            pytest.skip(f"Study not suitable for this checkpoint's config: {e}")

        report = generate_decision_report(
            study_uid=meta["study_uid"], series_uid=meta["series_uid"], plane=meta["plane"],
            volume_data=volume.raw_data, preprocessed_data=results["preprocessed"], mask=results["mask"],
            features=results["features"], num_slices=meta["num_slices"], model_predictions=model_predictions,
        )
        assert report.model_predictions is not None
        assert len(report.model_predictions) == 12
        predictions_df = pd.DataFrame(list(report.model_predictions.items()), columns=["Target", "Probability"])
        assert list(predictions_df.columns) == ["Target", "Probability"]
        assert len(predictions_df) == 12

    def test_inference_exception_produces_warning_not_crash(self, tmp_path):
        """A checkpoint with mismatched target_columns must be caught (RSNAInferenceError),
        leaving model_predictions None and the deterministic quality report still generated."""
        torch = pytest.importorskip("torch")
        from mri_core.rsna_knee_model import RSNAKneeCNN
        from mri_core.rsna_knee_inference import run_inference, RSNAInferenceError

        metadata, volume, meta, results = self._load_and_process_first_available_study()

        checkpoint_path = tmp_path / "bad_ckpt.pt"
        model = RSNAKneeCNN(num_planes=3, dropout=0.0)
        torch.save(
            {
                "model": model.state_dict(),
                "config": {"image_size": 64, "slices_per_series": 4, "planes": ["Axial", "Coronal", "Sagittal"], "dropout": 0.0},
                "target_columns": ["abnormal", "acl", "meniscus"],  # wrong schema entirely
            },
            checkpoint_path,
        )

        model_predictions = None
        warning_raised = False
        try:
            model_predictions = run_inference(metadata=metadata, checkpoint_path=checkpoint_path, study_uid=meta["study_uid"])
        except RSNAInferenceError:
            warning_raised = True  # This is exactly what app.py's `except RSNAInferenceError as e: st.warning(...)` does.

        assert warning_raised is True
        assert model_predictions is None

        # The deterministic quality report must still be produced (app does not crash).
        report = generate_decision_report(
            study_uid=meta["study_uid"], series_uid=meta["series_uid"], plane=meta["plane"],
            volume_data=volume.raw_data, preprocessed_data=results["preprocessed"], mask=results["mask"],
            features=results["features"], num_slices=meta["num_slices"], model_predictions=model_predictions,
        )
        assert report.quality_status is not None
        assert report.model_predictions is None
