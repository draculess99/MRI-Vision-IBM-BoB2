"""UI tests for the opt-in RSNA smoke-checkpoint path, driven through Streamlit's headless AppTest.

Checkpoints used here are synthetic (random-init RSNAKneeCNN) files in temp directories; the app is run with
its working directory set there, so its relative checkpoint paths resolve to the temp files. Real local RSNA
studies are read (never written); tests skip cleanly when no inference-ready study is available.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
torch = pytest.importorskip("torch")
pytest.importorskip("pandas")

from streamlit.testing.v1 import AppTest

from mri_core.rsna_integration import discover_available_studies, load_rsna_metadata_safe
from mri_core.rsna_knee_dataset import TARGET_COLUMNS
from mri_core.rsna_knee_inference import RSNAInferenceError, run_inference
from mri_core.rsna_knee_model import RSNAKneeCNN

pytestmark = pytest.mark.rsna

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
RSNA_ROOT = ROOT / "data" / "rsna-knee"
SMOKE_REL = Path("outputs/rsna-knee-smoke/checkpoint_smoke.pt")
PROD_REL = Path("outputs/rsna-knee/checkpoint_best.pt")
SMOKE_WARNING = ("Experimental smoke-test model — trained on only 3 studies; "
                 "outputs are not clinically meaningful and are not a diagnosis.")
NOT_AVAILABLE = "Not available — model checkpoint not loaded"
CHECKBOX_KEY = "rsna_use_smoke_checkpoint"


def make_checkpoint(path, target_columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.save(
        {
            "model": RSNAKneeCNN(num_planes=3, dropout=0.0).state_dict(),
            "config": {"image_size": 16, "slices_per_series": 2, "planes": ["Axial", "Coronal", "Sagittal"], "dropout": 0.0},
            "target_columns": list(target_columns) if target_columns is not None else list(TARGET_COLUMNS),
        },
        path,
    )
    return path


@pytest.fixture(scope="module")
def ready(tmp_path_factory):
    """First locally available RSNA study that a 3-plane checkpoint can run on, plus the loaded metadata."""
    if not (RSNA_ROOT / "train.csv").is_file():
        pytest.skip("RSNA data directory not found")
    metadata = load_rsna_metadata_safe(RSNA_ROOT)
    probe = make_checkpoint(tmp_path_factory.mktemp("probe") / "probe.pt")
    for study_uid in discover_available_studies(RSNA_ROOT, metadata):
        try:
            run_inference(metadata, probe, study_uid)
        except RSNAInferenceError:
            continue
        return metadata, study_uid
    pytest.skip("No fully downloaded RSNA study available")


def run_app(monkeypatch, workdir, study_uid, *, smoke=None, process=True):
    """Drive app.py in RSNA mode from `workdir`. smoke=None leaves the checkbox untouched (default off)."""
    monkeypatch.chdir(workdir)
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    next(r for r in at.sidebar.radio if r.label == "Data Source").set_value("Explore RSNA Studies")
    at.run()
    next(s for s in at.sidebar.selectbox if s.label == "Select Study").set_value(study_uid)
    if smoke is not None:
        at.sidebar.checkbox(key=CHECKBOX_KEY).set_value(smoke)
    at.run()
    if process:
        at.sidebar.button[0].click().run()
    return at


def probability_frames(at):
    return [d.value for d in at.dataframe if list(d.value.columns) == ["Target", "Probability"]]


def quality_status(at):
    return next(m for m in at.metric if m.label == "Quality Status").value


def warnings(at):
    return [w.value for w in at.warning]


def infos(at):
    return [i.value for i in at.info]


# ------------------------------------------------------------------------------ smoke checkpoint present

def test_smoke_checkpoint_enabled_shows_twelve_ordered_probabilities_with_warning(ready, tmp_path, monkeypatch):
    metadata, study_uid = ready
    checkpoint = make_checkpoint(tmp_path / SMOKE_REL)
    expected = run_inference(metadata, checkpoint, study_uid)

    at = run_app(monkeypatch, tmp_path, study_uid, smoke=True)

    assert not at.exception
    (frame,) = probability_frames(at)
    assert list(frame["Target"]) == list(TARGET_COLUMNS) and len(frame) == 12
    assert list(frame["Probability"]) == pytest.approx(list(expected.values()))
    assert all(0.0 <= p <= 1.0 for p in frame["Probability"])
    assert SMOKE_WARNING in warnings(at)
    assert any("predictions available" in text for text in infos(at))
    assert any("not a clinical diagnosis" in c.value for c in at.caption)


def test_smoke_warning_is_rendered_above_the_probability_table(ready, tmp_path, monkeypatch):
    _, study_uid = ready
    make_checkpoint(tmp_path / SMOKE_REL)
    at = run_app(monkeypatch, tmp_path, study_uid, smoke=True)

    elements = list(at.main)
    warning_at = next(i for i, el in enumerate(elements) if el.type == "warning" and el.value == SMOKE_WARNING)
    table_at = next(i for i, el in enumerate(elements)
                    if el.type == "dataframe" and list(el.value.columns) == ["Target", "Probability"])
    assert warning_at < table_at


def test_real_smoke_checkpoint_runs_end_to_end_when_present(ready, monkeypatch):
    real = ROOT / SMOKE_REL
    if not real.is_file():
        pytest.skip("local smoke checkpoint not present")
    _, study_uid = ready
    at = run_app(monkeypatch, ROOT, study_uid, smoke=True)
    assert not at.exception
    (frame,) = probability_frames(at)
    assert list(frame["Target"]) == list(TARGET_COLUMNS)
    assert SMOKE_WARNING in warnings(at)


# ------------------------------------------------------------------------------ smoke checkpoint absent / off

def test_enabled_but_missing_checkpoint_degrades_to_model_unavailable(ready, tmp_path, monkeypatch):
    _, study_uid = ready
    at = run_app(monkeypatch, tmp_path, study_uid, smoke=True)
    assert not at.exception
    assert not probability_frames(at)
    assert SMOKE_WARNING not in warnings(at)
    assert any("Smoke-test checkpoint not found" in text for text in infos(at))
    assert any(NOT_AVAILABLE in text for text in infos(at))
    assert quality_status(at)  # the deterministic quality report still renders


def test_checkpoint_present_but_checkbox_off_is_not_used(ready, tmp_path, monkeypatch):
    _, study_uid = ready
    make_checkpoint(tmp_path / SMOKE_REL)
    at = run_app(monkeypatch, tmp_path, study_uid)  # default: unchecked
    assert not at.exception
    assert at.sidebar.checkbox(key=CHECKBOX_KEY).value is False
    assert not probability_frames(at)
    assert any(NOT_AVAILABLE in text for text in infos(at))


def test_no_checkpoint_anywhere_keeps_previous_behavior(ready, tmp_path, monkeypatch):
    _, study_uid = ready
    at = run_app(monkeypatch, tmp_path, study_uid)
    assert not at.exception
    assert not probability_frames(at)
    assert any(NOT_AVAILABLE in text for text in infos(at))
    assert not any("Smoke-test checkpoint not found" in text for text in infos(at))


# ------------------------------------------------------------------------------ production path unchanged

def test_production_checkpoint_path_still_works_without_smoke_label(ready, tmp_path, monkeypatch):
    _, study_uid = ready
    make_checkpoint(tmp_path / PROD_REL)
    at = run_app(monkeypatch, tmp_path, study_uid)
    assert not at.exception
    (frame,) = probability_frames(at)
    assert list(frame["Target"]) == list(TARGET_COLUMNS)
    assert SMOKE_WARNING not in warnings(at)


def test_enabling_smoke_never_reads_or_writes_the_production_path(ready, tmp_path, monkeypatch):
    _, study_uid = ready
    prod = make_checkpoint(tmp_path / PROD_REL)
    before = prod.read_bytes()
    make_checkpoint(tmp_path / SMOKE_REL, target_columns=["abnormal", "acl", "meniscus"])  # invalid smoke file
    at = run_app(monkeypatch, tmp_path, study_uid, smoke=True)
    assert not at.exception
    assert not probability_frames(at)  # the smoke path was used (and rejected), not the valid production file
    assert prod.read_bytes() == before


# ------------------------------------------------------------------------------ failures and independence

def test_inference_failure_is_a_warning_and_the_quality_report_still_renders(ready, tmp_path, monkeypatch):
    _, study_uid = ready
    make_checkpoint(tmp_path / SMOKE_REL, target_columns=["abnormal", "acl", "meniscus"])
    at = run_app(monkeypatch, tmp_path, study_uid, smoke=True)
    assert not at.exception
    assert any("Model inference unavailable" in text for text in warnings(at))
    assert not probability_frames(at)
    assert quality_status(at) in ("🟢 OK", "🟡 REVIEW", "🔴 INVALID")


def test_quality_status_is_independent_of_the_smoke_model(ready, tmp_path, monkeypatch):
    _, study_uid = ready
    make_checkpoint(tmp_path / SMOKE_REL)
    without_model = run_app(monkeypatch, tmp_path, study_uid, smoke=False)
    with_model = run_app(monkeypatch, tmp_path, study_uid, smoke=True)
    assert probability_frames(with_model) and not probability_frames(without_model)
    assert quality_status(with_model) == quality_status(without_model)
    flags = lambda at: [w for w in warnings(at) if w.startswith("**Quality Flags:**")]
    assert flags(with_model) == flags(without_model)
    metric_values = lambda at: {m.label: m.value for m in at.metric}
    assert metric_values(with_model) == metric_values(without_model)


# ------------------------------------------------------------------------------ other modes unaffected

def test_upload_mode_is_unchanged_and_shows_no_smoke_control(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Upload an image or MRI file" in text for text in infos(at))
    assert len(at.sidebar.checkbox) == 0
    assert not probability_frames(at)


def test_smoke_control_appears_only_in_rsna_mode(ready, tmp_path, monkeypatch):
    _, study_uid = ready
    at = run_app(monkeypatch, tmp_path, study_uid, process=False)
    assert at.sidebar.checkbox(key=CHECKBOX_KEY).label == "Use experimental smoke-test checkpoint"
    assert at.sidebar.checkbox(key=CHECKBOX_KEY).value is False


def test_paths_and_label_are_wired_as_specified():
    source = APP.read_text(encoding="utf-8")
    assert 'RSNA_CHECKPOINT_PATH = Path("outputs/rsna-knee/checkpoint_best.pt")' in source
    assert 'RSNA_SMOKE_CHECKPOINT_PATH = Path("outputs/rsna-knee-smoke/checkpoint_smoke.pt")' in source
    assert "Experimental smoke-test model — trained on only 3 studies; " in source
    assert "outputs are not clinically meaningful and are not a diagnosis." in source
