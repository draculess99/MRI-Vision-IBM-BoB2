"""Opt-in (--run-rsna): read-only checks against the one downloaded RSNA study.

Pixel reading (tests/legacy_rsna_reader.py) must still match to within 1e-6 once fed the same series;
separately, series *selection* is now deterministic (select_series), not the old first-CSV-row pick,
so the equivalence check below drives the legacy reader with select_series rather than its old default.
The study must also still run through the model forward pass.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pandas")

from torch.utils.data import DataLoader

from mri_core.rsna_knee_dataset import TARGET_COLUMNS, load_rsna_metadata, load_series_volume, select_series
from mri_core.rsna_knee_model import RSNAKneeCNN
from mri_core.rsna_knee_train import RSNAKneeDicomDataset, _normalize

from .legacy_rsna_reader import legacy_item, legacy_series_arrays

pytestmark = pytest.mark.rsna_real

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data" / "rsna-knee"
DICOM_ROOT = CSV_DIR / "raw"
STUDY = "1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260"
TOLERANCE = 1e-6
BASE_CONFIG = json.loads((ROOT / "configs" / "rsna_knee.json").read_text(encoding="utf-8"))
CONFIGS = {
    "project-config": BASE_CONFIG,
    "sagittal-axial-64px-5-slices": {**BASE_CONFIG, "image_size": 64, "slices_per_series": 5, "planes": ["Sagittal", "Axial"]},
    "more-slices-than-series": {**BASE_CONFIG, "image_size": 32, "slices_per_series": 30},
}


@pytest.fixture(scope="module")
def real():
    if not (DICOM_ROOT / "train_series" / STUDY).is_dir():
        pytest.skip("downloaded RSNA study not present under data/rsna-knee/raw")
    metadata = load_rsna_metadata(CSV_DIR, dicom_root=DICOM_ROOT)
    frame = metadata.train[metadata.train.StudyInstanceUID.astype(str) == STUDY].reset_index(drop=True)
    assert len(frame) == 1
    return metadata, frame


@pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS.keys())
def test_dataset_output_matches_previous_reading_algorithm_given_the_same_series(real, config):
    """Pixel reading is unchanged: feeding the legacy reader the *same* select_series pick it must
    match the dataset to <=1e-6. (Selection itself changed on purpose; see test_series_selection_*.)"""
    metadata, frame = real
    new = RSNAKneeDicomDataset(metadata, frame, config)[0]
    old = legacy_item(metadata, frame.iloc[0], config, select=select_series)
    worst = float((new["images"] - old["images"]).abs().max())
    print(f"\nmax |new - previous| = {worst:.3e}  images {tuple(new['images'].shape)}  valid slices/plane {new['mask'].sum(1).tolist()}")
    assert new["images"].shape == old["images"].shape
    assert worst <= TOLERANCE
    assert torch.equal(new["mask"], old["mask"])
    torch.testing.assert_close(new["labels"], old["labels"], rtol=0, atol=0, equal_nan=True)
    assert new["study_uid"] == old["study_uid"] == STUDY


def test_series_selection_on_the_real_study_prefers_fluid_sensitive(real):
    """Documents which real series the new rule picks, and where that differs from the old first-row pick."""
    metadata, _ = real
    naive = lambda candidates: str(candidates.iloc[0]["SeriesInstanceUID"])
    rows = metadata.series_for(STUDY)
    changed = []
    for plane in ("Axial", "Coronal", "Sagittal"):
        candidates = rows[rows["Anatomical_Plane"] == plane]
        chosen, previous = select_series(candidates), naive(candidates)
        chosen_row = candidates[candidates.SeriesInstanceUID.astype(str) == chosen].iloc[0]
        print(f"\n{plane}: {len(candidates)} candidate(s); select_series -> ...{chosen[-6:]} "
              f"(Fluid_Sensitive={int(chosen_row.Fluid_Sensitive)}); old first-row pick -> ...{previous[-6:]}"
              + ("  [CHANGED]" if chosen != previous else ""))
        assert bool(chosen_row.Fluid_Sensitive) or not (candidates.Fluid_Sensitive == 1).any()
        if chosen != previous:
            changed.append(plane)
    assert changed == ["Coronal"]


def test_every_slice_of_every_series_matches_previous_normalization(real):
    metadata, _ = real
    series = metadata.series_for(STUDY)["SeriesInstanceUID"].astype(str).tolist()
    assert len(series) == 5
    total, worst = 0, 0.0
    for uid in series:
        previous = legacy_series_arrays(metadata.dicom_series_dir(STUDY, uid))
        volume = load_series_volume(metadata, STUDY, uid)
        assert volume.num_slices == len(previous)
        for index, array in enumerate(previous):
            worst = max(worst, float(abs(_normalize(volume.get_slice(index)) - _normalize(array)).max()))
            total += 1
    print(f"\n{total} slices across {len(series)} series: max |new - previous| = {worst:.3e}")
    assert total == 94
    assert worst <= TOLERANCE


def test_one_real_study_forward_pass_gives_twelve_finite_outputs(real):
    metadata, frame = real
    config = BASE_CONFIG
    batch = next(iter(DataLoader(RSNAKneeDicomDataset(metadata, frame, config), batch_size=1)))
    torch.manual_seed(int(config["seed"]))
    model = RSNAKneeCNN(len(config["planes"]), float(config["dropout"])).eval()
    with torch.inference_mode():
        logits = model(batch["images"], batch["mask"])
    print(f"\nimages {tuple(batch['images'].shape)} mask {tuple(batch['mask'].shape)} "
          f"labels {tuple(batch['labels'].shape)} logits {tuple(logits.shape)}")
    assert tuple(batch["images"].shape) == (1, 3, 9, 1, 128, 128)
    assert tuple(batch["mask"].shape) == (1, 3, 9) and bool(batch["mask"].all())
    assert tuple(batch["labels"].shape) == (1, len(TARGET_COLUMNS))
    assert tuple(logits.shape) == (1, 12)
    assert torch.isfinite(logits).all()
    probabilities = logits.sigmoid()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
