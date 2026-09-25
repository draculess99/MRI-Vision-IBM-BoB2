import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Install requirements-cnn.txt to run optional CNN tests")

from mri_core.cnn_dataset import MRNetDataset, augment_plane, training_class_weights
from mri_core.cnn_model import MRNetCNN
from mri_core import cnn_train
from mri_core.labels import TARGETS, load_labels
from mri_core.manifest import Manifest, build_manifest


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    value = json.loads(cnn_train.DEFAULT_CONFIG.read_text())
    value.update({"slices": 3, "resolution": 32, "epochs": 1, "batch_size": 4,
                  "num_workers": 0, "seeds": [0], "expected_train": 8, "expected_valid": 4})
    return value


@pytest.fixture
def synthetic(tmp_path):
    root = tmp_path / "source"
    labels = cnn_train.smoke_fixture(root)
    manifest = build_manifest(load_labels(labels), root)
    return root, labels, manifest


@pytest.mark.parametrize("planes", [["axial"], ["axial", "coronal", "sagittal"]])
def test_dataset_sampling_padding_shapes_and_label_alignment(synthetic, config, planes):
    root, _, manifest = synthetic
    config["planes"] = planes
    source = root / "axial" / "0000.npy"
    original = source.read_bytes()
    dataset = MRNetDataset(manifest, "train", config)
    batch = dataset[0]
    assert batch["exam_id"] == "0000"
    assert batch["images"].shape == (len(planes), 3, 1, 32, 32)
    assert batch["mask"].tolist() == [[True, True, False]] * len(planes)
    assert batch["indices"].tolist() == [[0, 1, -1]] * len(planes)
    assert torch.count_nonzero(batch["images"][:, 2]) == 0
    assert batch["labels"].tolist() == [0, 1, 0]
    assert dataset[-1]["exam_id"] == "0007"
    assert MRNetDataset(manifest, "valid", config)[0]["exam_id"] == "0008"
    assert torch.isfinite(batch["images"]).all()
    assert source.read_bytes() == original


def test_long_volume_uses_evenly_spaced_sampling(synthetic, config):
    root, _, manifest = synthetic
    np.save(root / "axial" / "0000.npy", np.zeros((10, 32, 32), dtype=np.uint8))
    assert MRNetDataset(manifest, "train", config)[0]["indices"][0].tolist() == [0, 4, 9]


def test_augmentation_is_reproducible_and_shared_within_plane(synthetic, config):
    _, _, manifest = synthetic
    image = torch.rand(1, 1, 32, 32).repeat(2, 1, 1, 1)
    transformed = augment_plane(image, torch.Generator().manual_seed(7))
    assert torch.equal(transformed[0], transformed[1])
    assert torch.equal(transformed, augment_plane(image, torch.Generator().manual_seed(7)))
    dataset = MRNetDataset(manifest, "train", config, augment=True, seed=7)
    first = dataset[0]["images"]
    assert torch.equal(first, dataset[0]["images"])
    dataset.set_epoch(1)
    assert not torch.equal(first, dataset[0]["images"])
    valid = MRNetDataset(manifest, "valid", config)
    before = valid[0]["images"]
    valid.set_epoch(10)
    assert torch.equal(before, valid[0]["images"])
    with pytest.raises(ValueError, match="training split"):
        MRNetDataset(manifest, "valid", config, augment=True)


def test_class_weights_never_use_validation_labels(synthetic):
    _, _, manifest = synthetic
    rows = copy.deepcopy(manifest.rows)
    for row in rows:
        if row["split"] == "valid":
            for target in TARGETS:
                row[target] = 1
    assert torch.equal(training_class_weights(rows), training_class_weights(manifest.rows))
    assert training_class_weights(rows).tolist() == [1, 1, 1]
    rows[0]["acl"] = 0
    assert training_class_weights(rows)[1] == pytest.approx(5 / 3)
    for row in rows:
        if row["split"] == "train":
            row["acl"] = 0
    with pytest.raises(ValueError, match="both classes"):
        training_class_weights(rows)


def test_model_shapes_padding_invariance_and_gradient_flow():
    cnn_train.seed_everything(0)
    model = MRNetCNN(3, dropout=0)
    images = torch.rand(2, 3, 3, 1, 32, 32)
    mask = torch.ones(2, 3, 3, dtype=torch.bool)
    mask[:, :, 2] = False
    logits = model(images, mask)
    assert logits.shape == (2, 3)
    changed = images.clone()
    changed[:, :, 2] = float("nan")
    assert torch.equal(logits, model(changed, mask))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert 90000 < sum(p.numel() for p in model.parameters()) < 110000
    mask[:, 0] = False
    with pytest.raises(ValueError, match="at least one"):
        model(images, mask)


def test_checkpoint_reload_prediction_parity(tmp_path, config):
    model = MRNetCNN(3, config["dropout"]).eval()
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "last.pt"
    cnn_train.save_checkpoint(path, model, optimizer, config, 0, 1, torch.Generator().manual_seed(0))
    loaded, checkpoint = cnn_train.load_checkpoint(path, torch.device("cpu"))
    images, mask = torch.rand(2, 3, 2, 1, 32, 32), torch.ones(2, 3, 2, dtype=torch.bool)
    assert torch.equal(model(images, mask), loaded(images, mask))
    assert checkpoint["targets"] == list(TARGETS)
    assert checkpoint["epoch"] == 1
    assert "optimizer" in checkpoint and "loader_rng" in checkpoint


def test_training_is_reproducible_and_validation_independent(synthetic, config, tmp_path):
    _, _, manifest = synthetic
    config["planes"] = ["axial"]
    first = cnn_train.train_one_seed(manifest, config, 0, tmp_path / "first", torch.device("cpu"), compare_reference=False)
    rows = copy.deepcopy(manifest.rows)
    for row in rows:
        if row["split"] == "valid":
            for target in TARGETS:
                row[target] = 1 - row[target]
    changed = Manifest(rows, manifest.paths, (), ())
    cnn_train.train_one_seed(changed, config, 0, tmp_path / "second", torch.device("cpu"), compare_reference=False)
    _, a = cnn_train.load_checkpoint(tmp_path / "first" / "last.pt", torch.device("cpu"))
    _, b = cnn_train.load_checkpoint(tmp_path / "second" / "last.pt", torch.device("cpu"))
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    assert first["status"] == "complete"
    assert first["training_seconds"] > 0


def test_cpu_full_run_refused_and_smoke_rejects_real_paths(monkeypatch, synthetic, config, tmp_path):
    root, labels, _ = synthetic
    with pytest.raises(ValueError, match="require CUDA"):
        cnn_train.execute(root, labels, tmp_path / "outputs", config, torch.device("cpu"))
    with pytest.raises(SystemExit):
        cnn_train.main(["smoke", "--dataset-root", str(root)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        cnn_train.main(["train", "--dataset-root", str(root)])


def test_output_guard_rejects_tracked_and_source_locations(tmp_path):
    for output in (cnn_train.REPO_ROOT / "models", cnn_train.REPO_ROOT / "outputs" / "v0.4", tmp_path / "source"):
        with pytest.raises(ValueError):
            cnn_train.prepare_output(output, tmp_path / "source", tmp_path / "source" / "labels")


def test_manifest_fingerprint_detects_split_and_label_changes(synthetic):
    _, _, manifest = synthetic
    before = cnn_train.manifest_fingerprint(manifest.rows)
    assert before == cnn_train.manifest_fingerprint(tuple(reversed(manifest.rows)))
    changed = copy.deepcopy(manifest.rows)
    changed[0]["acl"] = 1 - changed[0]["acl"]
    assert cnn_train.manifest_fingerprint(changed) != before
    changed[0]["acl"] = 1 - changed[0]["acl"]
    changed[0]["split"] = "valid"
    assert cnn_train.manifest_fingerprint(changed) != before


@pytest.mark.parametrize("key,value", [("threshold", 0.4), ("epochs", 0), ("planes", ["axial", "axial"]),
                                      ("seeds", [0, 0]), ("resolution", 8), ("learning_rate", float("nan"))])
def test_invalid_experiment_config_rejected(config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        cnn_train.validate_config(config)


def test_synthetic_execute_artifacts_and_source_hashes(synthetic, config, tmp_path, monkeypatch):
    root, labels, _ = synthetic
    output = tmp_path / "outputs"
    monkeypatch.setattr(cnn_train, "ARTIFACT_ROOTS", (output,))
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    config["planes"] = ["axial"]
    directory = cnn_train.execute(root, labels, output, config, torch.device("cpu"), smoke=True)
    assert json.loads((directory / "provenance.json").read_text())["mode"] == "synthetic_smoke"
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["seeds"] == [0]
    assert len(summary["metrics"]["acl"]["accuracy"]["values"]) == 1
    assert (directory / "seed-0" / "predictions.csv").is_file()
    assert (directory / "seed-0" / "last.pt").is_file()
    assert not list(directory.rglob("*.npy"))
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
