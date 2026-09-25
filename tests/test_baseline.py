import csv
import hashlib
import json

import numpy as np
import pytest

from mri_core.baseline import classification_metrics, main, train_and_evaluate, validate_output_dir
from mri_core.labels import TARGETS, load_labels
from mri_core.manifest import PLANES
from tests.test_labels import label_dir


def test_metrics_known_predictions_and_confusion_layout():
    metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.4, 0.9]))
    assert metrics["roc_auc"] == 0.75
    assert metrics["average_precision"] == pytest.approx(5/6)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert all(metrics[name] == 0.5 for name in ("accuracy", "precision", "recall", "f1"))


def test_undefined_metrics_reported_as_null_with_notes():
    metrics = classification_metrics(np.array([0, 0]), np.array([0.1, 0.2]))
    assert metrics["roc_auc"] is None and metrics["average_precision"] is None
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0
    assert len(metrics["notes"]) == 5
    json.dumps(metrics, allow_nan=False)


def test_validation_never_changes_fitted_models_or_scaler(label_dir):
    rows = load_labels(label_dir).records
    matrix = np.random.default_rng(0).normal(size=(12, 6))
    result = train_and_evaluate(rows, matrix)
    changed_matrix = matrix.copy()
    changed_matrix[8:] = changed_matrix[8:] * 100 + 500
    changed_rows = [dict(row) for row in rows]
    for row in changed_rows[8:]:
        for target in TARGETS:
            row[target] = 1 - row[target]
    changed = train_and_evaluate(changed_rows, changed_matrix)
    for target in TARGETS:
        scaler = result.models[target].named_steps["scaler"]
        assert scaler.n_samples_seen_ == 8
        assert np.allclose(scaler.mean_, matrix[:8].mean(axis=0))
        assert np.array_equal(scaler.mean_, changed.models[target].named_steps["scaler"].mean_)
        assert np.array_equal(result.models[target].named_steps["classifier"].coef_,
                              changed.models[target].named_steps["classifier"].coef_)
        assert np.array_equal(result.dummies[target].class_prior_, changed.dummies[target].class_prior_)
        assert result.report["targets"][target]["dummy"]["roc_auc"] == 0.5
    assert result.report["split_counts"] == {"train": 8, "valid": 4}


@pytest.mark.parametrize("problem", ["overlap", "single_class", "invalid_label", "unknown_split", "no_valid", "nonfinite", "unaligned"])
def test_invalid_training_inputs_rejected(label_dir, problem):
    rows = [dict(row) for row in load_labels(label_dir).records]
    matrix = np.ones((12, 3))
    if problem == "overlap":
        rows[8]["exam_id"] = rows[0]["exam_id"]
    elif problem == "single_class":
        for row in rows[:8]:
            row["acl"] = 0
    elif problem == "invalid_label":
        rows[0]["acl"] = 2
    elif problem == "unknown_split":
        rows[0]["split"] = "test"
    elif problem == "no_valid":
        for row in rows:
            row["split"] = "train"
    elif problem == "nonfinite":
        matrix[0, 0] = np.nan
    else:
        matrix = matrix[:-1]
    with pytest.raises(ValueError):
        train_and_evaluate(rows, matrix)


def test_output_cannot_overlap_source_directories(tmp_path):
    source = tmp_path / "source"
    for output in (source, source / "outputs", tmp_path):
        with pytest.raises(ValueError, match="separate"):
            validate_output_dir(output, source, source / "labels")


def test_synthetic_cli_end_to_end_reports_recovery_and_preserves_sources(label_dir, tmp_path, capsys):
    root = tmp_path / "images"
    rng = np.random.default_rng(42)
    for plane in PLANES:
        directory = root / plane
        directory.mkdir(parents=True)
        for i in range(12):
            np.save(directory / f"{i:04d}.npy", rng.integers(0, 256, size=(3, 16, 16), dtype=np.uint8))
    anomaly = label_dir / "train_abnormal.csv"
    anomaly.write_text(anomaly.read_text().replace("0000,0", '"_0000","_0"'))
    paths = list(root.rglob("*.npy")) + list(label_dir.glob("*.csv"))
    original_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output = tmp_path / "derived"
    assert main(["--dataset-root", str(root), "--labels-dir", str(label_dir),
                 "--output-dir", str(output), "--recover-first-row"]) == 0
    assert "CSV recovery:" in capsys.readouterr().out
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["manifest"]["exam_count"] == 12
    assert report["manifest"]["plane_counts"] == dict.fromkeys(PLANES, 12)
    assert len(report["label_recoveries"]) == 1
    assert len(report["feature_names"]) == 60
    assert report["feature_extraction_seconds"] > 0 and report["training_seconds"] > 0
    for target in TARGETS:
        assert report["targets"][target]["train"]["count"] == 8
        assert report["targets"][target]["valid"]["count"] == 4
        assert set(report["targets"][target]["logistic_regression"]) == {
            "roc_auc", "average_precision", "accuracy", "precision", "recall", "f1", "confusion_matrix", "notes"}
    with (output / "manifest.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream))[0]["exam_id"] == "0000"
    assert {p.name for p in output.iterdir()} == {"manifest.csv", "features.csv", "report.json"}
    assert original_hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_cli_reports_incomplete_manifest_without_training(label_dir, tmp_path):
    root = tmp_path / "empty_images"
    root.mkdir()
    output = tmp_path / "derived"
    with pytest.raises(ValueError, match="No model fitted"):
        main(["--dataset-root", str(root), "--labels-dir", str(label_dir), "--output-dir", str(output)])
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "incomplete_manifest"
    assert len(report["manifest"]["missing_images"]) == 36
    assert not (output / "features.csv").exists()
