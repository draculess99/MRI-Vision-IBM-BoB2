"""Train-only fitting and held-out evaluation for three MRNet binary targets.

Run with: python -m mri_core.baseline --help
"""

import argparse
import csv
import hashlib
import json
import platform
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baseline_features import FEATURE_NAMES, extract_features
from .dataset_discovery import DEFAULT_MRNET_ROOT
from .labels import SPLITS, TARGETS, load_labels
from .manifest import MANIFEST_FIELDS, PLANES, build_manifest

THRESHOLD = 0.5


@dataclass
class BaselineResult:
    report: dict
    models: dict
    dummies: dict


def distribution(values: np.ndarray) -> dict:
    positive = int(values.sum())
    return {"count": len(values), "negative": len(values) - positive,
            "positive": positive, "prevalence": float(values.mean())}


def classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    predicted = (scores >= THRESHOLD).astype(int)
    both_classes = len(np.unique(labels)) == 2
    notes = []
    if not both_classes:
        notes.append("ROC-AUC is undefined: validation contains only one class.")
    if not np.any(labels == 1):
        notes.append("Average precision is undefined: validation contains no positives.")
    if not np.any(predicted == 1):
        notes.append("Precision uses zero_division=0: no positive predictions.")
    if not np.any(labels == 1):
        notes.append("Recall uses zero_division=0: no positive labels.")
    if not np.any(labels == 1) and not np.any(predicted == 1):
        notes.append("F1 uses zero_division=0: no positive labels or predictions.")
    return {
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "average_precision": float(average_precision_score(labels, scores)) if np.any(labels == 1) else None,
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "notes": notes,
    }


def train_and_evaluate(rows, features: np.ndarray) -> BaselineResult:
    """Fit scaler, logistic regression, and dummy exclusively on train rows."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != len(rows) or features.shape[1] == 0 or not np.isfinite(features).all():
        raise ValueError("Expected a finite feature matrix aligned with manifest rows")
    if any(row["split"] not in SPLITS for row in rows):
        raise ValueError("Unknown split; expected train or valid")
    if len({row["exam_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate examination IDs or train/validation overlap")
    train_mask = np.array([row["split"] == "train" for row in rows])
    valid_mask = ~train_mask
    if not train_mask.any() or not valid_mask.any():
        raise ValueError("Both training and validation examinations are required")
    all_labels = {}
    for target in TARGETS:
        values = np.array([row[target] for row in rows])
        if not np.isin(values, [0, 1]).all():
            raise ValueError(f"{target}: expected binary labels")
        if len(np.unique(values[train_mask])) != 2:
            raise ValueError(f"{target}: training requires both classes")
        all_labels[target] = values.astype(int)
    train_x, valid_x = features[train_mask], features[valid_mask]
    models, dummies, target_reports = {}, {}, {}
    training_seconds = 0.0
    for target in TARGETS:
        train_y, valid_y = all_labels[target][train_mask], all_labels[target][valid_mask]
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs",
                                               max_iter=2000, random_state=0)),
        ])
        dummy = DummyClassifier(strategy="prior", random_state=0)
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train_x, train_y)
        dummy.fit(train_x, train_y)
        training_seconds += time.perf_counter() - start
        target_reports[target] = {
            "train": distribution(train_y), "valid": distribution(valid_y),
            "logistic_regression": classification_metrics(valid_y, model.predict_proba(valid_x)[:, 1]),
            "dummy": classification_metrics(valid_y, dummy.predict_proba(valid_x)[:, 1]),
        }
        models[target], dummies[target] = model, dummy
    report = {
        "split_counts": {"train": int(train_mask.sum()), "valid": int(valid_mask.sum())},
        "training_seconds": training_seconds,
        "configuration": {"threshold": THRESHOLD, "C": 1.0, "class_weight": "balanced",
                          "solver": "lbfgs", "max_iter": 2000, "random_state": 0,
                          "dummy_strategy": "prior", "fitting_split": "train",
                          "confusion_matrix_layout": "[[TN, FP], [FN, TP]]"},
        "targets": target_reports,
    }
    return BaselineResult(report, models, dummies)


def validate_output_dir(output_dir: Path, dataset_root: Path, label_dir: Path) -> Path:
    """Do not allow derived artifacts within, or as ancestors of, source directories."""
    output_dir = output_dir.resolve()
    for source in (dataset_root.resolve(), label_dir.resolve()):
        if output_dir == source or output_dir in source.parents or source in output_dir.parents:
            raise ValueError(f"Output directory must be separate from source data: {source}")
    for filename in ("manifest.csv", "features.csv", "report.json"):
        artifact = output_dir / filename
        if artifact.is_symlink():
            raise ValueError(f"Refusing to overwrite symlink: {artifact}")
    return output_dir


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_MRNET_ROOT))
    parser.add_argument("--labels-dir", type=Path, help="Default: <dataset-root>/labels")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v0.4"))
    parser.add_argument("--recover-first-row", action="store_true",
                        help="Explicitly recover and report _dddd,_0/_1 first rows; source CSVs remain unchanged")
    args = parser.parse_args(argv)
    label_dir = args.labels_dir if args.labels_dir is not None else args.dataset_root / "labels"
    output_dir = validate_output_dir(args.output_dir, args.dataset_root, label_dir)
    source_hashes = {f"{split}_{target}.csv": hashlib.sha256(
        (label_dir / f"{split}_{target}.csv").read_bytes()).hexdigest()
                     for split in SPLITS for target in TARGETS}
    labels = load_labels(label_dir, recover_first_row=args.recover_first_row)
    for recovery in labels.recoveries:
        print("CSV recovery: " + json.dumps(recovery), flush=True)
    manifest = build_manifest(labels, args.dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "manifest.csv", MANIFEST_FIELDS, manifest.rows)
    manifest_report = {
        "exam_count": len(manifest.rows),
        "plane_counts": {plane: sum(row[f"{plane}_available"] for row in manifest.rows) for plane in PLANES},
        "missing_images": list(manifest.missing_images),
        "unlabeled_images": list(manifest.unlabeled_images),
    }
    if manifest.missing_images or manifest.unlabeled_images:
        (output_dir / "report.json").write_text(json.dumps({"status": "incomplete_manifest",
            "manifest": manifest_report, "label_recoveries": labels.recoveries}, indent=2), encoding="utf-8")
        raise ValueError("Manifest contains missing or unlabeled images; see report.json. No model fitted.")
    start = time.perf_counter()
    def progress(done, total):
        if done == 1 or done % 100 == 0 or done == total:
            print(f"Feature extraction: {done}/{total} examinations", flush=True)
    features = extract_features(manifest, progress=progress)
    extraction_seconds = time.perf_counter() - start
    write_csv(output_dir / "features.csv", ("exam_id", "split", *FEATURE_NAMES),
              ({"exam_id": row["exam_id"], "split": row["split"],
                **dict(zip(FEATURE_NAMES, vector))} for row, vector in zip(manifest.rows, features)))
    result = train_and_evaluate(manifest.rows, features)
    result.report.update({
        "status": "complete", "manifest": manifest_report,
        "label_recoveries": labels.recoveries, "label_sha256": source_hashes,
        "feature_extraction_seconds": extraction_seconds,
        "feature_names": FEATURE_NAMES,
        "feature_configuration": {"slices_per_plane": 9, "normalization_percentiles": [1, 99],
                                  "histogram_bins": 32, "gradient_threshold": 0.1,
                                  "slice_aggregation": ["mean", "std"], "planes": PLANES},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "limitations": ["Handcrafted slice statistics do not localize or diagnose tears.",
                        "Validation is a single held-out split; no confidence intervals or external test set.",
                        "Class prevalence differs between training and validation.",
                        "Nine slices per plane may miss focal abnormalities; planes are concatenated, not registered.",
                        "Per-slice intensity normalization discards absolute signal scale.",
                        "Examination ID separation does not establish patient-level independence.",
                        "Fixed balanced class weights; probability calibration has not been assessed."]})
    (output_dir / "report.json").write_text(json.dumps(result.report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result.report, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
