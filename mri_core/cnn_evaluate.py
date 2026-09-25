"""ID-aligned validation evaluation using unchanged V0.4 metric semantics."""

import numpy as np

from .baseline import classification_metrics, distribution
from .labels import TARGETS

# Published rounded reference values, not fabricated per-examination predictions.
V04_REFERENCE = {
    "abnormal": {"roc_auc": 0.8552, "average_precision": 0.9365, "accuracy": 0.8167, "f1": 0.8778},
    "acl": {"roc_auc": 0.8171, "average_precision": 0.7877, "accuracy": 0.7417, "f1": 0.7156},
    "meniscus": {"roc_auc": 0.7240, "average_precision": 0.6239, "accuracy": 0.6667, "f1": 0.6610},
}


def evaluate_predictions(rows, predictions, *, compare_reference=True):
    valid = {row["exam_id"]: row for row in rows if row["split"] == "valid"}
    train = [row for row in rows if row["split"] == "train"]
    if len({row["exam_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate IDs or split overlap")
    ids = [record["exam_id"] for record in predictions]
    if not train or not valid or len(set(ids)) != len(ids) or set(ids) != set(valid):
        raise ValueError("Predictions must match validation IDs exactly, without duplicates")
    ordered = sorted(predictions, key=lambda record: record["exam_id"])
    report = {}
    for target in TARGETS:
        scores = np.array([record[target] for record in ordered], dtype=float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Predictions must be finite probabilities")
        labels = np.array([valid[record["exam_id"]][target] for record in ordered])
        training = np.array([row[target] for row in train])
        metrics = classification_metrics(labels, scores)
        dummy = classification_metrics(labels, np.full(len(labels), training.mean()))
        report[target] = {"train": distribution(training), "valid": distribution(labels),
                          "cnn": metrics, "dummy": dummy}
        if compare_reference:
            reference = dict(V04_REFERENCE[target])
            report[target]["v04_reference_rounded"] = reference
            report[target]["delta_vs_v04_rounded"] = {
                metric: None if metrics[metric] is None else metrics[metric] - value
                for metric, value in reference.items()}
    return {"threshold": 0.5, "confusion_matrix_layout": "[[TN, FP], [FN, TP]]",
            "reference_note": "V0.4 reference metrics are rounded to four decimals; no paired significance claim.",
            "targets": report}


def summarize_seeds(reports):
    """Summarize every predefined seed; do not select a best validation seed."""
    summary = {}
    for target in TARGETS:
        summary[target] = {}
        for metric in ("roc_auc", "average_precision", "accuracy", "precision", "recall", "f1"):
            values = [report["targets"][target]["cnn"][metric] for report in reports]
            valid = [value for value in values if value is not None]
            summary[target][metric] = {"values": values, "mean": float(np.mean(valid)) if valid else None,
                                       "std": float(np.std(valid)) if valid else None}
    return summary
