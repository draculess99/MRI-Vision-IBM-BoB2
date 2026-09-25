import numpy as np
import pytest

from mri_core.baseline import classification_metrics
from mri_core.cnn_evaluate import V04_REFERENCE, evaluate_predictions, summarize_seeds
from mri_core.labels import TARGETS


@pytest.fixture
def records():
    return [{"exam_id": f"{i:04d}", "split": "train" if i < 4 else "valid",
             **{target: i % 2 for target in TARGETS}} for i in range(8)]


def predictions():
    return [{"exam_id": f"{i:04d}", **{target: score for target in TARGETS}}
            for i, score in zip((7, 4, 6, 5), (0.9, 0.1, 0.6, 0.4))]


def test_metric_parity_id_alignment_and_rounded_reference(records):
    report = evaluate_predictions(records, predictions())
    expected = classification_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    for target in TARGETS:
        assert report["targets"][target]["cnn"] == expected
        assert report["targets"][target]["dummy"]["roc_auc"] == 0.5
        for metric, reference in V04_REFERENCE[target].items():
            assert report["targets"][target]["delta_vs_v04_rounded"][metric] == pytest.approx(expected[metric] - reference)
    assert V04_REFERENCE["abnormal"]["roc_auc"] == 0.8552
    assert V04_REFERENCE["acl"]["average_precision"] == 0.7877
    assert V04_REFERENCE["meniscus"]["accuracy"] == 0.6667


@pytest.mark.parametrize("problem", ["duplicate", "missing", "train_id", "nan", "out_of_range"])
def test_prediction_integrity_rejected(records, problem):
    values = predictions()
    if problem == "duplicate":
        values.append(dict(values[0]))
    elif problem == "missing":
        values.pop()
    elif problem == "train_id":
        values[0]["exam_id"] = "0000"
    elif problem == "nan":
        values[0]["acl"] = float("nan")
    else:
        values[0]["acl"] = 1.01
    with pytest.raises(ValueError):
        evaluate_predictions(records, values)


def test_synthetic_run_does_not_claim_real_reference_comparison(records):
    report = evaluate_predictions(records, predictions(), compare_reference=False)
    assert "v04_reference_rounded" not in report["targets"]["acl"]
    summary = summarize_seeds([report, report])
    assert summary["acl"]["roc_auc"]["std"] == 0
    assert len(summary["acl"]["roc_auc"]["values"]) == 2
