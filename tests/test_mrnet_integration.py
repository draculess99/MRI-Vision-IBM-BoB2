"""Optional read-only integration check; never required for CI."""

from pathlib import Path

import pytest

from mri_core.dataset_discovery import DEFAULT_MRNET_ROOT
from mri_core.labels import TARGETS, load_labels
from mri_core.manifest import PLANES, build_manifest


@pytest.mark.mrnet
def test_local_mrnet_labels_map_to_all_planes():
    root = Path(DEFAULT_MRNET_ROOT)
    label_dir = root / "labels"
    if not label_dir.is_dir():
        pytest.skip("Local MRNet label directory is unavailable")
    labels = load_labels(label_dir, recover_first_row=True)
    manifest = build_manifest(labels, root)
    assert len(manifest.rows) == 1250
    assert sum(row["split"] == "train" for row in manifest.rows) == 1130
    assert sum(row["split"] == "valid" for row in manifest.rows) == 120
    assert not manifest.missing_images and not manifest.unlabeled_images
    assert all(row[f"{plane}_available"] for row in manifest.rows for plane in PLANES)
    for recovery in labels.recoveries:
        assert recovery["row"] == 1
        assert recovery["original"] == ["_" + value for value in recovery["recovered"]]
    assert all(row[target] in (0, 1) for row in manifest.rows for target in TARGETS)
