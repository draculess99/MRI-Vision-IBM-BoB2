import csv

import pytest

from mri_core.labels import SPLITS, TARGETS, load_label_table, load_labels


@pytest.fixture
def label_dir(tmp_path):
    directory = tmp_path / "labels"
    directory.mkdir()
    for split, ids in (("train", range(8)), ("valid", range(8, 12))):
        for offset, target in enumerate(TARGETS):
            rows = [(f"{i:04d}", (i + offset) % 2) for i in ids]
            # Deliberately different row order for each task.
            rows = rows[offset:] + rows[:offset]
            with (directory / f"{split}_{target}.csv").open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)
    return directory


def test_join_preserves_ids_and_uses_keys_not_row_order(label_dir):
    labels = load_labels(label_dir)
    assert len(labels.records) == 12
    assert labels.recoveries == ()
    for row in labels.records:
        number = int(row["exam_id"])
        assert row["exam_id"] == f"{number:04d}"
        assert row["split"] == ("train" if number < 8 else "valid")
        assert [row[t] for t in TARGETS] == [number % 2, (number + 1) % 2, number % 2]


def test_known_anomaly_requires_explicit_recovery_and_is_reported(tmp_path):
    path = tmp_path / "train_abnormal.csv"
    original = b'"_0000","_1"\r\n0001,0\r\n'
    path.write_bytes(original)
    with pytest.raises(ValueError, match="expected four-digit"):
        load_label_table(path)
    labels, recoveries = load_label_table(path, recover_first_row=True)
    assert labels == {"0000": 1, "0001": 0}
    assert recoveries == [{"file": path.name, "row": 1,
                           "original": ["_0000", "_1"], "recovered": ["0000", "1"]}]
    assert path.read_bytes() == original


@pytest.mark.parametrize("contents", [
    "0000,2\n", "0000,-1\n", "0000,0.0\n", "0,1\n", " 0000,1\n",
    "exam_id,label\n", "0000,1,extra\n", "\n", "", "0000,1\n\n",
    "0000,1\n_0001,_0\n", "_0000,1\n", "0000,_1\n",
])
def test_malformed_rows_never_silently_discarded(tmp_path, contents):
    path = tmp_path / "labels.csv"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_label_table(path, recover_first_row=True)


@pytest.mark.parametrize("contents", ["0000,1\n0000,1\n", "0000,1\n0000,0\n", "_0000,_1\n0000,1\n"])
def test_duplicate_ids_rejected_including_recovered_row(tmp_path, contents):
    path = tmp_path / "labels.csv"
    path.write_text(contents)
    with pytest.raises(ValueError, match="duplicate"):
        load_label_table(path, recover_first_row=True)


def test_missing_task_file(label_dir):
    (label_dir / "train_acl.csv").unlink()
    with pytest.raises(FileNotFoundError):
        load_labels(label_dir)


def test_task_id_mismatch(label_dir):
    path = label_dir / "valid_meniscus.csv"
    path.write_text(path.read_text().replace("0008", "9999"))
    with pytest.raises(ValueError, match="task ID mismatch"):
        load_labels(label_dir)


def test_split_overlap(label_dir):
    for target in TARGETS:
        path = label_dir / f"valid_{target}.csv"
        path.write_text(path.read_text().replace("0008", "0000"))
    with pytest.raises(ValueError, match="overlap"):
        load_labels(label_dir)


def test_all_six_recoveries_retained(label_dir):
    for split in SPLITS:
        for target in TARGETS:
            path = label_dir / f"{split}_{target}.csv"
            lines = path.read_text().splitlines()
            exam_id, label = lines[0].split(",")
            lines[0] = f'"_{exam_id}","_{label}"'
            path.write_text("\n".join(lines) + "\n")
    loaded = load_labels(label_dir, recover_first_row=True)
    assert len(loaded.records) == 12
    assert len(loaded.recoveries) == 6
