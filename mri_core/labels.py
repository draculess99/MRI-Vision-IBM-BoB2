"""Strict MRNet label parsing without changing the source tables."""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

TARGETS = ("abnormal", "acl", "meniscus")
SPLITS = ("train", "valid")
EXAM_ID = re.compile(r"[0-9]{4}")


@dataclass(frozen=True)
class LabelSet:
    records: tuple[dict, ...]
    recoveries: tuple[dict, ...]


def load_label_table(path: Path, *, recover_first_row: bool = False) -> tuple[dict[str, int], list[dict]]:
    """Read headerless ID,label rows; optionally recover the known underscore anomaly.

    Recovery applies only to row 1 with exactly _dddd,_0 or _dddd,_1.
    All other malformed rows, including blank rows and headers, are errors.
    """
    path = Path(path)
    labels, recoveries = {}, []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.reader(stream, strict=True), start=1):
            original = list(row)
            if (recover_first_row and line == 1 and len(row) == 2
                    and re.fullmatch(r"_[0-9]{4}", row[0])
                    and row[1] in ("_0", "_1")):
                row = [row[0][1:], row[1][1:]]
                recoveries.append({"file": path.name, "row": line,
                                   "original": original, "recovered": row.copy()})
            if len(row) != 2 or EXAM_ID.fullmatch(row[0]) is None or row[1] not in ("0", "1"):
                raise ValueError(f"{path.name}:{line}: expected four-digit ID and binary 0/1 label; got {original!r}")
            exam_id, value = row
            if exam_id in labels:
                raise ValueError(f"{path.name}:{line}: duplicate examination ID {exam_id}")
            labels[exam_id] = int(value)
    if not labels:
        raise ValueError(f"{path.name}: empty label table")
    return labels, recoveries


def load_labels(label_dir: Path, *, recover_first_row: bool = False) -> LabelSet:
    """Join all six tables by ID and reject task mismatch or split overlap."""
    tables, recoveries, records = {}, [], []
    for split in SPLITS:
        tables[split] = {}
        for target in TARGETS:
            path = Path(label_dir) / f"{split}_{target}.csv"
            table, recovered = load_label_table(path, recover_first_row=recover_first_row)
            tables[split][target] = table
            recoveries.extend(recovered)
        reference = set(tables[split][TARGETS[0]])
        for target in TARGETS[1:]:
            actual = set(tables[split][target])
            if actual != reference:
                raise ValueError(f"{split}/{target}: task ID mismatch; missing={sorted(reference - actual)}, extra={sorted(actual - reference)}")
        for exam_id in sorted(reference):
            records.append({"exam_id": exam_id, "split": split,
                            **{target: tables[split][target][exam_id] for target in TARGETS}})
    overlap = set(tables["train"][TARGETS[0]]) & set(tables["valid"][TARGETS[0]])
    if overlap:
        raise ValueError(f"Train/validation ID overlap: {sorted(overlap)}")
    return LabelSet(tuple(records), tuple(recoveries))
