"""Label-first examination manifests; source image paths remain external."""

from dataclasses import dataclass
from pathlib import Path

from .labels import EXAM_ID, LabelSet

PLANES = ("axial", "coronal", "sagittal")
MANIFEST_FIELDS = ("exam_id", "split", "axial_available", "coronal_available",
                   "sagittal_available", "abnormal", "acl", "meniscus")


@dataclass(frozen=True)
class Manifest:
    rows: tuple[dict, ...]
    paths: dict[tuple[str, str], Path]
    missing_images: tuple[dict, ...]
    unlabeled_images: tuple[dict, ...]


def build_manifest(labels: LabelSet, dataset_root: Path) -> Manifest:
    """Scan every plane directory, rejecting ambiguous duplicate ID/plane files.

    Split membership comes exclusively from validated label tables, never filenames
    or ID ranges. Missing and unlabeled images are returned explicitly for reporting.
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    paths = {}
    for path in sorted(root.rglob("*")):
        plane = path.parent.name.lower()
        if not path.is_file() or path.suffix.lower() != ".npy" or plane not in PLANES:
            continue
        if EXAM_ID.fullmatch(path.stem) is None:
            raise ValueError(f"Invalid examination filename: {path}")
        key = (path.stem, plane)
        if key in paths:
            raise ValueError(f"Ambiguous examination {key}: {paths[key]} and {path}")
        paths[key] = path
    rows, missing = [], []
    for record in labels.records:
        row = dict(record)
        for plane in PLANES:
            available = (record["exam_id"], plane) in paths
            row[f"{plane}_available"] = available
            if not available:
                missing.append({"exam_id": record["exam_id"], "plane": plane})
        rows.append(row)
    labeled_ids = {row["exam_id"] for row in rows}
    unlabeled = [{"exam_id": exam_id, "plane": plane}
                 for exam_id, plane in sorted(paths) if exam_id not in labeled_ids]
    return Manifest(tuple(rows), paths, tuple(missing), tuple(unlabeled))
