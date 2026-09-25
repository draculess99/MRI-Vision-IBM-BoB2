import pytest

from mri_core.labels import load_labels
from mri_core.manifest import PLANES, build_manifest
from tests.test_labels import label_dir


@pytest.fixture
def image_root(tmp_path):
    root = tmp_path / "MRNet_ Knee MRI's_files"
    for plane in PLANES:
        directory = root / plane
        directory.mkdir(parents=True)
        for i in range(12):
            (directory / f"{i:04d}.npy").touch()
    return root


def test_complete_manifest_and_external_paths(label_dir, image_root):
    manifest = build_manifest(load_labels(label_dir), image_root)
    assert len(manifest.rows) == 12
    assert len(manifest.paths) == 36
    assert not manifest.missing_images and not manifest.unlabeled_images
    assert sum(row["split"] == "train" for row in manifest.rows) == 8
    assert all(row[f"{plane}_available"] for row in manifest.rows for plane in PLANES)
    assert all(path.is_relative_to(image_root) for path in manifest.paths.values())


def test_missing_plane_and_unlabeled_exam_are_explicit(label_dir, image_root):
    (image_root / "axial" / "0000.npy").unlink()
    (image_root / "coronal" / "9999.npy").touch()
    manifest = build_manifest(load_labels(label_dir), image_root)
    assert manifest.missing_images == ({"exam_id": "0000", "plane": "axial"},)
    assert manifest.unlabeled_images == ({"exam_id": "9999", "plane": "coronal"},)
    assert manifest.rows[0]["axial_available"] is False
    assert manifest.rows[0]["abnormal"] == 0


def test_duplicate_image_mapping_rejected(label_dir, image_root):
    other = image_root / "nested" / "axial"
    other.mkdir(parents=True)
    (other / "0000.npy").touch()
    with pytest.raises(ValueError, match="Ambiguous"):
        build_manifest(load_labels(label_dir), image_root)


def test_split_folders_supported_without_inferencing_split(label_dir, tmp_path):
    root = tmp_path / "images"
    for folder, ids in (("train", range(8)), ("valid", range(8, 12))):
        for plane in PLANES:
            path = root / folder / plane
            path.mkdir(parents=True)
            for i in ids:
                (path / f"{i:04d}.npy").touch()
    manifest = build_manifest(load_labels(label_dir), root)
    assert len(manifest.paths) == 36
    assert not manifest.missing_images


def test_invalid_exam_filename_rejected(label_dir, image_root):
    (image_root / "axial" / "1.npy").touch()
    with pytest.raises(ValueError, match="Invalid examination"):
        build_manifest(load_labels(label_dir), image_root)
