import numpy as np
import pytest

from mri_core.baseline_features import (FEATURE_NAMES, extract_features, sampled_indices,
                                       slice_features, volume_features)
from mri_core.labels import load_labels
from mri_core.manifest import PLANES, build_manifest
from tests.test_labels import label_dir


def test_slice_sampling_endpoints_short_volumes_and_no_duplicates():
    assert sampled_indices(44).tolist() == [0, 5, 10, 16, 21, 26, 32, 37, 43]
    assert sampled_indices(3).tolist() == [0, 1, 2]
    assert sampled_indices(1).tolist() == [0]
    with pytest.raises(ValueError):
        sampled_indices(0)


def test_constant_and_edge_slice_statistics():
    assert np.allclose(slice_features(np.ones((8, 8))), 0)
    image = np.zeros((8, 8))
    image[:, 4:] = 1
    features = slice_features(image)
    assert features[0] == 0.5
    assert features[1] == 0.5
    assert features[5] == 1.0  # Two equally likely intensities: entropy = 1 bit.
    assert features[6] > 0
    assert features[9] == 0.25
    assert np.allclose(features, slice_features(image * 100 + 20))


@pytest.mark.parametrize("image", [np.full((8, 8), np.nan), np.full((8, 8), np.inf), np.zeros((1, 8))])
def test_invalid_slice_rejected(image):
    with pytest.raises(ValueError):
        slice_features(image)


def test_volume_aggregation_is_deterministic_and_read_only(tmp_path):
    path = tmp_path / "0000.npy"
    image = np.zeros((8, 8), dtype=np.uint8)
    image[:, 4:] = 255
    np.save(path, np.stack([image, image]))
    original = path.read_bytes()
    actual = volume_features(path)
    assert actual.shape == (20,)
    assert np.allclose(actual[::2], slice_features(image))
    assert np.allclose(actual[1::2], 0)
    assert np.array_equal(actual, volume_features(path))
    assert path.read_bytes() == original


@pytest.mark.parametrize("array", [np.zeros((8, 8)), np.zeros((0, 8, 8)), np.zeros((2, 1, 8)),
                                  np.zeros((2, 8, 8, 1)), np.array([{'unsafe': True}], dtype=object)])
def test_invalid_or_pickled_volume_rejected(tmp_path, array):
    path = tmp_path / "0000.npy"
    np.save(path, array)
    with pytest.raises(ValueError):
        volume_features(path)


def test_exam_features_plane_order_and_missing_images(label_dir, tmp_path):
    root = tmp_path / "images"
    for plane_index, plane in enumerate(PLANES):
        directory = root / plane
        directory.mkdir(parents=True)
        for i in range(12):
            image = np.zeros((2, 8, 8), dtype=np.uint8)
            image[:, :, plane_index + 2:] = 255
            np.save(directory / f"{i:04d}.npy", image)
    manifest = build_manifest(load_labels(label_dir), root)
    matrix = extract_features(manifest)
    assert matrix.shape == (12, len(FEATURE_NAMES)) == (12, 60)
    assert np.isfinite(matrix).all()
    for index, plane in enumerate(PLANES):
        assert np.allclose(matrix[0, index*20:(index+1)*20], volume_features(root / plane / "0000.npy"))
    (root / "axial" / "0000.npy").unlink()
    with pytest.raises(ValueError, match="missing images"):
        extract_features(build_manifest(load_labels(label_dir), root))
