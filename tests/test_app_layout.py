"""Presentation-only checks for the 2x2 image grid: display copies are ~280 px tall, aspect-preserving, and
never alter the arrays used for processing; the grid stays a strict 2x2 with the four original labels."""

import ast
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("streamlit")

from mri_core.pipeline import process_mri_image
from mri_core.rsna_integration import load_rsna_study_series

from .test_app_rsna_smoke_ui import APP, RSNA_ROOT, SMOKE_REL, make_checkpoint, ready, run_app  # noqa: F401


def load_helper():
    """Exec only the display helper and its height constant straight from app.py (Streamlit code can't be imported)."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    wanted = [node for node in tree.body
              if (isinstance(node, ast.FunctionDef) and node.name == "fit_panel_height")
              or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "IMAGE_PANEL_HEIGHT" for t in node.targets))]
    assert len(wanted) == 2
    namespace = {"cv2": cv2, "np": np}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(APP), "exec"), namespace)
    return namespace


HELPER = load_helper()
fit = HELPER["fit_panel_height"]


# ------------------------------------------------------------------------------ display-copy helper

def test_panel_height_target_is_in_the_requested_range():
    assert 260 <= HELPER["IMAGE_PANEL_HEIGHT"] <= 300


@pytest.mark.parametrize("shape", [(512, 512), (384, 512), (640, 320), (100, 80), (280, 300)])
def test_display_copy_has_target_height_and_preserves_aspect_ratio(shape):
    image = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    panel = fit(image)
    assert panel.shape[0] == HELPER["IMAGE_PANEL_HEIGHT"]
    assert panel.shape[1] == round(shape[1] * HELPER["IMAGE_PANEL_HEIGHT"] / shape[0])
    assert abs(panel.shape[1] / panel.shape[0] - shape[1] / shape[0]) < 0.01


def test_display_resize_never_touches_the_source_array():
    image = np.random.default_rng(1).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    original = image.copy()
    panel = fit(image)
    assert np.array_equal(image, original) and image.shape == (512, 512, 3)
    assert panel is not image and not np.shares_memory(panel, image)
    panel[:] = 0
    assert np.array_equal(image, original)


def test_colour_channels_and_dtype_are_kept():
    color = np.zeros((300, 300, 3), dtype=np.uint8)
    color[..., 2] = 200
    panel = fit(color)
    assert panel.shape[2] == 3 and panel.dtype == np.uint8 and panel[..., 2].min() == 200


def test_binary_mask_stays_binary_with_nearest_interpolation():
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:400, 150:350] = 255
    assert set(np.unique(fit(mask, nearest=True))) == {0, 255}


def test_negative_stride_view_such_as_bgr_to_rgb_is_supported():
    bgr = np.random.default_rng(2).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    rgb_view = bgr[..., ::-1]
    assert fit(rgb_view).shape == (280, 280, 3)
    assert np.array_equal(rgb_view, bgr[..., ::-1])


# ------------------------------------------------------------------------------ grid structure in the running app

def grid_columns(at):
    """(parent-id, [markdown heading, image count]) for every column under the main area, in render order."""
    found = []

    def walk(block, parent):
        for child in block.children.values():
            if child.type == "column":
                headings = [m.value for m in child.children.values() if m.type == "markdown"]
                images = [m for m in child.children.values() if m.type == "image"]
                found.append((id(parent), headings, len(images)))
            elif hasattr(child, "children"):
                walk(child, child)

    walk(at.main, at.main)
    return found


def test_grid_is_a_strict_two_by_two_with_the_original_labels(ready, tmp_path, monkeypatch):  # noqa: F811
    _, study_uid = ready
    at = run_app(monkeypatch, tmp_path, study_uid)
    assert not at.exception
    assert len(at.get("image")) == 4

    grid = [entry for entry in grid_columns(at) if entry[2] == 1]
    assert [headings for _, headings, _ in grid] == [
        ["##### Original"], ["##### Preprocessed"], ["##### Segmentation Mask"], ["##### Overlay"],
    ]
    rows = [parent for parent, _, _ in grid]
    assert rows[0] == rows[1] and rows[2] == rows[3] and rows[0] != rows[2]  # two rows of two panels


def test_layout_css_is_scoped_to_the_image_grid_only(ready, tmp_path, monkeypatch):  # noqa: F811
    _, study_uid = ready
    at = run_app(monkeypatch, tmp_path, study_uid)
    styles = [h.proto.body for h in at.get("html")]
    assert styles and all(".st-key-image_grid" in body for body in styles)


def test_other_sections_still_render(ready, tmp_path, monkeypatch):  # noqa: F811
    _, study_uid = ready
    at = run_app(monkeypatch, tmp_path, study_uid)
    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert "Extracted Image Features" in subheaders and "Quality Assessment Report" in subheaders
    assert any(m.label == "Quality Status" for m in at.metric)


def test_processing_results_are_identical_to_a_direct_pipeline_call(ready, tmp_path, monkeypatch):  # noqa: F811
    """The resize is display-only: the features the app shows equal those from process_mri_image on the same slice."""
    metadata, study_uid = ready
    at = run_app(monkeypatch, tmp_path, study_uid)
    shown = next(d.value for d in at.dataframe if list(d.value.columns) == ["Feature", "Value"] and "mean_intensity" in set(d.value["Feature"]))

    plane = next(s for s in at.sidebar.selectbox if s.label == "Select Plane").value
    volume, _ = load_rsna_study_series(RSNA_ROOT, metadata, study_uid, plane)
    direct = process_mri_image(volume.get_display_slice(volume.default_slice_index), segmentation_method="Otsu",
                               is_mri=True, is_inverted=volume.is_inverted)["features"]
    for feature, value in direct.items():
        if isinstance(value, float):
            row = shown[shown["Feature"] == feature]["Value"].iloc[0]
            assert float(row) == pytest.approx(round(value, 2)), feature
