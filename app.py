from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import cv2

from mri_core.loader import load_mri
from mri_core.pipeline import process_mri_image
from mri_core.dataset_discovery import discover_mrnet_root, list_mrnet_exams, load_mrnet_exam
from mri_core.rsna_integration import (
    discover_rsna_root, load_rsna_metadata_safe, discover_available_studies,
    get_available_planes, load_rsna_study_series, RSNADiscoveryError, RSNAStudyNotAvailable
)
from mri_core.decision import generate_decision_report
from mri_core.rsna_knee_inference import run_inference, RSNAInferenceError
from mri_core.rsna_knee_ensemble import run_ensemble_inference
from mri_core.comparison import generate_comparison_report, ComparisonStatus

# Checkpoint location matches configs/rsna_knee.json's "checkpoint_path". No checkpoint
# is expected to exist yet; the RSNA viewer remains fully usable without one.
RSNA_CHECKPOINT_PATH = Path("outputs/rsna-knee/checkpoint_best.pt")
# The smoke checkpoint is used only when the user explicitly enables it; it never replaces RSNA_CHECKPOINT_PATH.
RSNA_SMOKE_CHECKPOINT_PATH = Path("outputs/rsna-knee-smoke/checkpoint_smoke.pt")
RSNA_SMOKE_WARNING = ("Experimental smoke-test model — trained on only 3 studies; "
                      "outputs are not clinically meaningful and are not a diagnosis.")
# 5-fold ensemble checkpoints (all must exist for ensemble to run)
RSNA_ENSEMBLE_CHECKPOINT_PATHS = [
    Path(f"outputs/rsna-knee-ensemble/checkpoint_rsna_stratified_fold{i}.pt")
    for i in range(1, 6)
]
RSNA_ENSEMBLE_WARNING = ("Experimental 5-fold ensemble — trained on a limited labeled dataset; "
                         "outputs are not clinically validated and are not a diagnosis.")

IMAGE_PANEL_HEIGHT = 280  # display-only height (px) of each panel in the 2x2 image grid


def fit_panel_height(image, height=IMAGE_PANEL_HEIGHT, nearest=False):
    """Aspect-preserving copy scaled to `height` px for display only; the input array is never modified."""
    rows, cols = image.shape[:2]
    if rows == 0 or cols == 0:
        return image
    scale = height / rows
    if nearest:
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(np.ascontiguousarray(image), (max(1, round(cols * scale)), height), interpolation=interpolation)


st.set_page_config(page_title="MRI Vision Core", layout="wide")

st.title("MRI Vision Core")
st.subheader("Medical MRI & OpenCV Image Processing & Feature Exploration")

st.warning("Research and educational prototype. Not for medical diagnosis or clinical decision-making.")

st.sidebar.header("Settings")

# Detect if local RSNA/MRNet datasets are available
mrnet_root = discover_mrnet_root()
try:
    rsna_root = discover_rsna_root()
    has_rsna = True
except RSNADiscoveryError:
    rsna_root = None
    has_rsna = False

# Build data source options
data_sources = ["Upload File"]
if has_rsna:
    data_sources.append("Explore RSNA Studies")
if mrnet_root:
    data_sources.append("Explore MRNet Dataset")

data_source = st.sidebar.radio("Data Source", data_sources, horizontal=False)

volume = None
status_msg = ""
decision_report = None
study_metadata = {}
use_ensemble_inference = False
use_smoke_checkpoint = False
# Prior-study state (RSNA comparison only)
prior_volume = None
prior_study_metadata = {}
prior_results = None
prior_decision_report = None
selected_prior_study = None

if data_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image / MRI Scan",
        type=["png", "jpg", "jpeg", "dcm", "nii", "gz", "npy"],
        help="Supported formats: PNG, JPG, JPEG, DICOM (.dcm), NIfTI (.nii, .nii.gz), NumPy (.npy)"
    )
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            volume = load_mri(file_bytes, filename=uploaded_file.name)
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
    else:
        status_msg = "Upload an image or MRI file (PNG, JPG, DICOM, NIfTI, NPY) and click 'Process Image' to begin."

elif data_source == "Explore RSNA Studies":
    try:
        metadata = load_rsna_metadata_safe(rsna_root)
        use_ensemble_inference = st.sidebar.checkbox(
            "Use 5-fold ensemble predictions (experimental)",
            value=False,
            key="rsna_use_ensemble_inference",
            help="Uses all 5 folds if available. Off by default.",
        )
        use_smoke_checkpoint = st.sidebar.checkbox(
            "Use experimental smoke-test checkpoint",
            value=False,
            key="rsna_use_smoke_checkpoint",
            help=f"Loads {RSNA_SMOKE_CHECKPOINT_PATH.as_posix()} if it exists. Off by default.",
        )
        available_studies = discover_available_studies(rsna_root, metadata)

        if not available_studies:
            st.sidebar.warning("No downloaded RSNA studies found. Download in progress?")
        else:
            st.sidebar.caption(f"Found {len(available_studies)} downloaded studies")
            selected_study = st.sidebar.selectbox("Select Study", available_studies)
            available_planes = get_available_planes(rsna_root, metadata, selected_study)

            if available_planes:
                selected_plane = st.sidebar.selectbox("Select Plane", available_planes)

                try:
                    volume, meta = load_rsna_study_series(
                        rsna_root, metadata, selected_study, selected_plane
                    )
                    study_metadata = meta
                    st.sidebar.success(f"Loaded {selected_plane} series")
                except RSNAStudyNotAvailable as e:
                    st.sidebar.warning(str(e))
                except Exception as e:
                    st.sidebar.error(f"Error loading series: {e}")

            else:
                st.sidebar.info(f"No planes available yet for study {selected_study}. Download in progress?")

            # ---- Prior Study selector (ChangeGuard) -------------------------
            # Exclude the currently selected study so a study cannot be compared
            # against itself.
            prior_options = ["— none —"] + [s for s in available_studies if s != selected_study]
            st.sidebar.divider()
            st.sidebar.markdown("**ChangeGuard — prior study (optional)**")
            selected_prior_study = st.sidebar.selectbox(
                "Select Prior Study",
                prior_options,
                index=0,
                key="rsna_prior_study",
                help="Select a prior study to compare image-derived features. Non-diagnostic.",
            )

            if selected_prior_study != "— none —" and available_planes:
                try:
                    prior_volume, prior_meta = load_rsna_study_series(
                        rsna_root, metadata, selected_prior_study, selected_plane
                    )
                    prior_study_metadata = prior_meta
                    st.sidebar.success(f"Prior study loaded ({selected_plane})")
                except RSNAStudyNotAvailable as e:
                    st.sidebar.warning(f"Prior study unavailable: {e}")
                except Exception as e:
                    st.sidebar.error(f"Error loading prior study: {e}")
    except RSNADiscoveryError as e:
        st.sidebar.error(f"RSNA discovery failed: {e}")
    except Exception as e:
        st.sidebar.error(f"Error loading RSNA metadata: {e}")

elif data_source == "Explore MRNet Dataset":
    st.sidebar.caption(f"Discovered MRNet at: `{mrnet_root.name}`")
    plane = st.sidebar.selectbox("Select Imaging Plane", ["axial", "coronal", "sagittal"])
    exams = list_mrnet_exams(plane=plane)
    if exams:
        exam_options = [e.name for e in exams[:200]]
        selected_exam = st.sidebar.selectbox("Select Examination", exam_options, index=0)
        try:
            volume = load_mrnet_exam(selected_exam, plane=plane)
        except Exception as e:
            st.error(f"Error loading exam '{selected_exam}': {e}")
    else:
        st.sidebar.warning(f"No exam files found for plane '{plane}'.")

seg_method = st.sidebar.selectbox("Select Segmentation Method", ["Otsu", "Adaptive"])

if volume is not None:
    try:
        # Display volume details
        st.sidebar.markdown(f"**Format:** `{volume.format_type}`")
        if volume.num_slices > 1:
            st.sidebar.info(f"Multi-slice volume: **{volume.num_slices}** slices")
            slice_idx = st.sidebar.slider(
                "Select Slice",
                min_value=0,
                max_value=volume.num_slices - 1,
                value=volume.default_slice_index,
                help=f"Default middle slice: {volume.default_slice_index}"
            )
            st.sidebar.caption(f"Viewing Slice {slice_idx + 1} of {volume.num_slices} (Index: {slice_idx})")
        else:
            slice_idx = 0
            
        # Non-identifying metadata expander
        if volume.metadata:
            with st.sidebar.expander("Technical Imaging Metadata", expanded=False):
                meta_df = pd.DataFrame(
                    [{"Property": k, "Value": str(v)} for k, v in volume.metadata.items()]
                )
                st.dataframe(meta_df, use_container_width=True, hide_index=True)
                
        # Retrieve display slice for selected index
        display_slice = volume.get_display_slice(slice_idx)
        
        # Action button
        process_btn = st.sidebar.button("Process Image")
        
        if process_btn:
            is_mri = (volume.format_type in ("DICOM", "NIFTI", "NUMPY"))
            results = process_mri_image(
                display_slice,
                segmentation_method=seg_method,
                is_mri=is_mri,
                is_inverted=volume.is_inverted
            )

            # 2x2 grid of compact display copies (~IMAGE_PANEL_HEIGHT px tall); processing arrays stay untouched.
            st.html("<style>.st-key-image_grid [data-testid='stVerticalBlock'] { gap: 0.4rem; }</style>")
            with st.container(key="image_grid"):
                col1, col2 = st.columns(2)
                with col1:
                    orig = results["original"]
                    panel = fit_panel_height(orig)
                    st.markdown("##### Original")
                    st.image(panel, channels="BGR" if len(orig.shape) == 3 else "GRAY", width=panel.shape[1])
                with col2:
                    panel = fit_panel_height(results["preprocessed"])
                    st.markdown("##### Preprocessed")
                    st.image(panel, clamp=True, width=panel.shape[1])

                col3, col4 = st.columns(2)
                with col3:
                    panel = fit_panel_height(results["mask"], nearest=True)
                    st.markdown("##### Segmentation Mask")
                    st.image(panel, clamp=True, width=panel.shape[1])
                with col4:
                    overlay_rgb = results["overlay"][..., ::-1] if len(results["overlay"].shape) == 3 else results["overlay"]
                    panel = fit_panel_height(overlay_rgb)
                    st.markdown("##### Overlay")
                    st.image(panel, width=panel.shape[1])

            st.divider()
            st.subheader("Extracted Image Features")

            features = results["features"]
            formatted_features = {k: round(v, 2) if isinstance(v, float) else v for k, v in features.items()}
            df = pd.DataFrame(list(formatted_features.items()), columns=["Feature", "Value"])
            st.dataframe(df, use_container_width=True)

            # Generate decision report if RSNA study
            if data_source == "Explore RSNA Studies" and study_metadata:
                st.divider()
                st.subheader("Quality Assessment Report")

                model_predictions = None

                # Inference logic: ensemble > smoke > production, in that priority order
                if use_ensemble_inference:
                    # Check if all 5 folds exist
                    all_folds_exist = all(p.is_file() for p in RSNA_ENSEMBLE_CHECKPOINT_PATHS)
                    if all_folds_exist:
                        try:
                            model_predictions = run_ensemble_inference(
                                metadata=metadata,
                                fold_checkpoint_paths=RSNA_ENSEMBLE_CHECKPOINT_PATHS,
                                study_uid=study_metadata["study_uid"],
                            )
                        except RSNAInferenceError as e:
                            st.warning(f"Ensemble inference unavailable for this study: {e}")
                    else:
                        missing_folds = [i for i, p in enumerate(RSNA_ENSEMBLE_CHECKPOINT_PATHS, start=1) if not p.is_file()]
                        st.info(f"5-fold ensemble not available (missing folds: {', '.join(map(str, missing_folds))}); "
                                "model predictions are unavailable.")
                else:
                    # Use existing smoke or production checkpoint
                    active_checkpoint_path = RSNA_SMOKE_CHECKPOINT_PATH if use_smoke_checkpoint else RSNA_CHECKPOINT_PATH
                    if active_checkpoint_path.is_file():
                        try:
                            model_predictions = run_inference(
                                metadata=metadata,
                                checkpoint_path=active_checkpoint_path,
                                study_uid=study_metadata["study_uid"],
                            )
                        except RSNAInferenceError as e:
                            st.warning(f"Model inference unavailable for this study: {e}")
                    elif use_smoke_checkpoint:
                        st.info(f"Smoke-test checkpoint not found at `{RSNA_SMOKE_CHECKPOINT_PATH.as_posix()}`; "
                                "model predictions are unavailable.")

                try:
                    decision_report = generate_decision_report(
                        study_uid=study_metadata["study_uid"],
                        series_uid=study_metadata["series_uid"],
                        plane=study_metadata["plane"],
                        volume_data=volume.raw_data,
                        preprocessed_data=results["preprocessed"],
                        mask=results["mask"],
                        features=features,
                        num_slices=study_metadata["num_slices"],
                        model_predictions=model_predictions,
                    )

                    # Display report summary
                    col_left, col_right = st.columns([1, 2])
                    with col_left:
                        status_color = {
                            "OK": "🟢",
                            "REVIEW": "🟡",
                            "INVALID": "🔴"
                        }
                        status_text = decision_report.quality_status.value
                        st.metric("Quality Status", f"{status_color[status_text]} {status_text}")

                    with col_right:
                        st.write("**Study Information**")
                        st.write(f"Study UID: `{decision_report.study_uid}`")
                        st.write(f"Plane: {decision_report.plane}")
                        st.write(f"Volume: {decision_report.num_slices} slices, {decision_report.image_height}×{decision_report.image_width}px")

                    # Quality metrics
                    st.write("**Image Quality Metrics**")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("Mean Intensity", f"{decision_report.quality_metrics.mean_intensity:.1f}")
                        st.metric("Std Deviation", f"{decision_report.quality_metrics.std_intensity:.1f}")
                    with metrics_col2:
                        st.metric("Intensity Range", f"{decision_report.quality_metrics.intensity_range:.1f}")
                        st.metric("Foreground %", f"{decision_report.quality_metrics.foreground_fraction*100:.1f}%")
                    with metrics_col3:
                        st.metric("Foreground Pixels", f"{decision_report.segmentation_quality.foreground_pixels:,}")
                        st.metric("Contours Found", "Yes" if decision_report.segmentation_quality.has_contours else "No")

                    # Quality flags
                    if decision_report.quality_flags:
                        st.warning(f"**Quality Flags:** {'; '.join(decision_report.quality_flags)}")

                    # Model status
                    st.info(f"**Model Status:** {decision_report.model_status}")

                    if decision_report.model_predictions is not None:
                        if use_ensemble_inference:
                            st.warning(RSNA_ENSEMBLE_WARNING)
                        elif use_smoke_checkpoint:
                            st.warning(RSNA_SMOKE_WARNING)
                        st.caption("Model outputs are experimental probabilities and are not a clinical diagnosis.")
                        predictions_df = pd.DataFrame(
                            list(decision_report.model_predictions.items()),
                            columns=["Target", "Probability"],
                        )
                        st.dataframe(predictions_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Error generating decision report: {e}")

            # ---- ChangeGuard comparison section -----------------------------
            if (
                selected_prior_study is not None
                and selected_prior_study != "— none —"
                and prior_volume is not None
                and decision_report is not None
            ):
                st.divider()
                with st.expander(
                    "ChangeGuard comparison — non-diagnostic; clinician review required",
                    expanded=True,
                ):
                    st.warning(
                        "This section shows image-derived feature changes between two studies. "
                        "It does not constitute a clinical finding, diagnosis, or medical opinion. "
                        "Clinician review is required before any clinical action is taken."
                    )
                    try:
                        # Process the prior study through the same pipeline.
                        prior_is_mri = (prior_volume.format_type in ("DICOM", "NIFTI", "NUMPY"))
                        prior_display_slice = prior_volume.get_display_slice(
                            prior_volume.default_slice_index
                        )
                        prior_results = process_mri_image(
                            prior_display_slice,
                            segmentation_method=seg_method,
                            is_mri=prior_is_mri,
                            is_inverted=prior_volume.is_inverted,
                        )
                        prior_decision_report = generate_decision_report(
                            study_uid=prior_study_metadata["study_uid"],
                            series_uid=prior_study_metadata["series_uid"],
                            plane=prior_study_metadata["plane"],
                            volume_data=prior_volume.raw_data,
                            preprocessed_data=prior_results["preprocessed"],
                            mask=prior_results["mask"],
                            features=prior_results["features"],
                            num_slices=prior_study_metadata["num_slices"],
                        )

                        # Side-by-side overlay images.
                        cg_col1, cg_col2 = st.columns(2)
                        with cg_col1:
                            st.markdown(f"**Prior study** — `{prior_study_metadata['study_uid']}`")
                            prior_overlay = prior_results["overlay"]
                            prior_overlay_rgb = (
                                prior_overlay[..., ::-1]
                                if len(prior_overlay.shape) == 3
                                else prior_overlay
                            )
                            st.image(
                                fit_panel_height(prior_overlay_rgb),
                                caption=f"Quality: {prior_decision_report.quality_status.value}",
                            )
                        with cg_col2:
                            st.markdown(f"**Current study** — `{study_metadata['study_uid']}`")
                            curr_overlay = results["overlay"]
                            curr_overlay_rgb = (
                                curr_overlay[..., ::-1]
                                if len(curr_overlay.shape) == 3
                                else curr_overlay
                            )
                            st.image(
                                fit_panel_height(curr_overlay_rgb),
                                caption=f"Quality: {decision_report.quality_status.value}",
                            )

                        # Run the deterministic comparison.
                        try:
                            comparison = generate_comparison_report(
                                prior=prior_decision_report,
                                current=decision_report,
                            )
                        except ValueError as cmp_err:
                            # INVALID quality blocks a meaningful comparison.
                            st.error(
                                f"ChangeGuard comparison cannot run: {cmp_err} "
                                "Resolve the image-quality issue and reload the study."
                            )
                        else:
                            # ---- Clinical Review Summary card ---------------
                            # Status is derived solely from the deterministic
                            # comparison thresholds and quality gates in
                            # mri_core/comparison.py — no model output is used.
                            is_stable = comparison.status == ComparisonStatus.STABLE
                            if is_stable:
                                card_bg   = "#d4edda"
                                card_border = "#28a745"
                                status_label = "🟢 STABLE"
                                summary_sentence = (
                                    "No image-derived change candidates exceeded "
                                    "the configured review thresholds."
                                )
                            else:
                                card_bg   = "#fff3cd"
                                card_border = "#e0a800"
                                status_label = "🟡 REVIEW"
                                # Condense reason strings to their opening clause
                                # (everything before the first period or the full
                                # string if no period found).
                                short_reasons = []
                                for r in comparison.reasons:
                                    clause = r.split(".")[0].strip()
                                    short_reasons.append(clause)
                                summary_sentence = "; ".join(short_reasons) + "."

                            st.markdown(
                                f"""
<div style="
    background:{card_bg};
    border-left:5px solid {card_border};
    border-radius:6px;
    padding:1rem 1.25rem;
    margin-bottom:1rem;
    color:#111827;
">
  <p style="font-size:1.15rem;font-weight:700;margin:0 0 0.35rem 0;color:#111827;">
    Clinical Review Summary &nbsp;—&nbsp; {status_label}
  </p>
  <p style="margin:0 0 0.4rem 0;font-size:0.95rem;color:#111827;">
    {summary_sentence}
  </p>
  <p style="margin:0;font-size:0.85rem;font-weight:700;color:#111827;">
    Status derived from deterministic comparison thresholds and image-quality
    gates only. &nbsp;Non-diagnostic; clinician review required.
  </p>
</div>
""",
                                unsafe_allow_html=True,
                            )
                            # -------------------------------------------------

                            if comparison.reasons:
                                st.markdown("**Change candidates requiring clinician review:**")
                                for reason in comparison.reasons:
                                    st.warning(reason)
                            else:
                                st.success(
                                    "No image-derived change candidates detected above threshold. "
                                    "This is not a clinical clearance."
                                )

                            # Auditable feature-delta table.
                            st.markdown("**Auditable image-feature deltas (prior → current)**")
                            delta_rows = [
                                {
                                    "Feature": d.feature,
                                    "Prior": round(d.prior_value, 4),
                                    "Current": round(d.current_value, 4),
                                    "Δ (absolute)": round(d.absolute_change, 4),
                                    "Δ (relative)": f"{d.relative_change * 100:.1f}%",
                                }
                                for d in comparison.feature_deltas
                            ]
                            st.dataframe(
                                pd.DataFrame(delta_rows),
                                use_container_width=True,
                                hide_index=True,
                            )

                    except Exception as e:
                        st.error(f"ChangeGuard processing error: {e}")

        else:
            # Preview before processing
            st.info("File loaded successfully. Click 'Process Image' in the sidebar to run analysis.")
            st.subheader(f"Selected Slice Preview (Slice {slice_idx + 1}/{volume.num_slices})")
            st.image(
                display_slice,
                channels="BGR" if (len(display_slice.shape) == 3 and display_slice.shape[2] == 3) else "GRAY",
                width=450
            )

    except Exception as e:
        st.error(f"Error processing slice: {e}")
else:
    if status_msg:
        st.info(status_msg)
