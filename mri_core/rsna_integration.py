"""Read-only RSNA dataset integration with graceful partial-download handling.

Discovers locally available RSNA studies and safely loads them through the existing
volume and metadata infrastructure. Handles incomplete downloads without crashing.
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from .dicom_series import DicomSeriesError
from .mri_volume import MRIVolume
from .rsna_knee_dataset import (
    RSNAKneeMetadata, load_rsna_metadata, load_series_volume, select_series
)


class RSNADiscoveryError(Exception):
    """Root RSNA data directory or metadata could not be discovered."""


class RSNAStudyNotAvailable(Exception):
    """Study exists in metadata but no local DICOM data was found."""


def discover_rsna_root(explicit_root: Optional[Path | str] = None) -> Path:
    """Discover RSNA data root directory.

    Search order:
    1. Explicit root if provided
    2. data/rsna-knee relative to current working directory
    3. data/rsna-knee relative to repo root (parent of mri_core)
    4. Environment variable RSNA_ROOT if set

    Raises RSNADiscoveryError if not found.
    """
    if explicit_root is not None:
        path = Path(explicit_root)
        if (path / "train.csv").is_file():
            return path
        raise RSNADiscoveryError(f"Explicit RSNA root {path} missing train.csv")

    candidates = [
        Path.cwd() / "data" / "rsna-knee",
        Path(__file__).resolve().parents[1] / "data" / "rsna-knee",
    ]

    for candidate in candidates:
        if (candidate / "train.csv").is_file():
            return candidate

    env_root = Path(os.environ.get("RSNA_ROOT", "")) if "RSNA_ROOT" in os.environ else None
    if env_root and (env_root / "train.csv").is_file():
        return env_root

    raise RSNADiscoveryError(
        f"Could not discover RSNA root. Tried: {candidates}. "
        "Set RSNA_ROOT environment variable or pass explicit_root."
    )


def load_rsna_metadata_safe(data_root: Path) -> RSNAKneeMetadata:
    """Load RSNA metadata CSVs. Raises RSNADiscoveryError if any required file is missing."""
    try:
        return load_rsna_metadata(data_root, dicom_root=data_root / "raw")
    except FileNotFoundError as e:
        raise RSNADiscoveryError(f"Failed to load RSNA metadata: {e}") from e
    except ValueError as e:
        raise RSNADiscoveryError(f"Invalid RSNA metadata: {e}") from e


def _read_cache_manifest(data_root: Path) -> Dict[str, Dict[str, List[str]]]:
    """Read the cache manifest if available to know which studies have been downloaded.

    Format: {study_uid: {series_uid: [file_paths]}}
    Returns empty dict if cache doesn't exist (no downloads yet).
    """
    cache_file = data_root / "kaggle_files_cache.json"
    if not cache_file.is_file():
        return {}
    try:
        with open(cache_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def discover_available_studies(data_root: Path, metadata: RSNAKneeMetadata) -> List[str]:
    """Enumerate study UIDs that have locally available DICOM data.

    A study is considered available if:
    - It appears in the train metadata
    - It has at least one series directory under raw/train_series/{study_uid}/

    Handles gracefully if cache doesn't exist (early in download) or is incomplete.
    """
    available = []
    train_series_root = data_root / "raw" / "train_series"

    if not train_series_root.is_dir():
        return []

    for study_path in sorted(train_series_root.iterdir()):
        if not study_path.is_dir():
            continue
        study_uid = study_path.name
        # Check if this study is in our metadata
        if study_uid in metadata.train["StudyInstanceUID"].astype(str).values:
            # Check if it has at least one series directory
            if any(p.is_dir() for p in study_path.iterdir()):
                available.append(study_uid)

    return available


def get_available_planes(
    data_root: Path, metadata: RSNAKneeMetadata, study_uid: str
) -> List[str]:
    """Get plane names that have downloadable series for this study.

    Returns list of plane names ("Axial", "Coronal", "Sagittal") that:
    - Are listed in metadata for this study
    - Have at least one series directory present locally (even if incomplete)
    """
    series_rows = metadata.series_for(study_uid, split="train")
    available_planes = []

    train_series_root = data_root / "raw" / "train_series"
    study_path = train_series_root / study_uid

    for plane in sorted(series_rows["Anatomical_Plane"].unique()):
        plane = str(plane)
        plane_candidates = series_rows[series_rows["Anatomical_Plane"] == plane]
        for _, row in plane_candidates.iterrows():
            series_uid = str(row["SeriesInstanceUID"])
            series_path = study_path / series_uid
            if series_path.is_dir() and any(p.is_file() for p in series_path.iterdir()):
                if plane not in available_planes:
                    available_planes.append(plane)
                break

    return sorted(available_planes)


def load_rsna_study_series(
    data_root: Path, metadata: RSNAKneeMetadata, study_uid: str, plane: str
) -> Tuple[MRIVolume, Dict]:
    """Load an RSNA study series for a given plane.

    Uses deterministic select_series to pick one series per plane.

    Args:
        data_root: Root RSNA data directory
        metadata: Loaded RSNA metadata
        study_uid: Study UID
        plane: Plane name ("Axial", "Coronal", or "Sagittal")

    Returns:
        (volume, metadata_dict) where volume is MRIVolume and metadata_dict is info

    Raises:
        RSNAStudyNotAvailable: Study or plane not available locally
        DicomSeriesError: Series is corrupted or unreadable
    """
    series_rows = metadata.series_for(study_uid, split="train")
    plane_candidates = series_rows[series_rows["Anatomical_Plane"].astype(str) == plane]

    if plane_candidates.empty:
        raise RSNAStudyNotAvailable(
            f"No {plane} series in metadata for study {study_uid}"
        )

    # Deterministically select series for this plane
    selected_series_uid = select_series(plane_candidates)

    # Check if it exists locally
    series_path = data_root / "raw" / "train_series" / study_uid / selected_series_uid
    if not series_path.is_dir():
        raise RSNAStudyNotAvailable(
            f"Series {selected_series_uid} for {plane} not yet downloaded "
            f"(expected at {series_path})"
        )

    # Load through existing infrastructure
    volume = load_series_volume(metadata, study_uid, selected_series_uid, split="train")

    return volume, {
        "study_uid": study_uid,
        "series_uid": selected_series_uid,
        "plane": plane,
        "num_slices": volume.num_slices,
    }
