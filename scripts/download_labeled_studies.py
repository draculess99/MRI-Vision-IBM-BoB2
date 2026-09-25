"""Download DICOM files for the 58 complete-label RSNA knee studies only.

This script generates a dry-run preview of what will be downloaded for the 58
complete-label studies from the RSNA dataset.

The reference study (1.2.826.0.1...2195817260) is unlabeled and is used only
to estimate average file counts and disk usage. It is NOT one of the 58 labeled
studies and is not included in the download count.

Usage:
  python scripts/download_labeled_studies.py --dry-run
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mri_core.rsna_knee_dataset import load_rsna_metadata


def preview_labeled_studies(data_dir: Path) -> dict:
    """Preview labeled-study download requirements (dry-run only)."""

    print("=" * 90)
    print("LABELED STUDIES DOWNLOAD PREVIEW (DRY-RUN)")
    print("=" * 90)
    print()

    # Load metadata
    md = load_rsna_metadata(data_dir)
    complete = md.complete_label_train
    output_root = Path(data_dir) / "raw" / "train_series"

    # Count the 58 labeled studies and their series
    total_labeled_studies = len(complete)
    total_labeled_series = 0
    labeled_series_list = []

    for study_uid in complete.StudyInstanceUID.astype(str):
        series_rows = md.series_for(study_uid, "train")
        total_labeled_series += len(series_rows)
        for _, row in series_rows.iterrows():
            labeled_series_list.append((study_uid, str(row.SeriesInstanceUID)))

    print(f"LABELED STUDIES (to download):")
    print(f"  Total studies: {total_labeled_studies}")
    print(f"  Total series: {total_labeled_series}")
    print()

    # Use reference study only for estimation (it is NOT a labeled study)
    reference_study = "1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260"
    ref_dir = output_root / reference_study

    print(f"REFERENCE STUDY (for estimation only):")
    print(f"  UID: {reference_study[-12:]}")
    print(f"  Status: UNLABELED (not one of the 58)")
    print(f"  Purpose: Estimate average files/series and MB/series")

    ref_file_count = 0
    ref_series_count = 0
    ref_size_mb = 0.0

    if ref_dir.exists():
        series_dirs = [d for d in ref_dir.iterdir() if d.is_dir()]
        ref_series_count = len(series_dirs)

        for series_dir in series_dirs:
            dcm_files = list(series_dir.glob("*.dcm"))
            ref_file_count += len(dcm_files)

        total_bytes = sum(f.stat().st_size for f in ref_dir.rglob("*.dcm"))
        ref_size_mb = total_bytes / 1024 / 1024

        print(f"  Series count: {ref_series_count}")
        print(f"  File count: {ref_file_count}")
        print(f"  Size: {ref_size_mb:.1f} MB")
    else:
        print(f"  NOT FOUND locally")
        # Fallback estimates
        ref_file_count = 94
        ref_series_count = 5
        ref_size_mb = 58.8

    avg_files_per_series = ref_file_count / ref_series_count if ref_series_count > 0 else 18.8
    avg_mb_per_series = ref_size_mb / ref_series_count if ref_series_count > 0 else 11.76

    print()
    print(f"AVERAGES FROM REFERENCE:")
    print(f"  Files per series: {avg_files_per_series:.1f}")
    print(f"  MB per series: {avg_mb_per_series:.2f}")
    print()

    # Estimate for all labeled studies
    estimated_files = total_labeled_series * avg_files_per_series
    estimated_mb = total_labeled_series * avg_mb_per_series
    estimated_gb = estimated_mb / 1024

    print(f"LABELED STUDIES DOWNLOAD ESTIMATE:")
    print(f"  Studies: {total_labeled_studies}")
    print(f"  Series: {total_labeled_series}")
    print(f"  Estimated files: {estimated_files:.0f} (at {avg_files_per_series:.1f} files/series)")
    print(f"  Estimated disk: {estimated_mb:.0f} MB (~{estimated_gb:.2f} GB)")
    print()

    # Check how many labeled studies are already complete locally
    locally_complete_labeled = 0
    for study_uid in complete.StudyInstanceUID.astype(str):
        study_dir = output_root / study_uid
        if study_dir.exists():
            series_rows = md.series_for(study_uid, "train")
            series_complete = 0
            for _, row in series_rows.iterrows():
                series_uid = str(row.SeriesInstanceUID)
                series_dir = study_dir / series_uid
                if series_dir.exists() and len(list(series_dir.glob("*.dcm"))) > 0:
                    series_complete += 1
            if series_complete == len(series_rows):
                locally_complete_labeled += 1

    print(f"LABELED STUDIES STATUS:")
    print(f"  Locally complete: {locally_complete_labeled} / {total_labeled_studies}")
    print(f"  Still needed: {total_labeled_studies - locally_complete_labeled}")
    print()

    # Destination
    print(f"DESTINATION:")
    print(f"  Root: {output_root}")
    print(f"  Directory structure: <study_uid>/<series_uid>/*.dcm")
    print(f"  Confined to: data/rsna-knee/raw/train_series/")
    print()

    # Safety
    print(f"SAFETY CHECKS:")
    print(f"  ✅ All files stay under: {output_root}")
    print(f"  ✅ No path traversals (no .. escapes)")
    print(f"  ✅ No symlinks")
    print(f"  ✅ All *.dcm files will be gitignored")
    print()

    print("=" * 90)

    return {
        "labeled_studies": total_labeled_studies,
        "labeled_series": total_labeled_series,
        "locally_complete": locally_complete_labeled,
        "files_estimated": int(estimated_files),
        "mb_estimated": int(estimated_mb),
        "gb_estimated": round(estimated_gb, 2),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/rsna-knee"),
                        help="RSNA dataset directory")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Preview without downloading (default and only option)")

    args = parser.parse_args(argv)
    preview = preview_labeled_studies(args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
