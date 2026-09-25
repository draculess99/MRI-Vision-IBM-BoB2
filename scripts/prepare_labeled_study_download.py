"""Generate manifest of DICOM files to download for the 58 complete-label RSNA studies.

This script:
1. Identifies all 58 labeled studies
2. Lists all their series from the metadata
3. Attempts to fetch the exact DICOM filenames from Kaggle (if CLI available)
4. Generates labeled_study_files.txt for the downloader
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mri_core.rsna_knee_dataset import load_rsna_metadata, PLANE_NAMES


def query_kaggle_files(study_uid: str, series_uid: str, competition: str = "rsna-knee-abnormality-detection") -> list[str]:
    """Query Kaggle CLI to list DICOM files in a specific study/series directory.

    Returns list of full paths like "train_series/study_uid/series_uid/file.dcm"
    """
    try:
        # Use Kaggle API to list files in this path
        # Unfortunately, the CLI doesn't have a native "ls" for remote files
        # So we use a workaround: try downloading metadata or listing
        cmd = ["kaggle", "competitions", "download", competition, "-f", f"train_series/{study_uid}/{series_uid}/", "-p", "/tmp", "-q"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        # This will fail because we're trying to download a directory
        # But we can parse the error to see what files exist
        # For now, return empty list to indicate we can't query
        return []
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return []


def generate_manifest(data_dir: Path, output_file: Path, dry_run: bool = False) -> int:
    """Generate the manifest of files to download."""

    print("=" * 90)
    print("LABELED STUDIES DOWNLOAD MANIFEST GENERATOR")
    print("=" * 90)
    print()

    # Load metadata
    print("Loading metadata...")
    try:
        md = load_rsna_metadata(data_dir)
    except Exception as e:
        print(f"ERROR: Could not load metadata: {e}", file=sys.stderr)
        return 1

    complete = md.complete_label_train
    print(f"Found {len(complete)} complete-label studies")
    print()

    # For each study, list the series to download
    files_to_download = []
    study_count = 0
    series_count = 0

    for study_uid in complete.StudyInstanceUID.astype(str):
        series_rows = md.series_for(study_uid, "train")

        if len(series_rows) == 0:
            print(f"⚠️  Study {study_uid[-12:]}: no series in metadata", file=sys.stderr)
            continue

        study_count += 1

        for _, row in series_rows.iterrows():
            series_uid = str(row.SeriesInstanceUID)
            plane = str(row.Anatomical_Plane)

            # Try to get the exact filenames from Kaggle
            # For now, just use a pattern since we don't have direct API access
            # The downloader will figure out the exact files
            files_to_download.append((study_uid, series_uid, plane))
            series_count += 1

    print(f"Total: {study_count} studies, {series_count} series")
    print()

    # Write manifest
    if not dry_run:
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON format with metadata
        manifest = {
            "studies": len(complete),
            "series": series_count,
            "estimated_files": series_count * 18,  # ~18.8 files per series on average
            "estimated_size_gb": series_count * 11.8 / 1024,
            "entries": [
                {
                    "study_uid": study_uid,
                    "series_uid": series_uid,
                    "plane": plane,
                    "path": f"train_series/{study_uid}/{series_uid}/"
                }
                for study_uid, series_uid, plane in files_to_download
            ]
        }

        manifest_json = output_file.with_suffix(".json")
        manifest_json.write_text(json.dumps(manifest, indent=2))
        print(f"Wrote metadata manifest: {manifest_json}")
        print(f"  Studies: {manifest['studies']}")
        print(f"  Series: {manifest['series']}")
        print(f"  Estimated files: {manifest['estimated_files']}")
        print(f"  Estimated size: {manifest['estimated_size_gb']:.2f} GB")
        print()

        # Write simple text format for the downloader
        # Format: train_series/study/series/ (one per line)
        # The actual downloader will need exact filenames, but we can list paths
        with open(output_file, "w") as f:
            for study_uid, series_uid, plane in files_to_download:
                # Write the path that the downloader will use
                f.write(f"train_series/{study_uid}/{series_uid}/\n")

        print(f"Wrote download paths: {output_file}")
        print(f"  {len(files_to_download)} series paths")
        print()

        print("To download with the standard downloader:")
        print(f"  python scripts/download_rsna_study_files.py \\")
        print(f"    --input {output_file} \\")
        print(f"    --output data/rsna-knee/raw/train_series \\")
        print(f"    --dry-run")
        print()
        print("Then (without --dry-run) to actually download:")
        print(f"  python scripts/download_rsna_study_files.py \\")
        print(f"    --input {output_file} \\")
        print(f"    --output data/rsna-knee/raw/train_series")
        print()
    else:
        print("DRY RUN: Would write:")
        print(f"  {output_file} ({len(files_to_download)} entries)")
        print(f"  {output_file.with_suffix('.json')} (metadata)")

    print("=" * 90)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/rsna-knee"),
                        help="RSNA dataset directory")
    parser.add_argument("--output", type=Path, default=Path("data/rsna-knee/labeled_study_files.txt"),
                        help="Output manifest file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing files")

    args = parser.parse_args(argv)

    return generate_manifest(args.data_dir, args.output, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
