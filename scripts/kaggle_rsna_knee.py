"""RSNA Knee metadata inspection, training, evaluation, and submission entrypoint."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mri_core.rsna_knee_dataset import TARGET_COLUMNS, load_rsna_metadata, metadata_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--data-dir", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--data-dir", type=Path, required=True); train.add_argument("--config", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--data-dir", type=Path, required=True); evaluate.add_argument("--checkpoint", type=Path, required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--data-dir", type=Path, required=True); submit.add_argument("--checkpoint", type=Path, required=True); submit.add_argument("--output", type=Path, required=True)
    for command in (inspect, train, evaluate, submit):
        command.add_argument("--dicom-root", type=Path, default=None,
                             help="folder holding train_series/ and test_series/ (default: --data-dir)")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        print(json.dumps(metadata_report(load_rsna_metadata(args.data_dir, args.dicom_root)), indent=2))
        return 0
    metadata = load_rsna_metadata(args.data_dir, args.dicom_root)
    if args.command == "train":
        from mri_core.rsna_knee_train import train_rsna
        config = json.loads(args.config.read_text(encoding="utf-8"))
        print(json.dumps(train_rsna(metadata, config, Path(config["checkpoint_path"])), indent=2)); return 0
    if args.command == "evaluate":
        from mri_core.rsna_knee_train import calculate_auc
        checkpoint = __import__("torch").load(args.checkpoint, map_location="cpu", weights_only=True)
        print(json.dumps({"checkpoint": str(args.checkpoint), "validation": checkpoint.get("validation"), "targets": TARGET_COLUMNS}, indent=2)); return 0
    from mri_core.rsna_knee_submit import write_submission
    write_submission(args.data_dir, args.checkpoint, args.output, dicom_root=args.dicom_root)
    print(json.dumps({"output": str(args.output), "columns": ["StudyInstanceUID", *TARGET_COLUMNS]})); return 0


if __name__ == "__main__":
    raise SystemExit(main())
