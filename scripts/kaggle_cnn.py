"""Kaggle-ready entrypoint; identical implementation to python -m mri_core.cnn_train."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mri_core.cnn_train import main


if __name__ == "__main__":
    raise SystemExit(main())
