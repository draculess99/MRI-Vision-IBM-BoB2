"""Fixed CNN experiment. Full runs require CUDA; CPU smoke uses synthetic data only."""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .baseline import validate_output_dir, write_csv
from .cnn_dataset import MRNetDataset, training_class_weights
from .cnn_evaluate import evaluate_predictions, summarize_seeds
from .cnn_model import MRNetCNN
from .labels import SPLITS, TARGETS, load_labels
from .manifest import MANIFEST_FIELDS, PLANES, build_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "v0.5_cnn.json"
ARTIFACT_ROOTS = (REPO_ROOT / "outputs" / "v0.5", REPO_ROOT / "outputs" / "v0.5-smoke",
                  Path("/kaggle/working/outputs/v0.5"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def validate_config(config):
    integer_fields = ("slices", "resolution", "epochs", "batch_size", "accumulation_steps",
                      "expected_train", "expected_valid")
    for name in integer_fields:
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["resolution"] < 16 or type(config["num_workers"]) is not int or config["num_workers"] < 0:
        raise ValueError("Invalid resolution or worker count")
    planes = config["planes"]
    if not planes or len(set(planes)) != len(planes) or not set(planes) <= set(PLANES):
        raise ValueError("Invalid planes")
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Seeds must be unique nonnegative 32-bit integers")
    for name in ("learning_rate", "weight_decay", "dropout", "rotation_degrees",
                 "translation_fraction", "contrast_fraction"):
        if not np.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"Invalid {name}")
    if (config["learning_rate"] == 0 or config["dropout"] >= 1
            or config["translation_fraction"] > 1 or config["contrast_fraction"] > 1):
        raise ValueError("Invalid optimizer or augmentation settings")
    if config["threshold"] != 0.5:
        raise ValueError("Reference experiment requires threshold 0.5")


def manifest_fingerprint(rows):
    records = [{key: row[key] for key in ("exam_id", "split", *TARGETS)} for row in rows]
    canonical = json.dumps(sorted(records, key=lambda row: row["exam_id"]), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_checkpoint(path, model, optimizer, config, seed, epoch, loader_generator):
    numpy_rng = np.random.get_state()
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "config": config, "seed": seed, "epoch": epoch,
               "targets": list(TARGETS), "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               "python_rng": random.getstate(),
               "numpy_rng": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
               "loader_rng": loader_generator.get_state()}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["targets"] != list(TARGETS):
        raise ValueError("Checkpoint target order mismatch")
    config = checkpoint["config"]
    model = MRNetCNN(len(config["planes"]), config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def predict(model, dataset, device, batch_size):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            probabilities = model(batch["images"].to(device), batch["mask"].to(device)).sigmoid().cpu().numpy()
            for exam_id, scores in zip(batch["exam_id"], probabilities):
                predictions.append({"exam_id": exam_id, **{t: float(p) for t, p in zip(TARGETS, scores)}})
    return predictions


def train_one_seed(manifest, config, seed, directory, device, *, benchmark=False, compare_reference=True):
    """Training accesses only train rows; validation is first loaded after the final epoch."""
    seed_everything(seed)
    directory.mkdir(parents=True, exist_ok=False)
    training = MRNetDataset(manifest, "train", config, augment=True, seed=seed)
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(training, batch_size=config["batch_size"], shuffle=True,
                        num_workers=config["num_workers"], worker_init_fn=seed_worker,
                        generator=loader_generator, pin_memory=device.type == "cuda", persistent_workers=False)
    model = MRNetCNN(len(config["planes"]), config["dropout"]).to(device)
    weights = training_class_weights(manifest.rows).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    synchronize(device)
    start = time.perf_counter()
    epochs = 1 if benchmark else config["epochs"]
    for epoch in range(epochs):
        training.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        synchronize(device)
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            logits = model(batch["images"].to(device), batch["mask"].to(device))
            loss = loss_function(logits, batch["labels"].to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            # Weight by actual exam count in this accumulation window, including the last short batch.
            group_start = (batch_index // config["accumulation_steps"]) * config["accumulation_steps"]
            group_end = min(group_start + config["accumulation_steps"], len(loader))
            group_examples = min(group_end * config["batch_size"], len(training)) - group_start * config["batch_size"]
            (loss * len(batch["exam_id"]) / group_examples).backward()
            total_loss += float(loss.detach()) * len(batch["exam_id"])
            if batch_index + 1 == group_end:
                nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        history.append({"epoch": epoch + 1, "training_loss": total_loss / len(training),
                        "seconds": time.perf_counter() - epoch_start})
        save_checkpoint(directory / "last.pt", model, optimizer, config, seed, epoch + 1, loader_generator)
        write_json(directory / "history.json", history)
        print(json.dumps({"seed": seed, **history[-1]}), flush=True)
    synchronize(device)
    runtime = time.perf_counter() - start
    training_report = {"seed": seed, "training_seconds": runtime,
                       "timing_definition": "Training loop including input loading, augmentation, and epoch checkpoint writes; excludes final evaluation",
                       "parameter_count": sum(p.numel() for p in model.parameters()),
                       "class_pos_weights": weights.cpu().tolist(),
                       "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                       "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
    if benchmark:
        training_report.update({"status": "benchmark_only", "projected_training_seconds_per_seed": runtime * config["epochs"],
                                "note": "One-epoch estimate includes startup overhead; validation was not loaded."})
        write_json(directory / "report.json", training_report)
        return training_report
    # Evaluate the final checkpoint, not a validation-selected checkpoint.
    model, _ = load_checkpoint(directory / "last.pt", device)
    validation = MRNetDataset(manifest, "valid", config, augment=False, seed=seed)
    synchronize(device)
    evaluation_start = time.perf_counter()
    predictions = predict(model, validation, device, config["batch_size"])
    synchronize(device)
    training_report["validation_inference_seconds"] = time.perf_counter() - evaluation_start
    write_csv(directory / "predictions.csv", ("exam_id", *TARGETS), predictions)
    report = evaluate_predictions(manifest.rows, predictions, compare_reference=compare_reference)
    report.update(training_report)
    report["status"] = "complete"
    write_json(directory / "report.json", report)
    return report


def provenance(config):
    sources = [*sorted((REPO_ROOT / "mri_core").glob("*.py")), DEFAULT_CONFIG]
    source_hashes = {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    def git(*arguments):
        result = subprocess.run(["git", "-C", str(REPO_ROOT), *arguments], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    try:
        revision, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    except OSError:
        revision, status = "unavailable", "unavailable"
    return {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda, "git_revision": revision, "git_status": status,
            "source_sha256": source_hashes, "config": config,
            "determinism_note": "Fixed seeds and deterministic operations; cross-device/version bitwise equality is not guaranteed."}


def prepare_output(root, dataset_root, labels_dir):
    root = validate_output_dir(Path(root), Path(dataset_root), Path(labels_dir))
    if not any(root == allowed.resolve() or allowed.resolve() in root.parents for allowed in ARTIFACT_ROOTS):
        raise ValueError("Outputs must be under repository outputs/v0.5[-smoke] or /kaggle/working/outputs/v0.5")
    run_dir = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def execute(dataset_root, labels_dir, output_root, config, device, *, recover=False, smoke=False, benchmark=False):
    validate_config(config)
    if not smoke and device.type != "cuda":
        raise ValueError("Full MRNet training and benchmarking require CUDA; use smoke for synthetic CPU checks")
    if smoke and (config["epochs"] != 1 or config["expected_train"] > 8 or config["resolution"] > 32):
        raise ValueError("CPU smoke must remain tiny")
    labels = load_labels(labels_dir, recover_first_row=recover)
    for recovery in labels.recoveries:
        print("CSV recovery: " + json.dumps(recovery), flush=True)
    manifest = build_manifest(labels, dataset_root)
    counts = {split: sum(row["split"] == split for row in manifest.rows) for split in SPLITS}
    if counts != {split: config[f"expected_{split}"] for split in SPLITS}:
        raise ValueError(f"Unexpected split counts: {counts}")
    fingerprint = manifest_fingerprint(manifest.rows)
    if not smoke and fingerprint != config["reference_manifest_sha256"]:
        raise ValueError("Labels or split membership differ from V0.4; refusing reference comparison")
    if manifest.missing_images or manifest.unlabeled_images:
        raise ValueError("Missing or unlabeled images; resolve the manifest before training")
    directory = prepare_output(output_root, dataset_root, labels_dir)
    write_json(directory / "config.json", config)
    write_csv(directory / "manifest.csv", MANIFEST_FIELDS, manifest.rows)
    info = provenance(config)
    info.update({"mode": "synthetic_smoke" if smoke else "benchmark" if benchmark else "full_experiment",
                 "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                 "split_counts": counts, "manifest_sha256": fingerprint, "label_recoveries": labels.recoveries,
                 "label_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(Path(labels_dir).glob("*.csv"))}})
    write_json(directory / "provenance.json", info)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (directory / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
    # Record exact deterministic sampling without reading pixel arrays or applying augmentations.
    from .baseline_features import sampled_indices
    sampling = []
    for row in manifest.rows:
        for plane in config["planes"]:
            volume = np.load(manifest.paths[(row["exam_id"], plane)], mmap_mode="r", allow_pickle=False)
            sampling.append({"exam_id": row["exam_id"], "plane": plane,
                             "indices": sampled_indices(volume.shape[0], config["slices"]).tolist()})
    write_json(directory / "sampled_indices.json", sampling)
    reports = []
    for seed in config["seeds"][:1] if benchmark else config["seeds"]:
        reports.append(train_one_seed(manifest, config, seed, directory / f"seed-{seed}", device,
                                      benchmark=benchmark, compare_reference=not smoke and config["planes"] == list(PLANES)))
    if not benchmark:
        write_json(directory / "summary.json", {"mode": info["mode"], "seeds": config["seeds"],
                   "seed_selection": "All predefined seeds reported; none selected by validation performance",
                   "metrics": summarize_seeds(reports),
                   "limitations": ["Same development validation set used for V0.4; not an untouched external test.",
                                    "Only 120 real validation exams; seed variability is not a confidence interval.",
                                    "Examination separation does not prove patient-level independence.",
                                    "Sparse low-resolution slice summaries may miss focal tears; probabilities are uncalibrated."]})
    print(f"Artifacts: {directory}", flush=True)
    return directory


def smoke_fixture(root):
    """Generate tiny synthetic volumes; never read the Stanford dataset."""
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for plane in PLANES:
        (root / plane).mkdir()
        for i in range(12):
            np.save(root / plane / f"{i:04d}.npy", rng.integers(0, 256, (2, 32, 32), dtype=np.uint8))
    for split, ids in (("train", range(8)), ("valid", range(8, 12))):
        for offset, target in enumerate(TARGETS):
            with (labels_dir / f"{split}_{target}.csv").open("w", newline="") as stream:
                csv.writer(stream).writerows((f"{i:04d}", (i + offset) % 2) for i in ids)
    return labels_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "benchmark", "train"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--recover-first-row", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    if args.mode == "smoke":
        if args.dataset_root is not None or args.labels_dir is not None or args.recover_first_row:
            parser.error("smoke generates synthetic data and accepts no dataset/label inputs")
        torch.set_num_threads(2)
        config.update({"experiment": "synthetic_smoke", "planes": ["axial"], "slices": 3,
                       "resolution": 32, "epochs": 1, "batch_size": 4, "accumulation_steps": 1,
                       "num_workers": 0, "seeds": [0], "expected_train": 8, "expected_valid": 4})
        with tempfile.TemporaryDirectory(prefix="mri-cnn-synthetic-") as temporary:
            root = Path(temporary)
            execute(root, smoke_fixture(root), args.output_root or ARTIFACT_ROOTS[1], config,
                    torch.device("cpu"), smoke=True)
        return 0
    if args.dataset_root is None:
        parser.error("--dataset-root is required for CUDA training/benchmarking")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required. No CPU fallback: use smoke for local synthetic checks.")
    execute(args.dataset_root, args.labels_dir or args.dataset_root / "labels",
            args.output_root or ARTIFACT_ROOTS[0], config, torch.device("cuda:0"),
            recover=args.recover_first_row, benchmark=args.mode == "benchmark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
