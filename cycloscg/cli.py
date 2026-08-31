from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Sequence

import matplotlib as mpl
# Headless batch rendering avoids a GUI/Tcl dependency on training servers.
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .data.dataset import DynamicCycleDataset, DynamicWaveformDataset
from .losses import CycloSCGLoss
from .data.mixing import DynamicMixer
from .data.records import DatabaseIndex
from .data.rpeak import detect_rpeaks
from .data.split import ParticipantSplit, create_participant_split
from .models import CycloSCGNet, ResidualUNet1D
from .training import CycloTrainer
from .training.trainer import BaselineTrainer
from .utils.config import load_yaml
from .utils.seed import resolve_device, seed_everything


def _json_print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _ensure_split(database: DatabaseIndex, path: Path, seed: int) -> ParticipantSplit:
    if path.is_file():
        return ParticipantSplit.load(path)
    split = create_participant_split(
        database.participant_ids("ChestSCG"),
        database.participant_ids("PositionNoise"),
        seed=seed,
    )
    split.save(path)
    return split


def inspect_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the manifest-driven SCG database")
    parser.add_argument("--database-root", default=r"H:\\数据库")
    parser.add_argument("--output", default="results/logs/dataset_audit.json")
    parser.add_argument("--quick", action="store_true", help="Inspect representative files only")
    args = parser.parse_args(argv)
    database = DatabaseIndex(args.database_root)
    report = database.audit(full_scan=not args.quick)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    _json_print(report)
    return 0 if report["passed"] else 2


def split_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create leakage-safe participant splits")
    parser.add_argument("--database-root", default=r"H:\\数据库")
    parser.add_argument("--output", default="configs/splits.json")
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args(argv)
    database = DatabaseIndex(args.database_root)
    split = create_participant_split(
        database.participant_ids("ChestSCG"),
        database.participant_ids("PositionNoise"),
        seed=args.seed,
    )
    split.save(args.output)
    _json_print(split.__dict__)
    return 0


def _configure_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save_publication_figure(fig: mpl.figure.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")


def rpeak_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect and visually verify ECG R peaks")
    parser.add_argument("--database-root", default=r"H:\\数据库")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--category",
        choices=["all", "rest", "walking"],
        default="all",
    )
    parser.add_argument("--output-root", default="results")
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Save R-peak/QC JSON without rendering per-record figures",
    )
    args = parser.parse_args(argv)
    database = DatabaseIndex(args.database_root)
    if args.category == "rest":
        records = database.records(category="CleanRestSCG")
    elif args.category == "walking":
        records = database.records(category="RealWalkingContaminatedSCG")
    else:
        records = [record for record in database.records() if record.cohort == "ChestSCG"]
    rng = random.Random(args.seed)
    selected = list(records)
    if args.n > 0:
        rng.shuffle(selected)
        selected = selected[: min(args.n, len(selected))]
    output_root = Path(args.output_root)
    if not args.no_figures:
        _configure_publication_style()
    failed: list[str] = []
    summaries: list[dict[str, object]] = []
    for record in selected:
        loaded = database.load_record(record, include_ecg=True)
        ecg = np.asarray(loaded["ecg"], dtype=np.float64)
        detection = detect_rpeaks(ecg, record.sampling_rate_hz, strict=False)
        summary = {"record_id": record.record_id, **detection.to_dict()}
        summaries.append(summary)
        rpeak_output = output_root / "rpeaks" / f"{record.record_id}.json"
        rpeak_output.parent.mkdir(parents=True, exist_ok=True)
        with rpeak_output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        if not detection.qc_passed:
            failed.append(record.record_id)

        if not args.no_figures:
            time = np.arange(len(ecg)) / record.sampling_rate_hz
            fig, axis = plt.subplots(figsize=(7.09, 2.35), constrained_layout=True)
            axis.plot(time, ecg, color="#496A81", linewidth=0.65, label="ECG")
            axis.scatter(
                time[detection.indices],
                ecg[detection.indices],
                s=10,
                color="#B24745",
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
                label="Detected R peak",
            )
            status = "PASS" if detection.qc_passed else "REVIEW"
            rr_text = (
                f"median RR={np.median(detection.rr_intervals_s):.3f} s"
                if len(detection.rr_intervals_s)
                else "median RR=NA"
            )
            axis.text(
                0.995,
                0.95,
                f"QC: {status}\nHR={detection.heart_rate_bpm:.1f} bpm\n{rr_text}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7,
            )
            axis.set(
                xlabel="Time (s)",
                ylabel="ECG (mV)",
                title=f"R-wave alignment QC | {record.record_id}",
            )
            axis.legend(loc="upper left", ncols=2)
            _save_publication_figure(
                fig, output_root / "figures" / "rpeaks" / f"{record.record_id}_rpeaks"
            )
            plt.close(fig)
    report = {"checked": len(selected), "failed_qc": failed, "records": summaries}
    report_path = output_root / "logs" / "rpeak_qc_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    _json_print({"checked": len(selected), "failed_qc": failed})
    return 0 if not failed else 2


def _worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def baseline_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the conventional waveform U-Net baseline")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--smoke", action="store_true", help="Run one tiny CPU validation epoch")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--val-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    project = config["project"]
    data_config = config["data"]
    model_config = config["model"]
    training = config["training"]
    seed = int(project["seed"])
    seed_everything(seed)
    database = DatabaseIndex(data_config["database_root"])
    split_path = Path(data_config["split_manifest"])
    split = _ensure_split(database, split_path, seed)

    train_clean = database.records("CleanRestSCG", split.train_clean_subjects)
    val_clean = database.records("CleanRestSCG", split.val_clean_subjects)
    train_noise = database.records("MotionNoiseProxy", split.train_noise_subjects)
    val_noise = database.records("MotionNoiseProxy", split.val_noise_subjects)
    train_count = int(data_config["train_samples_per_epoch"])
    val_count = int(data_config["val_samples_per_epoch"])
    window_samples = int(data_config["waveform_samples"])
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    output_dir = Path(project["output_root"])
    device_request = str(training["device"])
    if args.epochs is not None:
        epochs = args.epochs
    if args.train_samples is not None:
        train_count = args.train_samples
    if args.val_samples is not None:
        val_count = args.val_samples
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.output_root is not None:
        output_dir = Path(args.output_root)
    if args.device is not None:
        device_request = args.device
    if args.smoke:
        train_count, val_count = 8, 4
        window_samples = min(window_samples, 512)
        epochs, batch_size = 1, 2
        output_dir = Path("results/smoke_baseline")
        device_request = "cpu"

    train_mixer = DynamicMixer(
        data_config["snr_db_min"],
        data_config["snr_db_max"],
        data_config["random_polarity"],
        data_config["identity_probability"],
    )
    validation_mixer = DynamicMixer(
        data_config["snr_db_min"],
        data_config["snr_db_max"],
        data_config["random_polarity"],
        0.0,
    )
    common = {
        "database": database,
        "window_samples": window_samples,
        "signal_columns": data_config["signal_columns"],
        "ecg_columns": data_config["ecg_columns"],
        "normalize_by_clean_rms": data_config["normalize_by_clean_rms"],
    }
    train_dataset = DynamicWaveformDataset(
        clean_records=train_clean,
        noise_records=train_noise,
        samples_per_epoch=train_count,
        mixer=train_mixer,
        seed=seed,
        **common,
    )
    val_dataset = DynamicWaveformDataset(
        clean_records=val_clean,
        noise_records=val_noise,
        samples_per_epoch=val_count,
        mixer=validation_mixer,
        seed=seed + 10_000,
        **common,
    )
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": int(training["num_workers"]),
        "worker_init_fn": _worker_seed,
        "generator": generator,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = ResidualUNet1D(**model_config)
    device = resolve_device(device_request)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = dict(config)
    resolved["runtime"] = {
        "device": str(device),
        "smoke": bool(args.smoke),
        "database_manifest_records": len(database.frame),
    }
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, allow_unicode=True, sort_keys=False)
    trainer = BaselineTrainer(
        model,
        optimizer,
        device,
        output_dir,
        patience=int(training["patience"]),
        min_delta=float(training["min_delta"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        mixed_precision=bool(training.get("mixed_precision", False)),
    )
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs,
        resume=args.resume if args.resume is not None else training.get("resume"),
    )
    _json_print(
        {
            "device": str(device),
            "epochs_completed": len(history),
            "best_val_loss": trainer.best_val_loss,
            "best_checkpoint": str(output_dir / "checkpoints" / "best.pt"),
            "gpu_peak_memory_gb": (
                torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
            ),
        }
    )
    return 0


def cycloscg_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CycloSCGNet on controlled contamination")
    parser.add_argument("--config", default="configs/cycloscg.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--val-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume")
    parser.add_argument("--init-checkpoint")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    project = config["project"]
    data_config = config["data"]
    model_config = dict(config["model"])
    loss_config = config["loss"]
    training = config["training"]
    seed = int(project["seed"])
    seed_everything(seed)
    database = DatabaseIndex(data_config["database_root"])
    split = _ensure_split(database, Path(data_config["split_manifest"]), seed)
    train_count = int(data_config["train_samples_per_epoch"])
    val_count = int(data_config["val_samples_per_epoch"])
    beats_per_window = int(data_config["beats_per_window"])
    phase_length = int(data_config["phase_length"])
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    output_dir = Path(project["output_root"])
    device_request = str(training["device"])
    if args.epochs is not None:
        epochs = args.epochs
    if args.train_samples is not None:
        train_count = args.train_samples
    if args.val_samples is not None:
        val_count = args.val_samples
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.output_root is not None:
        output_dir = Path(args.output_root)
    if args.device is not None:
        device_request = args.device
    if args.smoke:
        train_count, val_count = 4, 2
        beats_per_window, phase_length = 4, 128
        epochs, batch_size = 1, 2
        model_config["base_channels"] = 4
        output_dir = Path("results/smoke_cycloscg")
        device_request = "cpu"

    common = {
        "database": database,
        "beats_per_window": beats_per_window,
        "phase_length": phase_length,
        "signal_columns": data_config["signal_columns"],
        "ecg_columns": data_config["ecg_columns"],
        "normalize_by_clean_rms": data_config["normalize_by_clean_rms"],
        "strict_rpeak_qc": data_config["strict_rpeak_qc"],
    }
    train_dataset = DynamicCycleDataset(
        clean_records=database.records("CleanRestSCG", split.train_clean_subjects),
        noise_records=database.records("MotionNoiseProxy", split.train_noise_subjects),
        samples_per_epoch=train_count,
        mixer=DynamicMixer(
            data_config["snr_db_min"],
            data_config["snr_db_max"],
            data_config["random_polarity"],
            data_config["identity_probability"],
        ),
        seed=seed,
        **common,
    )
    val_dataset = DynamicCycleDataset(
        clean_records=database.records("CleanRestSCG", split.val_clean_subjects),
        noise_records=database.records("MotionNoiseProxy", split.val_noise_subjects),
        samples_per_epoch=val_count,
        mixer=DynamicMixer(
            data_config["snr_db_min"],
            data_config["snr_db_max"],
            data_config["random_polarity"],
            0.0,
        ),
        seed=seed + 10_000,
        **common,
    )
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": int(training["num_workers"]),
        "worker_init_fn": _worker_seed,
        "generator": generator,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    model = CycloSCGNet(**model_config)
    if args.resume and args.init_checkpoint:
        raise ValueError("Use either --resume or --init-checkpoint, not both")
    if args.init_checkpoint:
        initialization = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(initialization["model_state"])
    loss_function = CycloSCGLoss(
        weights=loss_config["weights"],
        stft_resolutions=loss_config["stft_resolutions"],
        cyclic_lags=loss_config["cyclic_lags"],
        use_cyclic_loss=loss_config["use_cyclic_loss"],
    )
    device = resolve_device(device_request)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = dict(config)
    resolved["model"] = dict(model_config)
    resolved["data"] = dict(data_config)
    resolved["data"]["beats_per_window"] = beats_per_window
    resolved["data"]["phase_length"] = phase_length
    resolved["runtime"] = {
        "device": str(device),
        "smoke": bool(args.smoke),
        "beats_per_window": beats_per_window,
        "phase_length": phase_length,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": epochs,
        "train_samples_per_epoch": train_count,
        "val_samples_per_epoch": val_count,
        "batch_size": batch_size,
        "mixed_precision": bool(training.get("mixed_precision", False)),
        "initialized_from": args.init_checkpoint,
    }
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, allow_unicode=True, sort_keys=False)
    trainer = CycloTrainer(
        model,
        loss_function,
        optimizer,
        device,
        output_dir,
        patience=int(training["patience"]),
        min_delta=float(training["min_delta"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        mixed_precision=bool(training.get("mixed_precision", False)),
    )
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs,
        resume=args.resume if args.resume is not None else training.get("resume"),
    )
    _json_print(
        {
            "device": str(device),
            "epochs_completed": len(history),
            "best_val_loss": trainer.best_val_loss,
            "best_checkpoint": str(output_dir / "checkpoints" / "best.pt"),
            "gpu_peak_memory_gb": (
                torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
            ),
        }
    )
    return 0
