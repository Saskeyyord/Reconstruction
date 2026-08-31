from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from cycloscg.data.cardiac_phase import normalize_cardiac_cycles
from cycloscg.data.records import DatabaseIndex
from cycloscg.data.rpeak import detect_rpeaks
from cycloscg.data.split import ParticipantSplit
from cycloscg.evaluation.common import (
    load_baseline_model,
    load_cycloscg_model,
    reconstruct_cycle_matrix,
)
from cycloscg.metrics import cycle_structure_metrics, cyclostationary_metrics
from cycloscg.utils.config import load_yaml
from cycloscg.utils.seed import resolve_device, seed_everything
from cycloscg.visualization.publication import (
    plot_real_walking,
    plot_real_walking_waveform_before_after,
)


def _normalize_recording(signal: np.ndarray) -> tuple[np.ndarray, float]:
    scale = max(float(np.sqrt(np.mean(np.asarray(signal, dtype=np.float64) ** 2))), 1e-6)
    return np.asarray(signal, dtype=np.float32) / scale, scale


def _structural_row(
    participant: str,
    condition: str,
    method: str,
    cycles: np.ndarray,
    rest_cycles: np.ndarray,
) -> dict[str, object]:
    metrics = cycle_structure_metrics(cycles)
    rest_metrics = cycle_structure_metrics(rest_cycles)
    metrics.update(cyclostationary_metrics(cycles, rest_cycles))
    metrics.update(
        {
            "pairwise_distance_to_rest": abs(
                metrics["mean_pairwise_correlation"] - rest_metrics["mean_pairwise_correlation"]
            ),
            "template_distance_to_rest": abs(
                metrics["mean_template_correlation"] - rest_metrics["mean_template_correlation"]
            ),
            "variability_relative_error_to_rest": abs(
                metrics["cycle_variability"] - rest_metrics["cycle_variability"]
            )
            / max(rest_metrics["cycle_variability"], 1e-12),
        }
    )
    return {
        "participant": participant,
        "condition": condition,
        "method": method,
        "num_cycles": len(cycles),
        **metrics,
    }


def walking_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="External real-walking structural validation")
    parser.add_argument("--config", default="configs/cycloscg.yaml")
    parser.add_argument("--cycloscg-checkpoint", required=True)
    parser.add_argument("--baseline-config", default="configs/baseline.yaml")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--all-subjects", action="store_true")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    data_config = config["data"]
    database = DatabaseIndex(data_config["database_root"])
    split = ParticipantSplit.load(data_config["split_manifest"])
    participant_ids = (
        database.participant_ids("ChestSCG") if args.all_subjects else split.test_clean_subjects
    )
    device = resolve_device(args.device)
    cycloscg, _ = load_cycloscg_model(args.config, args.cycloscg_checkpoint, device)
    baseline = None
    if args.baseline_checkpoint:
        baseline, _ = load_baseline_model(args.baseline_config, args.baseline_checkpoint, device)
    phase_length = int(data_config["phase_length"])
    beats_per_window = int(data_config["beats_per_window"])
    rows: list[dict[str, object]] = []
    qc_exclusions: list[dict[str, object]] = []
    example: dict[str, np.ndarray] = {}
    for participant in participant_ids:
        rest_record = database.records("CleanRestSCG", [participant])[0]
        rest_loaded = database.load_record(
            rest_record,
            signal_columns=data_config["signal_columns"],
            ecg_columns=data_config["ecg_columns"],
            include_ecg=True,
        )
        rest_signal, _ = _normalize_recording(np.asarray(rest_loaded["signal"]))
        rest_detection = detect_rpeaks(
            np.asarray(rest_loaded["ecg"]), rest_record.sampling_rate_hz, strict=True
        )
        rest_cycles = normalize_cardiac_cycles(rest_signal, rest_detection.indices, phase_length)
        rows.append(_structural_row(participant, "Rest", "Rest reference", rest_cycles, rest_cycles))
        for walking_record in database.records("RealWalkingContaminatedSCG", [participant]):
            loaded = database.load_record(
                walking_record,
                signal_columns=data_config["signal_columns"],
                ecg_columns=data_config["ecg_columns"],
                include_ecg=True,
            )
            detection = detect_rpeaks(
                np.asarray(loaded["ecg"]), walking_record.sampling_rate_hz, strict=False
            )
            if not detection.qc_passed:
                qc_exclusions.append(
                    {
                        "record_id": walking_record.record_id,
                        "participant": participant,
                        "reasons": list(detection.qc_messages),
                    }
                )
                continue
            walking_signal, _ = _normalize_recording(np.asarray(loaded["signal"]))
            raw_cycles = normalize_cardiac_cycles(
                walking_signal, detection.indices, phase_length
            )
            reconstructed, weights = reconstruct_cycle_matrix(
                cycloscg, raw_cycles, beats_per_window, device
            )
            condition = walking_record.condition
            rows.append(_structural_row(participant, condition, "Raw walking", raw_cycles, rest_cycles))
            rows.append(
                _structural_row(participant, condition, "CycloSCGNet", reconstructed, rest_cycles)
            )
            if baseline is not None:
                with torch.no_grad():
                    baseline_signal = baseline(
                        torch.from_numpy(walking_signal[None]).to(device)
                    )[0].cpu().numpy()
                baseline_cycles = normalize_cardiac_cycles(
                    baseline_signal, detection.indices, phase_length
                )
                rows.append(
                    _structural_row(participant, condition, "Waveform U-Net", baseline_cycles, rest_cycles)
                )
            example_key = "walk1" if "1step" in condition else "walk2"
            if f"{example_key}_raw" not in example:
                example[f"{example_key}_raw"] = raw_cycles
                example[f"{example_key}_reconstructed"] = reconstructed
                example[f"{example_key}_weights"] = weights

    output_root = Path(args.output_root)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(metrics_dir / "walking_structural_metrics.csv", index=False)
    with (metrics_dir / "walking_qc_exclusions.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "records_considered": len(participant_ids) * 2,
                "records_excluded": len(qc_exclusions),
                "exclusions": qc_exclusions,
                "interpretation": "Rest is an individual reference distribution, not synchronized walking ground truth.",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    if {"walk1_raw", "walk1_reconstructed", "walk2_raw", "walk2_reconstructed"}.issubset(example):
        np.savez_compressed(metrics_dir / "walking_example.npz", **example)
        plot_real_walking(
            example["walk1_raw"],
            example["walk1_reconstructed"],
            example["walk2_raw"],
            example["walk2_reconstructed"],
            output_root / "figures" / "real_walking",
        )
        plot_real_walking_waveform_before_after(
            example["walk1_raw"],
            example["walk1_reconstructed"],
            example["walk2_raw"],
            example["walk2_reconstructed"],
            output_root / "figures" / "waveform_before_after",
        )
    print(
        f"Saved {len(frame)} structural rows; excluded {len(qc_exclusions)} walking records by ECG QC"
    )
    return 0
