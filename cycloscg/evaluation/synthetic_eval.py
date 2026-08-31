from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from cycloscg.data.cardiac_phase import normalize_aligned_pair, normalize_cardiac_cycles
from cycloscg.data.mixing import extract_with_wrap, mix_at_snr
from cycloscg.data.records import DatabaseIndex
from cycloscg.data.rpeak import detect_rpeaks
from cycloscg.data.split import ParticipantSplit
from cycloscg.evaluation.common import load_baseline_model, load_cycloscg_model
from cycloscg.metrics import cycle_structure_metrics, cyclostationary_metrics, waveform_metrics
from cycloscg.utils.config import load_yaml
from cycloscg.utils.seed import resolve_device, seed_everything
from cycloscg.visualization.publication import (
    plot_correlation_matrices,
    plot_cycle_matrices,
    plot_metric_curves,
    plot_reliability_weights,
    plot_severe_case,
    plot_singular_spectra,
    plot_synthetic_components,
    plot_synthetic_single_cycle_before_after,
    plot_synthetic_waveform_before_after,
)


def _metrics(estimate: np.ndarray, target: np.ndarray, noisy: np.ndarray) -> dict[str, float]:
    return {
        **waveform_metrics(estimate, target, noisy),
        **cycle_structure_metrics(estimate, target),
        **cyclostationary_metrics(estimate, target),
    }


def synthetic_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Held-out synthetic contamination benchmark")
    parser.add_argument("--config", default="configs/cycloscg.yaml")
    parser.add_argument("--baseline-config", default="configs/baseline.yaml")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--cycloscg-checkpoint")
    parser.add_argument("--samples-per-snr", type=int)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    data_config = config["data"]
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    device = resolve_device(args.device)
    database = DatabaseIndex(data_config["database_root"])
    split = ParticipantSplit.load(data_config["split_manifest"])
    clean_records = database.records("CleanRestSCG", split.test_clean_subjects)
    noise_records = database.records("MotionNoiseProxy", split.test_noise_subjects)
    if not clean_records or not noise_records:
        raise RuntimeError("Held-out clean/noise records are empty")
    baseline = None
    if args.baseline_checkpoint:
        baseline, _ = load_baseline_model(args.baseline_config, args.baseline_checkpoint, device)
    cycloscg = None
    if args.cycloscg_checkpoint:
        cycloscg, _ = load_cycloscg_model(args.config, args.cycloscg_checkpoint, device)
    beats_per_window = int(data_config["beats_per_window"])
    phase_length = int(data_config["phase_length"])
    sample_count = args.samples_per_snr or int(data_config.get("test_samples_per_snr", 128))
    snr_levels = [float(value) for value in data_config["benchmark_snr_db"]]
    rng = np.random.default_rng(seed + 20_000)
    cache: dict[str, dict[str, object]] = {}

    def loaded(record, ecg=False):
        key = f"{record.record_id}:{int(ecg)}"
        if key not in cache:
            cache[key] = database.load_record(
                record,
                signal_columns=data_config["signal_columns"],
                ecg_columns=data_config["ecg_columns"],
                include_ecg=ecg,
            )
        return cache[key]

    rows: list[dict[str, object]] = []
    example_candidates: list[tuple[float, dict[str, object]]] = []
    for snr_db in snr_levels:
        for sample_index in range(sample_count):
            clean_record = clean_records[int(rng.integers(len(clean_records)))]
            noise_record = noise_records[int(rng.integers(len(noise_records)))]
            clean_loaded = loaded(clean_record, ecg=True)
            clean_full = np.asarray(clean_loaded["signal"], dtype=np.float32)
            detection = detect_rpeaks(
                np.asarray(clean_loaded["ecg"]), clean_record.sampling_rate_hz, strict=True
            )
            if len(detection.indices) < beats_per_window + 1:
                raise RuntimeError(f"Too few clean beats in {clean_record.record_id}")
            beat_start = int(rng.integers(0, len(detection.indices) - beats_per_window))
            left = int(detection.indices[beat_start])
            right = int(detection.indices[beat_start + beats_per_window])
            clean_time = clean_full[left:right]
            noise_full = np.asarray(loaded(noise_record)["signal"], dtype=np.float32)
            noise_start = int(rng.integers(len(noise_full)))
            noise_time = extract_with_wrap(noise_full, noise_start, len(clean_time))
            polarity = float(rng.choice([-1.0, 1.0])) if data_config["random_polarity"] else 1.0
            noisy_time, scale = mix_at_snr(clean_time, noise_time, snr_db, polarity)
            normalization = max(float(np.sqrt(np.mean(clean_time.astype(np.float64) ** 2))), 1e-6)
            clean_time_normalized = clean_time / normalization
            noisy_time_normalized = noisy_time / normalization
            local_peaks = detection.indices[beat_start : beat_start + beats_per_window + 1] - left
            noisy_cycles, clean_cycles = normalize_aligned_pair(
                noisy_time_normalized, clean_time_normalized, local_peaks, phase_length
            )
            estimates: list[tuple[str, np.ndarray]] = [("Raw", noisy_cycles)]
            baseline_cycles = None
            if baseline is not None:
                with torch.no_grad():
                    tensor = torch.from_numpy(noisy_time_normalized[None].astype(np.float32)).to(device)
                    baseline_time = baseline(tensor)[0].cpu().numpy()
                baseline_cycles = normalize_cardiac_cycles(baseline_time, local_peaks, phase_length)
                estimates.append(("Waveform U-Net", baseline_cycles))
            reliability = None
            cyclo_cycles = None
            if cycloscg is not None:
                with torch.no_grad():
                    output = cycloscg(
                        torch.from_numpy(noisy_cycles[None]).to(device), return_aux=True
                    )
                cyclo_cycles = output.reconstruction[0].cpu().numpy()
                reliability = output.reliability_weights[0].cpu().numpy()
                estimates.append(("CycloSCGNet", cyclo_cycles))
            metrics_by_method: dict[str, dict[str, float]] = {}
            for method, estimate in estimates:
                evaluated = _metrics(estimate, clean_cycles, noisy_cycles)
                metrics_by_method[method] = evaluated
                rows.append(
                    {
                        "sample_index": sample_index,
                        "snr_db": snr_db,
                        "method": method,
                        "clean_subject": clean_record.participant_id,
                        "noise_subject": noise_record.participant_id,
                        **evaluated,
                    }
                )
            if snr_db == min(snr_levels):
                selection_method = "CycloSCGNet" if "CycloSCGNet" in metrics_by_method else "Raw"
                selection_rmse = float(metrics_by_method[selection_method]["rmse"])
                candidate: dict[str, object] = {
                    "clean_time": clean_time_normalized,
                    "noise_time": polarity * scale * noise_time / normalization,
                    "noisy_time": noisy_time_normalized,
                    "clean_cycles": clean_cycles,
                    "noisy_cycles": noisy_cycles,
                    "baseline_cycles": baseline_cycles if baseline_cycles is not None else noisy_cycles,
                    "cyclo_cycles": cyclo_cycles if cyclo_cycles is not None else noisy_cycles,
                    "reliability": reliability if reliability is not None else np.full(beats_per_window, 1 / beats_per_window),
                    "sampling_rate_hz": clean_record.sampling_rate_hz,
                    "sample_index": sample_index,
                    "clean_subject": clean_record.participant_id,
                    "noise_subject": noise_record.participant_id,
                    "selection_method": selection_method,
                    "selection_rmse": selection_rmse,
                }
                example_candidates.append((selection_rmse, candidate))

    example: dict[str, object] | None = None
    if example_candidates:
        median_rmse = float(np.median([rmse for rmse, _ in example_candidates]))
        _, example = min(example_candidates, key=lambda item: abs(item[0] - median_rmse))
        example["group_median_rmse"] = median_rmse

    output_root = Path(args.output_root)
    metrics_dir = output_root / "metrics"
    figures_dir = output_root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(metrics_dir / "synthetic_per_sample.csv", index=False)
    metric_columns = [
        column
        for column in frame.columns
        if column not in {"sample_index", "snr_db", "method", "clean_subject", "noise_subject"}
    ]
    summary_rows: list[dict[str, object]] = []
    for (method, snr_db), group in frame.groupby(["method", "snr_db"]):
        for metric in metric_columns:
            values = group[metric].dropna()
            summary_rows.append(
                {
                    "method": method,
                    "snr_db": snr_db,
                    "metric": metric,
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "n": len(values),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(metrics_dir / "synthetic_summary.csv", index=False)
    if cycloscg is not None:
        identity_rows: list[dict[str, object]] = []
        for clean_record in clean_records:
            clean_loaded = loaded(clean_record, ecg=True)
            clean_signal = np.asarray(clean_loaded["signal"], dtype=np.float32)
            detection = detect_rpeaks(
                np.asarray(clean_loaded["ecg"]), clean_record.sampling_rate_hz, strict=True
            )
            clean_cycles = normalize_cardiac_cycles(clean_signal, detection.indices, phase_length)
            if len(clean_cycles) < beats_per_window:
                continue
            clean_window = clean_cycles[:beats_per_window]
            scale = max(float(np.sqrt(np.mean(clean_window.astype(np.float64) ** 2))), 1e-6)
            clean_window = clean_window / scale
            with torch.no_grad():
                identity_output = cycloscg(
                    torch.from_numpy(clean_window[None]).to(device), return_aux=True
                ).reconstruction[0].cpu().numpy()
            identity_rows.append(
                {
                    "participant": clean_record.participant_id,
                    **waveform_metrics(identity_output, clean_window),
                }
            )
        pd.DataFrame(identity_rows).to_csv(
            metrics_dir / "identity_preservation.csv", index=False
        )
    for metric in ("rmse", "pearson_r", "snr_improvement_db"):
        plot_metric_curves(
            summary[summary["metric"] == metric],
            metric,
            figures_dir / f"synthetic_{metric}",
        )
    if example is not None:
        np.savez_compressed(metrics_dir / "synthetic_example.npz", **example)
        plot_synthetic_components(
            example["clean_time"],
            example["noise_time"],
            example["noisy_time"],
            float(example["sampling_rate_hz"]),
            figures_dir / "synthetic_components",
        )
        plot_cycle_matrices(
            example["clean_cycles"], example["noisy_cycles"], example["cyclo_cycles"], figures_dir / "cycle_matrices"
        )
        plot_reliability_weights(example["reliability"], figures_dir / "reliability_weights")
        plot_correlation_matrices(
            example["clean_cycles"], example["noisy_cycles"], example["cyclo_cycles"], figures_dir / "cycle_correlations"
        )
        plot_singular_spectra(
            example["clean_cycles"], example["noisy_cycles"], example["cyclo_cycles"], figures_dir / "singular_spectra"
        )
        plot_severe_case(
            example["noisy_cycles"][0],
            example["clean_cycles"][0],
            example["baseline_cycles"][0],
            example["cyclo_cycles"][0],
            figures_dir / "severe_case",
        )
        plot_synthetic_waveform_before_after(
            example["clean_cycles"],
            example["noisy_cycles"],
            example["cyclo_cycles"],
            figures_dir / "waveform_before_after",
            selection_note=(
                f"Representative held-out sample: {example['selection_method']} RMSE nearest "
                f"the {min(snr_levels):.0f} dB group median "
                f"({float(example['selection_rmse']):.3f} vs {float(example['group_median_rmse']):.3f})."
            ),
        )
        plot_synthetic_single_cycle_before_after(
            example["clean_cycles"],
            example["noisy_cycles"],
            example["cyclo_cycles"],
            figures_dir / "waveform_before_after_single_cycle",
            selection_note=(
                f"Held-out sample RMSE nearest the {min(snr_levels):.0f} dB group median "
                f"({float(example['selection_rmse']):.3f} vs {float(example['group_median_rmse']):.3f})."
            ),
        )
    print(f"Saved {len(frame)} rows to {metrics_dir / 'synthetic_per_sample.csv'}")
    return 0
