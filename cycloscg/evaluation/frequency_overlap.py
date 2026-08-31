from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr

from cycloscg.data.records import DatabaseIndex
from cycloscg.data.rpeak import detect_rpeaks
from cycloscg.data.split import ParticipantSplit
from cycloscg.utils.config import load_yaml
from cycloscg.visualization.publication import plot_frequency_overlap


def estimate_cadence(
    acceleration: np.ndarray,
    sampling_rate_hz: float,
    nominal_rate_hz: float | None,
) -> float:
    values = np.asarray(acceleration, dtype=np.float64)
    frequencies, power = welch(
        values - np.mean(values),
        fs=sampling_rate_hz,
        nperseg=min(len(values), int(8 * sampling_rate_hz)),
    )
    if nominal_rate_hz is None:
        low, high = 0.5, 3.0
    else:
        low, high = max(0.4, nominal_rate_hz - 0.45), min(3.2, nominal_rate_hz + 0.45)
    mask = (frequencies >= low) & (frequencies <= high)
    if not np.any(mask):
        raise ValueError("Cadence search band contains no frequency bins")
    return float(frequencies[mask][np.argmax(power[mask])])


def harmonic_proximity(cardiac_hz: float, cadence_hz: float) -> tuple[float, int, int]:
    candidates = [
        (abs(gait_order * cadence_hz - cardiac_order * cardiac_hz), gait_order, cardiac_order)
        for gait_order in range(1, 5)
        for cardiac_order in range(1, 4)
    ]
    return min(candidates, key=lambda values: values[0])


def frequency_overlap_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gait-heart harmonic overlap mechanism analysis")
    parser.add_argument("--config", default="configs/cycloscg.yaml")
    parser.add_argument("--walking-metrics")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--all-subjects", action="store_true")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    data_config = config["data"]
    database = DatabaseIndex(data_config["database_root"])
    split = ParticipantSplit.load(data_config["split_manifest"])
    participant_ids = (
        database.participant_ids("ChestSCG") if args.all_subjects else split.test_clean_subjects
    )
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for record in database.records("RealWalkingContaminatedSCG", participant_ids):
        loaded = database.load_record(
            record,
            signal_columns=("accel_z_centered_m_s2", "accel_z_m_s2"),
            ecg_columns=data_config["ecg_columns"],
            include_ecg=True,
        )
        detection = detect_rpeaks(np.asarray(loaded["ecg"]), record.sampling_rate_hz)
        if not detection.qc_passed:
            exclusions.append({"record_id": record.record_id, "reasons": list(detection.qc_messages)})
            continue
        cardiac_hz = 1.0 / float(np.median(detection.rr_intervals_s))
        cadence_hz = estimate_cadence(
            np.asarray(loaded["signal"]), record.sampling_rate_hz, record.known_step_rate_hz
        )
        gap, gait_order, cardiac_order = harmonic_proximity(cardiac_hz, cadence_hz)
        rows.append(
            {
                "participant": record.participant_id,
                "condition": record.condition,
                "heart_rate_bpm": detection.heart_rate_bpm,
                "cardiac_frequency_hz": cardiac_hz,
                "nominal_cadence_hz": record.known_step_rate_hz,
                "cadence_hz": cadence_hz,
                "minimum_harmonic_gap_hz": gap,
                "closest_gait_harmonic": gait_order,
                "closest_cardiac_harmonic": cardiac_order,
                "overlap_score": float(np.exp(-gap / 0.15)),
            }
        )
    frame = pd.DataFrame(rows)
    if args.walking_metrics:
        metrics = pd.read_csv(args.walking_metrics)
        pivot = metrics.pivot_table(
            index=["participant", "condition"],
            columns="method",
            values="phase_gram_distance",
            aggfunc="mean",
        ).reset_index()
        required = {"Raw walking", "CycloSCGNet", "Waveform U-Net"}
        if required.issubset(pivot.columns):
            pivot["cycloscg_gain_vs_raw"] = pivot["Raw walking"] - pivot["CycloSCGNet"]
            pivot["baseline_gain_vs_raw"] = pivot["Raw walking"] - pivot["Waveform U-Net"]
            pivot["cycloscg_advantage_over_baseline"] = (
                pivot["cycloscg_gain_vs_raw"] - pivot["baseline_gain_vs_raw"]
            )
            frame = frame.merge(pivot, on=["participant", "condition"], how="left")
    output_root = Path(args.output_root)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(metrics_dir / "gait_heart_overlap.csv", index=False)
    statistics: dict[str, object] = {
        "records_included": len(frame),
        "records_excluded_by_ecg_qc": len(exclusions),
        "exclusions": exclusions,
    }
    if "cycloscg_advantage_over_baseline" in frame.columns:
        valid = frame[["minimum_harmonic_gap_hz", "cycloscg_advantage_over_baseline"]].dropna()
        if len(valid) >= 3:
            result = spearmanr(
                valid["minimum_harmonic_gap_hz"], valid["cycloscg_advantage_over_baseline"]
            )
            statistics["spearman_gap_vs_advantage"] = {
                "rho": float(result.statistic),
                "pvalue": float(result.pvalue),
                "n": len(valid),
            }
    with (metrics_dir / "gait_heart_overlap_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, ensure_ascii=False, indent=2)
    plot_frequency_overlap(frame, output_root / "figures" / "gait_heart_overlap")
    print(f"Saved {len(frame)} gait-heart overlap records; QC exclusions={len(exclusions)}")
    return 0

