from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cycloscg.visualization.publication import (
    plot_real_walking_waveform_before_after,
    plot_synthetic_single_cycle_before_after,
    plot_synthetic_waveform_before_after,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot direct contaminated-versus-denoised SCG waveforms")
    parser.add_argument(
        "--synthetic-example",
        default="results/gpu_evaluation_v2/metrics/synthetic_example.npz",
    )
    parser.add_argument(
        "--walking-example",
        default="results/gpu_walking_v2/metrics/walking_example.npz",
    )
    parser.add_argument("--synthetic-output", default="results/gpu_evaluation_v2/figures/waveform_before_after")
    parser.add_argument("--walking-output", default="results/gpu_walking_v2/figures/waveform_before_after")
    args = parser.parse_args()

    synthetic = np.load(args.synthetic_example)
    selection_note = None
    if {"selection_method", "selection_rmse", "group_median_rmse"}.issubset(synthetic.files):
        selection_note = (
            f"Representative held-out sample: {str(synthetic['selection_method'])} RMSE nearest "
            f"the severe-SNR group median ({float(synthetic['selection_rmse']):.3f} vs "
            f"{float(synthetic['group_median_rmse']):.3f})."
        )
    synthetic_paths = plot_synthetic_waveform_before_after(
        synthetic["clean_cycles"],
        synthetic["noisy_cycles"],
        synthetic["cyclo_cycles"],
        Path(args.synthetic_output),
        selection_note=selection_note,
    )
    single_cycle_paths = plot_synthetic_single_cycle_before_after(
        synthetic["clean_cycles"],
        synthetic["noisy_cycles"],
        synthetic["cyclo_cycles"],
        Path(args.synthetic_output).with_name(
            f"{Path(args.synthetic_output).name}_single_cycle"
        ),
        selection_note=selection_note,
    )
    walking = np.load(args.walking_example)
    walking_paths = plot_real_walking_waveform_before_after(
        walking["walk1_raw"],
        walking["walk1_reconstructed"],
        walking["walk2_raw"],
        walking["walk2_reconstructed"],
        Path(args.walking_output),
    )
    print(
        f"Saved {len(synthetic_paths) + len(single_cycle_paths)} synthetic and "
        f"{len(walking_paths)} walking figure files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
