from __future__ import annotations

import argparse
import copy
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import pandas as pd
import yaml

from cycloscg.utils.config import load_yaml
from cycloscg.visualization.publication import plot_ablation


ABLATION_LABELS = {
    "A2_cycle_aligned": "A2",
    "A3_attention": "A3",
    "A4_consensus": "A4",
    "A5_cycle_loss": "A5",
    "A6_covariance_loss": "A6",
    "A7_singular_loss": "A7",
    "Full": "Full",
}


def _summarize_evaluations(
    evaluation_root: Path,
    variants: Sequence[str],
    output_root: Path,
) -> pd.DataFrame:
    """Aggregate variants evaluated on the same deterministic recipes."""
    metrics = (
        "rmse",
        "pearson_r",
        "snr_improvement_db",
        "cycle_coherence_error",
        "phase_gram_distance",
        "singular_spectrum_distance",
    )
    method_specs: list[tuple[str, str, pd.DataFrame]] = []
    for index, variant in enumerate(variants):
        path = evaluation_root / variant / "metrics" / "synthetic_per_sample.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluated ablation table: {path}")
        frame = pd.read_csv(path)
        if index == 0:
            method_specs.extend(
                (
                    ("A0", "Raw", frame),
                    ("A1", "Waveform U-Net", frame),
                )
            )
        method_specs.append((ABLATION_LABELS.get(variant, variant), "CycloSCGNet", frame))

    rows: list[dict[str, object]] = []
    by_snr_rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    for label, method, frame in method_specs:
        selected = frame.loc[frame["method"] == method]
        if selected.empty:
            raise ValueError(f"Method {method!r} is absent for ablation {label}")
        rows.append(
            {
                "variant": label,
                "method": method,
                **{metric: float(selected[metric].mean()) for metric in metrics},
            }
        )
        for snr_db, group in selected.groupby("snr_db", sort=True):
            by_snr_rows.append(
                {
                    "variant": label,
                    "method": method,
                    "snr_db": float(snr_db),
                    **{metric: float(group[metric].mean()) for metric in metrics},
                }
            )

    for variant in variants:
        path = evaluation_root / variant / "metrics" / "identity_preservation.csv"
        if path.exists():
            frame = pd.read_csv(path)
            identity_rows.append(
                {
                    "variant": ABLATION_LABELS.get(variant, variant),
                    **{
                        metric: float(frame[metric].mean())
                        for metric in ("rmse", "pearson_r", "prd_percent", "output_snr_db")
                    },
                }
            )

    summary = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(metrics_dir / "ablation_summary.csv", index=False)
    pd.DataFrame(by_snr_rows).to_csv(metrics_dir / "ablation_summary_by_snr.csv", index=False)
    pd.DataFrame(identity_rows).to_csv(metrics_dir / "ablation_identity_summary.csv", index=False)
    order = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "Full"]
    ordered = summary.set_index("variant").reindex(order)
    for metric in metrics:
        plot_ablation(
            order,
            ordered[metric].to_numpy(),
            output_root / "figures" / f"ablation_{metric}",
            metric,
        )
    return summary


def ablation_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare/run cumulative CycloSCG ablations")
    parser.add_argument("--matrix", default="configs/ablation_matrix.yaml")
    parser.add_argument("--output-root", default="results/ablation")
    parser.add_argument("--run", action="store_true", help="Train all A2-Full variants")
    parser.add_argument("--summary-csv", help="Optional evaluated table with variant and metric columns")
    parser.add_argument("--metric", default="rmse")
    parser.add_argument(
        "--evaluation-root",
        help="Directory containing <variant>/metrics/synthetic_per_sample.csv tables",
    )
    args = parser.parse_args(argv)
    matrix = load_yaml(args.matrix)
    base = load_yaml(matrix["base_config"])
    output_root = Path(args.output_root)
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    generated: list[tuple[str, Path]] = []
    for name, overrides in matrix["variants"].items():
        config = copy.deepcopy(base)
        config["project"]["output_root"] = str(output_root / name)
        config["model"].update(overrides["model"])
        config["loss"]["weights"] = overrides["loss_weights"]
        path = config_dir / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        generated.append((name, path))
    if args.run:
        for name, path in generated:
            print(f"Training {name} from {path}")
            subprocess.run(
                [sys.executable, "scripts/train_cycloscg.py", "--config", str(path)],
                check=True,
            )
    if args.summary_csv:
        frame = pd.read_csv(args.summary_csv)
        required = {"variant", args.metric}
        if not required.issubset(frame.columns):
            raise ValueError(f"Summary must contain {sorted(required)}")
        order = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "Full"]
        values = [frame.loc[frame["variant"] == name, args.metric].mean() for name in order]
        plot_ablation(order, values, output_root / f"ablation_{args.metric}", args.metric)
    if args.evaluation_root:
        summary = _summarize_evaluations(
            Path(args.evaluation_root),
            [name for name, _ in generated],
            output_root,
        )
        print(f"Aggregated {len(summary)} ablation methods")
    print(f"Prepared {len(generated)} ablation configurations in {config_dir}")
    return 0
