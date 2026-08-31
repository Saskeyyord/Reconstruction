from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cycloscg.metrics.cycle_metrics import pairwise_correlation_numpy
from cycloscg.metrics.cyclostationary_metrics import singular_spectrum_numpy


DEFAULT_PALETTE = {
    "clean": "#0F4D92",
    "raw": "#767676",
    "noise": "#B64342",
    "baseline": "#7884B4",
    "cycloscg": "#42949E",
    "accent": "#9A4D8E",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_publication_figure(fig: mpl.figure.Figure, output: str | Path) -> list[Path]:
    base = Path(output).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    svg = base.with_suffix(".svg")
    pdf = base.with_suffix(".pdf")
    tiff = base.with_suffix(".tiff")
    png = base.with_suffix(".png")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(
        tiff,
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(png, dpi=300, bbox_inches="tight")
    saved = [svg, pdf, tiff, png]
    plt.close(fig)
    return saved


def _panel_label(axis: mpl.axes.Axes, label: str) -> None:
    axis.text(-0.10, 1.03, label, transform=axis.transAxes, fontweight="bold", va="bottom")


def plot_synthetic_components(
    clean: np.ndarray,
    noise: np.ndarray,
    noisy: np.ndarray,
    sampling_rate_hz: float,
    output: str | Path,
    palette: Mapping[str, str] = DEFAULT_PALETTE,
) -> list[Path]:
    apply_publication_style()
    time = np.arange(len(clean)) / sampling_rate_hz
    fig, axes = plt.subplots(3, 1, figsize=(7.09, 4.5), sharex=True, constrained_layout=True)
    for axis, values, label, color, panel in zip(
        axes,
        (clean, noise, noisy),
        ("Clean resting SCG", "Independent motion-noise proxy", "Synthetic contaminated SCG"),
        (palette["clean"], palette["noise"], palette["raw"]),
        ("a", "b", "c"),
    ):
        axis.plot(time, values, color=color, linewidth=0.7)
        axis.set_ylabel("SCG (a.u.)")
        axis.set_title(label, loc="left")
        _panel_label(axis, panel)
    axes[-1].set_xlabel("Time (s)")
    return save_publication_figure(fig, output)


def plot_cycle_matrices(
    clean: np.ndarray,
    noisy: np.ndarray,
    reconstructed: np.ndarray,
    output: str | Path,
) -> list[Path]:
    apply_publication_style()
    vmax = np.percentile(np.abs(np.concatenate([clean.ravel(), noisy.ravel(), reconstructed.ravel()])), 99)
    fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.5), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for axis, matrix, title, panel in zip(
        axes,
        (clean, noisy, reconstructed),
        ("Clean target", "Synthetic noisy", "CycloSCGNet"),
        ("a", "b", "c"),
    ):
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("Cardiac phase")
        axis.set_yticks(np.arange(matrix.shape[0]))
        _panel_label(axis, panel)
    axes[0].set_ylabel("Cycle index")
    fig.colorbar(image, ax=axes, label="Normalized amplitude", shrink=0.8)
    return save_publication_figure(fig, output)


def plot_reliability_weights(
    weights: np.ndarray,
    output: str | Path,
    palette: Mapping[str, str] = DEFAULT_PALETTE,
) -> list[Path]:
    apply_publication_style()
    values = np.asarray(weights).reshape(-1)
    fig, axis = plt.subplots(figsize=(7.09, 2.3), constrained_layout=True)
    axis.bar(np.arange(1, len(values) + 1), values, color=palette["cycloscg"], edgecolor="white")
    axis.axhline(1.0 / len(values), color=palette["raw"], linestyle="--", linewidth=1, label="Uniform weight")
    axis.set(xlabel="Cycle index", ylabel="Reliability weight", title="Learned beat reliability")
    axis.legend()
    return save_publication_figure(fig, output)


def plot_severe_case(
    noisy: np.ndarray,
    clean: np.ndarray,
    baseline: np.ndarray,
    cycloscg: np.ndarray,
    output: str | Path,
    palette: Mapping[str, str] = DEFAULT_PALETTE,
) -> list[Path]:
    apply_publication_style()
    phase = np.linspace(0.0, 1.0, len(clean), endpoint=False)
    fig, axes = plt.subplots(2, 1, figsize=(7.09, 4.0), sharex=True, constrained_layout=True)
    axes[0].plot(phase, clean, color=palette["clean"], linewidth=1.2, label="Clean target")
    axes[0].plot(phase, noisy, color=palette["raw"], linewidth=0.8, alpha=0.8, label="Noisy input")
    axes[1].plot(phase, clean, color=palette["clean"], linewidth=1.2, label="Clean target")
    axes[1].plot(phase, baseline, color=palette["baseline"], linewidth=0.9, label="Waveform U-Net")
    axes[1].plot(phase, cycloscg, color=palette["cycloscg"], linewidth=0.9, label="CycloSCGNet")
    for axis, panel in zip(axes, ("a", "b")):
        axis.set_ylabel("SCG (a.u.)")
        axis.legend(ncols=3, loc="upper right")
        _panel_label(axis, panel)
    axes[-1].set_xlabel("Normalized cardiac phase")
    return save_publication_figure(fig, output)


def plot_metric_curves(
    summary,
    metric: str,
    output: str | Path,
    palette: Mapping[str, str] = DEFAULT_PALETTE,
) -> list[Path]:
    """Plot mean±SD from a table with method/snr_db/mean/std/n columns."""
    apply_publication_style()
    fig, axis = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    for method, frame in summary.groupby("method"):
        frame = frame.sort_values("snr_db")
        method_key = {
            "cycloscgnet": "cycloscg",
            "waveform u-net": "baseline",
        }.get(str(method).lower(), str(method).lower())
        color = palette.get(method_key, palette["accent"])
        axis.plot(frame["snr_db"], frame["mean"], marker="o", color=color, label=method)
        axis.fill_between(
            frame["snr_db"],
            frame["mean"] - frame["std"],
            frame["mean"] + frame["std"],
            color=color,
            alpha=0.15,
        )
    axis.set(xlabel="Input SNR (dB)", ylabel=metric, title=f"{metric} across contamination levels")
    axis.legend()
    return save_publication_figure(fig, output)


def plot_ablation(
    labels: Sequence[str],
    values: Sequence[float],
    output: str | Path,
    ylabel: str,
    palette: Mapping[str, str] = DEFAULT_PALETTE,
) -> list[Path]:
    apply_publication_style()
    labels = list(labels)
    values = np.asarray(values, dtype=float)
    colors = [
        palette["raw"]
        if label == "A0"
        else palette["baseline"]
        if label == "A1"
        else palette["cycloscg"]
        if label == "Full"
        else mpl.colors.to_rgba(palette["baseline"], 0.72)
        for label in labels
    ]
    finite_sorted = np.sort(values[np.isfinite(values)])[::-1]
    needs_zoom = len(finite_sorted) > 1 and finite_sorted[0] > 2.2 * finite_sorted[1]
    if needs_zoom:
        # A separate, explicitly labelled zoom panel avoids hiding differences
        # among trainable methods while retaining the raw-input reference.
        fig, (raw_axis, axis) = plt.subplots(
            1,
            2,
            figsize=(7.09, 2.8),
            constrained_layout=True,
            gridspec_kw={"width_ratios": (0.9, 4.6)},
        )
        raw_bars = raw_axis.bar(labels[:1], values[:1], color=colors[:1], width=0.62)
        raw_axis.bar_label(raw_bars, fmt="%.2f", padding=2, fontsize=6)
        raw_axis.set(ylabel=ylabel, title="Raw input (full scale)")
        raw_axis.set_ylim(0.0, max(values[0] * 1.13, 1e-6))
        bars = axis.bar(labels[1:], values[1:], color=colors[1:], width=0.72)
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=6)
        axis.set(title="Trainable methods (zoom)")
        axis.set_ylim(0.0, max(np.nanmax(values[1:]) * 1.16, 1e-6))
        axis.tick_params(axis="x", rotation=30)
    else:
        fig, axis = plt.subplots(figsize=(7.09, 2.8), constrained_layout=True)
        bars = axis.bar(labels, values, color=colors, width=0.72)
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=6)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=30)
        axis.set_ylim(0.0, max(np.nanmax(values) * 1.16, 1e-6))
    fig.suptitle("Controlled cumulative ablation", fontsize=8)
    return save_publication_figure(fig, output)


def plot_synthetic_waveform_before_after(
    clean_cycles: np.ndarray,
    noisy_cycles: np.ndarray,
    reconstructed_cycles: np.ndarray,
    output: str | Path,
    selection_note: str | None = None,
) -> list[Path]:
    """Show the contaminated and reconstructed waveforms on directly comparable axes."""
    clean = np.asarray(clean_cycles, dtype=float)
    noisy = np.asarray(noisy_cycles, dtype=float)
    reconstructed = np.asarray(reconstructed_cycles, dtype=float)
    if clean.ndim != 2 or clean.shape != noisy.shape or clean.shape != reconstructed.shape:
        raise ValueError("clean, noisy, and reconstructed cycles must share shape [K, L]")
    cycles, phase_length = clean.shape
    x = np.arange(cycles * phase_length, dtype=float) / phase_length
    clean_flat = clean.reshape(-1)
    noisy_flat = noisy.reshape(-1)
    reconstructed_flat = reconstructed.reshape(-1)
    injected_artifact = noisy_flat - clean_flat
    removed_component = noisy_flat - reconstructed_flat
    input_error_power = max(float(np.mean(injected_artifact**2)), 1e-12)
    output_error_power = max(float(np.mean((reconstructed_flat - clean_flat) ** 2)), 1e-12)
    clean_power = max(float(np.mean(clean_flat**2)), 1e-12)
    input_snr = 10.0 * np.log10(clean_power / input_error_power)
    output_snr = 10.0 * np.log10(clean_power / output_error_power)
    pearson = float(np.corrcoef(clean_flat, reconstructed_flat)[0, 1])

    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 4.6), sharex=True, constrained_layout=True)
    top_limit = 1.05 * max(float(np.max(np.abs(noisy_flat))), 1e-6)
    morphology_limit = 1.08 * max(
        float(np.max(np.abs(clean_flat))),
        float(np.max(np.abs(reconstructed_flat))),
        1e-6,
    )
    artifact_limit = 1.05 * max(
        float(np.max(np.abs(injected_artifact))),
        float(np.max(np.abs(removed_component))),
        1e-6,
    )
    axes[0, 0].plot(x, noisy_flat, color=DEFAULT_PALETTE["raw"], lw=0.65, label="Contaminated input")
    axes[0, 0].plot(x, clean_flat, color=DEFAULT_PALETTE["clean"], lw=0.9, label="Clean target")
    axes[0, 0].set(title="Before denoising", ylabel="Normalized SCG")
    axes[0, 0].set_ylim(-top_limit, top_limit)
    axes[0, 0].legend(loc="upper right", ncols=2)

    axes[0, 1].plot(x, noisy_flat, color=DEFAULT_PALETTE["raw"], lw=0.65, label="Contaminated input")
    axes[0, 1].plot(
        x,
        reconstructed_flat,
        color=DEFAULT_PALETTE["cycloscg"],
        lw=0.9,
        label="CycloSCGNet output",
    )
    axes[0, 1].set(title="Direct before–after comparison")
    axes[0, 1].set_ylim(-top_limit, top_limit)
    axes[0, 1].legend(loc="upper right", ncols=2)

    axes[1, 0].plot(x, clean_flat, color=DEFAULT_PALETTE["clean"], lw=0.9, label="Clean target")
    axes[1, 0].plot(
        x,
        reconstructed_flat,
        color=DEFAULT_PALETTE["cycloscg"],
        lw=0.9,
        label="CycloSCGNet output",
    )
    axes[1, 0].set(
        title="Recovered morphology (zoomed amplitude)",
        xlabel="Concatenated cardiac cycles (beat index)",
        ylabel="Normalized SCG",
    )
    axes[1, 0].set_ylim(-morphology_limit, morphology_limit)
    axes[1, 0].legend(loc="upper right", ncols=2)
    axes[1, 0].text(
        0.01,
        0.04,
        f"Input SNR {input_snr:.1f} dB  →  Output SNR {output_snr:.1f} dB\n"
        f"Pearson r = {pearson:.3f}",
        transform=axes[1, 0].transAxes,
        fontsize=6,
        va="bottom",
    )

    axes[1, 1].plot(
        x,
        injected_artifact,
        color=DEFAULT_PALETTE["noise"],
        lw=0.65,
        alpha=0.75,
        label="Known injected artifact",
    )
    axes[1, 1].plot(
        x,
        removed_component,
        color=DEFAULT_PALETTE["accent"],
        lw=0.75,
        label="Model-removed component",
    )
    axes[1, 1].set(
        title="Removed component versus injected artifact",
        xlabel="Concatenated cardiac cycles (beat index)",
    )
    axes[1, 1].set_ylim(-artifact_limit, artifact_limit)
    axes[1, 1].legend(loc="upper right", ncols=2)
    for label, axis in zip(("a", "b", "c", "d"), axes.flat):
        _panel_label(axis, label)
        for boundary in range(2, cycles, 2):
            axis.axvline(boundary, color="#D8D8D8", lw=0.35, zorder=0)
    fig.suptitle(
        f"Severe synthetic contamination: waveform before and after reconstruction (K={cycles})",
        fontsize=8,
    )
    if selection_note:
        fig.text(0.5, -0.01, selection_note, ha="center", fontsize=6)
    return save_publication_figure(fig, output)


def plot_synthetic_single_cycle_before_after(
    clean_cycles: np.ndarray,
    noisy_cycles: np.ndarray,
    reconstructed_cycles: np.ndarray,
    output: str | Path,
    selection_note: str | None = None,
) -> list[Path]:
    """Readable single-beat before/after view without selecting the best beat."""
    clean = np.asarray(clean_cycles, dtype=float)
    noisy = np.asarray(noisy_cycles, dtype=float)
    reconstructed = np.asarray(reconstructed_cycles, dtype=float)
    if clean.ndim != 2 or clean.shape != noisy.shape or clean.shape != reconstructed.shape:
        raise ValueError("clean, noisy, and reconstructed cycles must share shape [K, L]")
    per_cycle_rmse = np.sqrt(np.mean((reconstructed - clean) ** 2, axis=1))
    median_rmse = float(np.median(per_cycle_rmse))
    cycle_index = int(np.argmin(np.abs(per_cycle_rmse - median_rmse)))
    clean_cycle = clean[cycle_index]
    noisy_cycle = noisy[cycle_index]
    reconstructed_cycle = reconstructed[cycle_index]
    phase = np.linspace(0.0, 1.0, clean.shape[1], endpoint=False)
    clean_power = max(float(np.mean(clean_cycle**2)), 1e-12)
    input_error = max(float(np.mean((noisy_cycle - clean_cycle) ** 2)), 1e-12)
    output_error = max(float(np.mean((reconstructed_cycle - clean_cycle) ** 2)), 1e-12)
    input_snr = 10.0 * np.log10(clean_power / input_error)
    output_snr = 10.0 * np.log10(clean_power / output_error)
    pearson = float(np.corrcoef(clean_cycle, reconstructed_cycle)[0, 1])
    full_limit = 1.06 * max(float(np.max(np.abs(noisy_cycle))), 1e-6)
    zoom_limit = 1.08 * max(
        float(np.max(np.abs(clean_cycle))),
        float(np.max(np.abs(reconstructed_cycle))),
        1e-6,
    )

    apply_publication_style()
    fig, axes = plt.subplots(3, 1, figsize=(7.09, 4.7), sharex=True, constrained_layout=True)
    axes[0].plot(phase, noisy_cycle, color=DEFAULT_PALETTE["raw"], lw=1.0, label="Contaminated input")
    axes[0].plot(phase, clean_cycle, color=DEFAULT_PALETTE["clean"], lw=1.1, label="Clean target")
    axes[0].set(title="Before denoising", ylabel="Normalized SCG", ylim=(-full_limit, full_limit))
    axes[0].legend(loc="upper right", ncols=2)

    axes[1].plot(phase, noisy_cycle, color=DEFAULT_PALETTE["raw"], lw=1.0, label="Contaminated input")
    axes[1].plot(
        phase,
        reconstructed_cycle,
        color=DEFAULT_PALETTE["cycloscg"],
        lw=1.1,
        label="CycloSCGNet output",
    )
    axes[1].set(
        title="Direct contaminated-versus-denoised comparison (same amplitude scale)",
        ylabel="Normalized SCG",
        ylim=(-full_limit, full_limit),
    )
    axes[1].legend(loc="upper right", ncols=2)

    axes[2].plot(phase, clean_cycle, color=DEFAULT_PALETTE["clean"], lw=1.1, label="Clean target")
    axes[2].plot(
        phase,
        reconstructed_cycle,
        color=DEFAULT_PALETTE["cycloscg"],
        lw=1.1,
        label="CycloSCGNet output",
    )
    axes[2].fill_between(
        phase,
        clean_cycle,
        reconstructed_cycle,
        color=DEFAULT_PALETTE["accent"],
        alpha=0.12,
        linewidth=0,
    )
    axes[2].set(
        title="Recovered morphology (zoomed amplitude)",
        xlabel="Normalized cardiac phase",
        ylabel="Normalized SCG",
        ylim=(-zoom_limit, zoom_limit),
    )
    axes[2].legend(loc="upper right", ncols=2)
    axes[2].text(
        0.01,
        0.06,
        f"Input SNR {input_snr:.1f} dB  →  Output SNR {output_snr:.1f} dB;  "
        f"Pearson r = {pearson:.3f}",
        transform=axes[2].transAxes,
        fontsize=6,
        va="bottom",
    )
    for label, axis in zip(("a", "b", "c"), axes):
        _panel_label(axis, label)
    fig.suptitle(
        f"Representative severe-contamination heartbeat (cycle {cycle_index + 1}/{len(clean)})",
        fontsize=8,
    )
    note = (
        "Heartbeat selected by reconstruction RMSE nearest the 12-cycle median; not the best-performing beat."
    )
    if selection_note:
        note = f"{selection_note} {note}"
    fig.text(0.5, -0.01, note, ha="center", fontsize=6)
    return save_publication_figure(fig, output)


def plot_real_walking_waveform_before_after(
    walk1_raw: np.ndarray,
    walk1_reconstructed: np.ndarray,
    walk2_raw: np.ndarray,
    walk2_reconstructed: np.ndarray,
    output: str | Path,
) -> list[Path]:
    """Direct real-walking comparison using the median correction-energy cycle."""
    pairs = (
        ("Walking: nominal 1 step s$^{-1}$", walk1_raw, walk1_reconstructed),
        ("Walking: nominal 2 steps s$^{-1}$", walk2_raw, walk2_reconstructed),
    )
    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 4.2), constrained_layout=True)
    for row, (condition, raw_values, reconstructed_values) in enumerate(pairs):
        raw = np.asarray(raw_values, dtype=float)
        reconstructed = np.asarray(reconstructed_values, dtype=float)
        if raw.ndim != 2 or raw.shape != reconstructed.shape:
            raise ValueError("walking raw and reconstructed cycles must share shape [K, L]")
        correction = raw - reconstructed
        correction_rms = np.sqrt(np.mean(correction**2, axis=1))
        median_rms = float(np.median(correction_rms))
        cycle_index = int(np.argmin(np.abs(correction_rms - median_rms)))
        phase = np.linspace(0.0, 1.0, raw.shape[1], endpoint=False)
        raw_cycle = raw[cycle_index]
        reconstructed_cycle = reconstructed[cycle_index]
        removed_cycle = correction[cycle_index]
        waveform_limit = 1.08 * max(
            float(np.max(np.abs(raw_cycle))),
            float(np.max(np.abs(reconstructed_cycle))),
            1e-6,
        )
        removed_limit = 1.08 * max(float(np.max(np.abs(removed_cycle))), 1e-6)
        overlay_axis, removed_axis = axes[row]
        overlay_axis.plot(
            phase,
            raw_cycle,
            color=DEFAULT_PALETTE["raw"],
            lw=1.0,
            label="Raw contaminated walking",
        )
        overlay_axis.plot(
            phase,
            reconstructed_cycle,
            color=DEFAULT_PALETTE["cycloscg"],
            lw=1.0,
            label="CycloSCGNet output",
        )
        overlay_axis.fill_between(
            phase,
            raw_cycle,
            reconstructed_cycle,
            color=DEFAULT_PALETTE["accent"],
            alpha=0.16,
            linewidth=0,
            label="Changed by model",
        )
        overlay_axis.set(
            title=f"{condition}: direct comparison",
            ylabel="Normalized SCG",
            ylim=(-waveform_limit, waveform_limit),
        )
        overlay_axis.legend(loc="upper right", ncols=3, fontsize=6)
        removed_axis.plot(
            phase,
            removed_cycle,
            color=DEFAULT_PALETTE["accent"],
            lw=1.0,
        )
        removed_axis.axhline(0.0, color="#B8B8B8", lw=0.5)
        removed_axis.set(
            title=f"Model-estimated removed component (cycle {cycle_index + 1}/{len(raw)})",
            ylim=(-removed_limit, removed_limit),
        )
        removed_axis.text(
            0.02,
            0.05,
            f"Correction RMS = {correction_rms[cycle_index]:.3f}",
            transform=removed_axis.transAxes,
            fontsize=6,
            va="bottom",
        )
        if row == 1:
            overlay_axis.set_xlabel("Cardiac phase")
            removed_axis.set_xlabel("Cardiac phase")
    for label, axis in zip(("a", "b", "c", "d"), axes.flat):
        _panel_label(axis, label)
    fig.suptitle(
        "Real walking: contaminated waveform before and after reconstruction",
        fontsize=8,
    )
    fig.text(
        0.5,
        -0.01,
        "Representative cycle: correction RMS nearest the condition median. "
        "No synchronized clean walking ground truth is available.",
        ha="center",
        fontsize=6,
    )
    return save_publication_figure(fig, output)


def plot_correlation_matrices(
    clean: np.ndarray,
    noisy: np.ndarray,
    reconstructed: np.ndarray,
    output: str | Path,
) -> list[Path]:
    apply_publication_style()
    matrices = [pairwise_correlation_numpy(values) for values in (clean, noisy, reconstructed)]
    fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.45), constrained_layout=True)
    image = None
    for axis, matrix, title, panel in zip(
        axes, matrices, ("Clean target", "Noisy input", "Reconstructed"), ("a", "b", "c")
    ):
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set(title=title, xlabel="Cycle", ylabel="Cycle")
        _panel_label(axis, panel)
    fig.colorbar(image, ax=axes, label="Pearson correlation", shrink=0.8)
    return save_publication_figure(fig, output)


def plot_singular_spectra(
    clean: np.ndarray,
    noisy: np.ndarray,
    reconstructed: np.ndarray,
    output: str | Path,
    palette: Mapping[str, str] = DEFAULT_PALETTE,
) -> list[Path]:
    apply_publication_style()
    fig, axis = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    for values, label, color in (
        (clean, "Clean target", palette["clean"]),
        (noisy, "Noisy input", palette["raw"]),
        (reconstructed, "Reconstructed", palette["cycloscg"]),
    ):
        spectrum = singular_spectrum_numpy(values)
        axis.plot(np.arange(1, len(spectrum) + 1), spectrum, marker="o", markersize=3, color=color, label=label)
    axis.set(xlabel="Singular component", ylabel="Normalized singular value", yscale="log")
    axis.legend()
    return save_publication_figure(fig, output)


def plot_real_walking(
    walk1_raw: np.ndarray,
    walk1_reconstructed: np.ndarray,
    walk2_raw: np.ndarray,
    walk2_reconstructed: np.ndarray,
    output: str | Path,
    palette: Mapping[str, str] = DEFAULT_PALETTE,
) -> list[Path]:
    apply_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.09, 4.0), constrained_layout=True)
    for axis, raw, reconstructed, title, panel in zip(
        axes,
        (walk1_raw, walk2_raw),
        (walk1_reconstructed, walk2_reconstructed),
        (r"Walking: nominal 1 step s$^{-1}$", r"Walking: nominal 2 steps s$^{-1}$"),
        ("a", "b"),
    ):
        phase = np.linspace(0, 1, raw.shape[-1], endpoint=False)
        axis.plot(phase, np.median(raw, axis=0), color=palette["raw"], label="Raw walking")
        axis.plot(phase, np.median(reconstructed, axis=0), color=palette["cycloscg"], label="Reconstructed")
        axis.fill_between(
            phase,
            np.percentile(reconstructed, 25, axis=0),
            np.percentile(reconstructed, 75, axis=0),
            color=palette["cycloscg"],
            alpha=0.15,
            label="Reconstructed IQR",
        )
        axis.set(title=title, xlabel="Cardiac phase", ylabel="SCG (a.u.)")
        axis.legend(ncols=3)
        _panel_label(axis, panel)
    return save_publication_figure(fig, output)


def plot_frequency_overlap(
    frame,
    output: str | Path,
    palette: Mapping[str, str] = DEFAULT_PALETTE,
) -> list[Path]:
    """Visualize harmonic proximity and, when available, method advantage."""
    apply_publication_style()
    has_advantage = "cycloscg_advantage_over_baseline" in frame.columns
    columns = 2 if has_advantage else 1
    fig, axes = plt.subplots(1, columns, figsize=(7.09 if columns == 2 else 3.5, 2.8), constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    condition_colors = {
        "Walk_1step_per_s": palette["baseline"],
        "Walk_2steps_per_s": palette["cycloscg"],
    }
    condition_labels = {
        "Walk_1step_per_s": r"1 step s$^{-1}$",
        "Walk_2steps_per_s": r"2 steps s$^{-1}$",
    }
    for condition, group in frame.groupby("condition"):
        axes_array[0].scatter(
            group["cadence_hz"],
            group["cardiac_frequency_hz"],
            s=18,
            alpha=0.8,
            color=condition_colors.get(condition, palette["accent"]),
            label=condition_labels.get(condition, condition),
        )
    axes_array[0].set(xlabel="Estimated cadence (Hz)", ylabel="Cardiac fundamental (Hz)")
    axes_array[0].legend(fontsize=6)
    _panel_label(axes_array[0], "a")
    if has_advantage:
        axes_array[1].scatter(
            frame["minimum_harmonic_gap_hz"],
            frame["cycloscg_advantage_over_baseline"],
            color=palette["cycloscg"],
            s=18,
        )
        axes_array[1].axhline(0, color=palette["raw"], linewidth=0.8, linestyle="--")
        axes_array[1].set(
            xlabel="Minimum gait–cardiac harmonic gap (Hz)",
            ylabel="CycloSCGNet advantage\n(Gram-distance reduction)",
        )
        _panel_label(axes_array[1], "b")
    return save_publication_figure(fig, output)
