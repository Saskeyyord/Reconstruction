from __future__ import annotations

import numpy as np


def _normalized_cycles(cycles: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(cycles, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("cycles must have shape [K, L]")
    centered = values - values.mean(axis=1, keepdims=True)
    return centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), eps)


def pairwise_correlation_numpy(cycles: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    normalized = _normalized_cycles(cycles, eps)
    return normalized @ normalized.T


def mean_off_diagonal(matrix: np.ndarray) -> float:
    values = np.asarray(matrix)
    if len(values) < 2:
        return float("nan")
    mask = ~np.eye(len(values), dtype=bool)
    return float(np.mean(values[mask]))


def cycle_structure_metrics(
    cycles: np.ndarray,
    reference: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    values = np.asarray(cycles, dtype=np.float64)
    correlations = pairwise_correlation_numpy(values, eps)
    template = np.median(values, axis=0)
    template_centered = template - template.mean()
    template_norm = max(np.linalg.norm(template_centered), eps)
    normalized = _normalized_cycles(values, eps)
    template_correlations = normalized @ (template_centered / template_norm)
    metrics = {
        "mean_pairwise_correlation": mean_off_diagonal(correlations),
        "mean_template_correlation": float(np.mean(template_correlations)),
        "cycle_variability": float(np.mean(np.std(values, axis=0))),
    }
    if reference is not None:
        reference_correlations = pairwise_correlation_numpy(reference, eps)
        size = min(len(correlations), len(reference_correlations))
        metrics["cycle_coherence_error"] = float(
            np.linalg.norm(correlations[:size, :size] - reference_correlations[:size, :size], ord="fro")
            / max(size, 1)
        )
    return metrics

