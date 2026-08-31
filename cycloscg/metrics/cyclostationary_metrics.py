from __future__ import annotations

import numpy as np


def _normalize(cycles: np.ndarray, eps: float) -> np.ndarray:
    values = np.asarray(cycles, dtype=np.float64)
    centered = values - values.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=1, keepdims=True))
    return centered / np.maximum(rms, eps)


def phase_gram_numpy(cycles: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    normalized = _normalize(cycles, eps)
    return normalized.T @ normalized / max(len(normalized), 1)


def singular_spectrum_numpy(cycles: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(cycles, dtype=np.float64)
    values = values - values.mean(axis=1, keepdims=True)
    singular = np.linalg.svd(values, compute_uv=False)
    return singular / max(float(singular.sum()), eps)


def cyclostationary_metrics(
    cycles: np.ndarray,
    reference: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    spectrum = singular_spectrum_numpy(cycles, eps)
    entropy = -float(np.sum(spectrum * np.log(np.maximum(spectrum, eps))))
    metrics = {
        "singular_first_fraction": float(spectrum[0]),
        "singular_top3_fraction": float(spectrum[:3].sum()),
        "singular_entropy": entropy,
    }
    if reference is not None:
        gram = phase_gram_numpy(cycles, eps)
        reference_gram = phase_gram_numpy(reference, eps)
        gram_distance = float(
            np.linalg.norm(gram - reference_gram, ord="fro")
            / max(np.linalg.norm(reference_gram, ord="fro"), eps)
        )
        reference_spectrum = singular_spectrum_numpy(reference, eps)
        size = max(len(spectrum), len(reference_spectrum))
        left = np.pad(spectrum, (0, size - len(spectrum)))
        right = np.pad(reference_spectrum, (0, size - len(reference_spectrum)))
        metrics.update(
            {
                "phase_gram_distance": gram_distance,
                "phase_gram_similarity": 1.0 / (1.0 + gram_distance),
                "singular_spectrum_distance": float(np.mean(np.abs(left - right))),
            }
        )
    return metrics

