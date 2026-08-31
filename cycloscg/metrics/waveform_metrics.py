from __future__ import annotations

import numpy as np


def _power(values: np.ndarray, eps: float = 1e-12) -> float:
    return float(max(np.mean(np.asarray(values, dtype=np.float64) ** 2), eps))


def waveform_metrics(
    estimate: np.ndarray,
    target: np.ndarray,
    noisy_input: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    estimate_values = np.asarray(estimate, dtype=np.float64).reshape(-1)
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    if estimate_values.shape != target_values.shape:
        raise ValueError("estimate and target must share a shape")
    error = estimate_values - target_values
    rmse = float(np.sqrt(np.mean(error**2)))
    target_rms = float(np.sqrt(_power(target_values, eps)))
    centered_norm = float(np.linalg.norm(target_values - target_values.mean()))
    correlation = (
        float(np.corrcoef(estimate_values, target_values)[0, 1])
        if np.std(estimate_values) > eps and np.std(target_values) > eps
        else float("nan")
    )
    metrics = {
        "rmse": rmse,
        "nrmse": rmse / max(target_rms, eps),
        "mae": float(np.mean(np.abs(error))),
        "pearson_r": correlation,
        "prd_percent": 100.0 * float(np.linalg.norm(error)) / max(centered_norm, eps),
        "output_snr_db": 10.0 * np.log10(_power(target_values, eps) / _power(error, eps)),
    }
    if noisy_input is not None:
        noisy_values = np.asarray(noisy_input, dtype=np.float64).reshape(-1)
        if noisy_values.shape != target_values.shape:
            raise ValueError("noisy_input and target must share a shape")
        input_error = noisy_values - target_values
        input_snr = 10.0 * np.log10(_power(target_values, eps) / _power(input_error, eps))
        metrics["input_snr_db"] = float(input_snr)
        metrics["snr_improvement_db"] = float(metrics["output_snr_db"] - input_snr)
    return {key: float(value) for key, value in metrics.items()}

