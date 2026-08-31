from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def signal_power(signal: np.ndarray, eps: float = 1e-12) -> float:
    values = np.asarray(signal, dtype=np.float64)
    return float(max(np.mean(values * values), eps))


def achieved_snr_db(clean: np.ndarray, noisy: np.ndarray, eps: float = 1e-12) -> float:
    clean_values = np.asarray(clean, dtype=np.float64)
    error = np.asarray(noisy, dtype=np.float64) - clean_values
    return float(10.0 * np.log10(signal_power(clean_values, eps) / signal_power(error, eps)))


def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    polarity: float = 1.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Mix independent motion-noise proxy at an exact power-ratio SNR.

    The noise proxy is not claimed to be physiologically pure noise. It is an
    independently recorded motion-contamination proxy used only for controlled
    supervised training and synthetic benchmarking.
    """
    clean_values = np.asarray(clean, dtype=np.float64)
    noise_values = np.asarray(noise, dtype=np.float64)
    if clean_values.shape != noise_values.shape:
        raise ValueError(
            f"clean and noise must have the same shape, got {clean_values.shape} and {noise_values.shape}"
        )
    if not np.isfinite(clean_values).all() or not np.isfinite(noise_values).all():
        raise ValueError("clean and noise must contain only finite values")
    clean_power = signal_power(clean_values, eps)
    noise_power = signal_power(noise_values, eps)
    scale = np.sqrt(clean_power / (noise_power * (10.0 ** (float(snr_db) / 10.0))))
    scaled_noise = float(np.sign(polarity) or 1.0) * scale * noise_values
    return (clean_values + scaled_noise).astype(np.float32), float(scale)


def extract_with_wrap(signal: np.ndarray, start: int, length: int) -> np.ndarray:
    """Extract an arbitrary offset window, wrapping short proxy recordings."""
    values = np.asarray(signal)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("signal must be a non-empty 1D array")
    if length <= 0:
        raise ValueError("length must be positive")
    indices = (np.arange(length, dtype=np.int64) + int(start)) % len(values)
    return values[indices]


@dataclass(frozen=True)
class MixMetadata:
    target_snr_db: float
    scale: float
    polarity: float
    noise_start: int
    identity: bool


class DynamicMixer:
    def __init__(
        self,
        snr_db_min: float = -15.0,
        snr_db_max: float = 5.0,
        random_polarity: bool = True,
        identity_probability: float = 0.1,
    ):
        if snr_db_min > snr_db_max:
            raise ValueError("snr_db_min must not exceed snr_db_max")
        if not 0.0 <= identity_probability <= 1.0:
            raise ValueError("identity_probability must lie in [0, 1]")
        self.snr_db_min = float(snr_db_min)
        self.snr_db_max = float(snr_db_max)
        self.random_polarity = bool(random_polarity)
        self.identity_probability = float(identity_probability)

    def __call__(
        self, clean: np.ndarray, noise: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, MixMetadata]:
        clean_values = np.asarray(clean, dtype=np.float32)
        if rng.random() < self.identity_probability:
            return clean_values.copy(), MixMetadata(float("inf"), 0.0, 1.0, 0, True)
        noise_start = int(rng.integers(0, max(len(noise), 1)))
        noise_window = extract_with_wrap(noise, noise_start, len(clean_values))
        snr_db = float(rng.uniform(self.snr_db_min, self.snr_db_max))
        polarity = float(rng.choice([-1.0, 1.0])) if self.random_polarity else 1.0
        mixed, scale = mix_at_snr(clean_values, noise_window, snr_db, polarity)
        return mixed, MixMetadata(snr_db, scale, polarity, noise_start, False)

