from __future__ import annotations

import numpy as np


def validate_rpeaks(r_peaks: np.ndarray, signal_length: int) -> np.ndarray:
    peaks = np.asarray(r_peaks, dtype=np.int64)
    if peaks.ndim != 1 or len(peaks) < 2:
        raise ValueError("At least two 1D R-peak indices are required")
    if np.any(np.diff(peaks) <= 1):
        raise ValueError("R peaks must be strictly increasing and separated by samples")
    # The final R peak may equal ``signal_length`` for a segment sliced exactly
    # from the first R peak up to (but not including) the final boundary sample.
    if peaks[0] < 0 or peaks[-1] > signal_length:
        raise ValueError("R peaks fall outside the signal")
    return peaks


def normalize_cardiac_cycles(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    phase_length: int = 256,
) -> np.ndarray:
    """Warp consecutive R-to-R intervals to a common cardiac-phase grid.

    Returns a cycle matrix with shape ``[K, L]``, where ``K`` is the number of
    R-to-R intervals and ``L`` is ``phase_length``. This representation makes
    cardiac-phase-dependent cross-cycle statistics explicit without claiming
    that all physiological beats are identical.
    """
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("signal must be a finite 1D array")
    if phase_length < 8:
        raise ValueError("phase_length must be at least 8")
    peaks = validate_rpeaks(r_peaks, len(values))
    target_phase = np.linspace(0.0, 1.0, phase_length, endpoint=False)
    cycles: list[np.ndarray] = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        beat = values[int(left) : int(right)]
        source_phase = np.linspace(0.0, 1.0, len(beat), endpoint=False)
        cycles.append(np.interp(target_phase, source_phase, beat))
    return np.stack(cycles).astype(np.float32)


def normalize_aligned_pair(
    noisy: np.ndarray,
    clean: np.ndarray,
    r_peaks: np.ndarray,
    phase_length: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exactly the same R-wave alignment to input and clean target."""
    noisy_values = np.asarray(noisy)
    clean_values = np.asarray(clean)
    if noisy_values.shape != clean_values.shape:
        raise ValueError("noisy and clean signals must share a shape")
    noisy_cycles = normalize_cardiac_cycles(noisy_values, r_peaks, phase_length)
    clean_cycles = normalize_cardiac_cycles(clean_values, r_peaks, phase_length)
    return noisy_cycles, clean_cycles


def cycle_windows(cycles: np.ndarray, beats_per_window: int = 12, stride: int = 1) -> np.ndarray:
    """Create sliding multi-cycle windows with shape ``[N, K, L]``."""
    values = np.asarray(cycles)
    if values.ndim != 2:
        raise ValueError("cycles must have shape [num_cycles, phase_length]")
    if beats_per_window <= 0 or stride <= 0:
        raise ValueError("beats_per_window and stride must be positive")
    if len(values) < beats_per_window:
        return np.empty((0, beats_per_window, values.shape[1]), dtype=values.dtype)
    starts = range(0, len(values) - beats_per_window + 1, stride)
    return np.stack([values[start : start + beats_per_window] for start in starts])
