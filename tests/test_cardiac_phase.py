import numpy as np

from cycloscg.data.cardiac_phase import (
    cycle_windows,
    normalize_aligned_pair,
    normalize_cardiac_cycles,
)


def test_phase_normalization_dimensions_and_alignment() -> None:
    signal = np.sin(np.linspace(0, 20 * np.pi, 1000, endpoint=False))
    peaks = np.array([0, 91, 195, 302, 401, 505, 610, 710, 815, 920, 1000])
    cycles = normalize_cardiac_cycles(signal, peaks, phase_length=256)
    assert cycles.shape == (10, 256)
    noisy_cycles, clean_cycles = normalize_aligned_pair(signal + 0.1, signal, peaks, 128)
    assert noisy_cycles.shape == clean_cycles.shape == (10, 128)
    windows = cycle_windows(cycles, beats_per_window=4, stride=2)
    assert windows.shape == (4, 4, 256)

