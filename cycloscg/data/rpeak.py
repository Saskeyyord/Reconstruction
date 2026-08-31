from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


class RPeakQualityError(RuntimeError):
    """Raised when ECG quality is insufficient for cardiac-phase alignment."""


@dataclass(frozen=True)
class RPeakDetection:
    indices: np.ndarray
    rr_intervals_s: np.ndarray
    heart_rate_bpm: float
    filtered_ecg: np.ndarray
    envelope: np.ndarray
    qc_passed: bool
    qc_messages: tuple[str, ...]

    def to_dict(self, include_traces: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "indices": self.indices.astype(int).tolist(),
            "rr_intervals_s": self.rr_intervals_s.astype(float).tolist(),
            "heart_rate_bpm": float(self.heart_rate_bpm),
            "qc_passed": bool(self.qc_passed),
            "qc_messages": list(self.qc_messages),
        }
        if include_traces:
            payload["filtered_ecg"] = self.filtered_ecg.astype(float).tolist()
            payload["envelope"] = self.envelope.astype(float).tolist()
        return payload


def _fill_nonfinite(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64).copy()
    finite = np.isfinite(values)
    if finite.all():
        return values
    if finite.sum() < 2:
        raise ValueError("ECG contains fewer than two finite samples")
    indices = np.arange(len(values))
    values[~finite] = np.interp(indices[~finite], indices[finite], values[finite])
    return values


def preprocess_ecg(
    ecg: np.ndarray,
    sampling_rate_hz: float,
    band_hz: tuple[float, float] = (5.0, 25.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Band-pass ECG and compute a Pan-Tompkins-like QRS energy envelope."""
    values = _fill_nonfinite(ecg)
    fs = float(sampling_rate_hz)
    if values.ndim != 1 or len(values) < int(2.0 * fs):
        raise ValueError("ECG must be a 1D trace at least two seconds long")
    low, high = band_hz
    if not 0 < low < high < fs / 2:
        raise ValueError("Invalid ECG band-pass frequencies")
    values = values - np.median(values)
    sos = butter(3, (low, high), btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, values)
    derivative_energy = np.gradient(filtered) ** 2
    integration_samples = max(3, int(round(0.12 * fs)))
    kernel = np.ones(integration_samples, dtype=np.float64) / integration_samples
    envelope = np.convolve(derivative_energy, kernel, mode="same")
    return filtered.astype(np.float64), envelope.astype(np.float64)


def _refine_peaks(
    coarse: np.ndarray, filtered: np.ndarray, fs: float, radius_s: float = 0.08
) -> np.ndarray:
    radius = max(1, int(round(radius_s * fs)))
    refined: list[int] = []
    for peak in coarse:
        left = max(0, int(peak) - radius)
        right = min(len(filtered), int(peak) + radius + 1)
        local = int(np.argmax(np.abs(filtered[left:right]))) + left
        if not refined or local - refined[-1] >= int(round(0.22 * fs)):
            refined.append(local)
        elif abs(filtered[local]) > abs(filtered[refined[-1]]):
            refined[-1] = local
    return np.asarray(refined, dtype=np.int64)


def _candidate_score(peaks: np.ndarray, fs: float, duration_s: float) -> float:
    if len(peaks) < 3:
        return -1e9 + len(peaks)
    rr = np.diff(peaks) / fs
    median_rr = float(np.median(rr))
    heart_rate = 60.0 / max(median_rr, 1e-6)
    plausible = float(np.mean((rr >= 0.30) & (rr <= 2.0)))
    rr_cv = float(np.std(rr) / max(np.mean(rr), 1e-6))
    count_rate = len(peaks) / max(duration_s, 1e-6)
    rate_penalty = max(0.0, count_rate - 3.0) + max(0.0, 0.45 - count_rate)
    hr_penalty = 0.0 if 35.0 <= heart_rate <= 220.0 else 4.0
    return 6.0 * plausible - min(rr_cv, 2.0) - rate_penalty - hr_penalty


def _envelope_strength(envelope: np.ndarray, index: int, radius: int) -> float:
    left = max(0, int(index) - radius)
    right = min(len(envelope), int(index) + radius + 1)
    return float(np.max(envelope[left:right]))


def _repair_rr_outliers(
    peaks: np.ndarray,
    envelope: np.ndarray,
    filtered: np.ndarray,
    fs: float,
    envelope_floor: float,
) -> np.ndarray:
    """Repair obvious missed/double detections relative to local cycle timing.

    Gait may be periodic, so this step uses only the ECG QRS-energy envelope and
    relative RR consistency; it does not use SCG or walking cadence.
    """
    repaired = np.asarray(peaks, dtype=np.int64)
    radius = max(1, int(round(0.08 * fs)))
    for _ in range(6):
        if len(repaired) < 4:
            break
        rr_samples = np.diff(repaired)
        median_rr = float(np.median(rr_samples))
        changed = False

        short = np.where(rr_samples < 0.55 * median_rr)[0]
        if len(short):
            pair_index = int(short[0])
            first, second = repaired[pair_index], repaired[pair_index + 1]
            trial_first = np.delete(repaired, pair_index)
            trial_second = np.delete(repaired, pair_index + 1)

            def timing_cost(candidate_peaks: np.ndarray) -> float:
                ratios = np.diff(candidate_peaks) / max(median_rr, 1.0)
                return float(np.mean(np.abs(np.log(np.clip(ratios, 0.1, 10.0)))))

            first_cost = timing_cost(trial_first)
            second_cost = timing_cost(trial_second)
            if np.isclose(first_cost, second_cost):
                first_strength = _envelope_strength(envelope, int(first), radius)
                second_strength = _envelope_strength(envelope, int(second), radius)
                remove_index = pair_index if first_strength < second_strength else pair_index + 1
            else:
                remove_index = pair_index if first_cost < second_cost else pair_index + 1
            repaired = np.delete(repaired, remove_index)
            changed = True
            continue

        long = np.where(rr_samples > 1.60 * median_rr)[0]
        if len(long):
            gap_index = int(long[0])
            left, right = int(repaired[gap_index]), int(repaired[gap_index + 1])
            margin = int(round(0.45 * median_rr))
            search_left, search_right = left + margin, right - margin
            if search_right > search_left:
                candidate = search_left + int(np.argmax(envelope[search_left:search_right]))
                if _envelope_strength(envelope, candidate, radius) >= envelope_floor:
                    repaired = np.sort(np.append(repaired, candidate))
                    repaired = _refine_peaks(repaired, filtered, fs)
                    changed = True
                    continue
        if not changed:
            break
    return repaired


def detect_rpeaks(
    ecg: np.ndarray,
    sampling_rate_hz: float = 256.0,
    min_rr_s: float = 0.30,
    min_beats: int = 5,
    strict: bool = False,
) -> RPeakDetection:
    """Detect ECG R peaks with adaptive QRS-envelope thresholds and explicit QC.

    With ``strict=True`` an unreliable record raises :class:`RPeakQualityError`,
    preventing silent use of poor cardiac-cycle alignments during training.
    """
    fs = float(sampling_rate_hz)
    filtered, envelope = preprocess_ecg(ecg, fs)
    center = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - center)))
    robust_scale = max(1.4826 * mad, float(np.std(envelope)) * 0.15, 1e-12)
    distance = max(1, int(round(min_rr_s * fs)))
    duration_s = len(filtered) / fs

    candidates: list[np.ndarray] = []
    for multiplier in (4.0, 3.0, 2.0, 1.25, 0.75):
        coarse, _ = find_peaks(
            envelope,
            distance=distance,
            height=center + multiplier * robust_scale,
            prominence=max(0.5 * multiplier * robust_scale, 1e-12),
        )
        candidates.append(_refine_peaks(coarse, filtered, fs))
    peaks = max(candidates, key=lambda values: _candidate_score(values, fs, duration_s))
    peaks = _repair_rr_outliers(
        peaks,
        envelope,
        filtered,
        fs,
        envelope_floor=center + 0.35 * robust_scale,
    )
    rr = np.diff(peaks).astype(np.float64) / fs
    heart_rate = float(60.0 / np.median(rr)) if len(rr) else float("nan")

    messages: list[str] = []
    if len(peaks) < min_beats:
        messages.append(f"insufficient_beats:{len(peaks)}<{min_beats}")
    if not np.isfinite(heart_rate) or not 35.0 <= heart_rate <= 220.0:
        messages.append(f"implausible_heart_rate_bpm:{heart_rate:.2f}")
    if len(rr):
        short_fraction = float(np.mean(rr < 0.30))
        long_fraction = float(np.mean(rr > 2.0))
        if short_fraction > 0.10:
            messages.append(f"too_many_short_rr:{short_fraction:.3f}")
        elif np.any(rr < 0.25):
            messages.append(f"very_short_rr_s:{float(rr.min()):.3f}")
        if long_fraction > 0.10:
            messages.append(f"too_many_long_rr:{long_fraction:.3f}")
        elif np.any(rr > 2.5):
            messages.append(f"very_long_rr_s:{float(rr.max()):.3f}")
        median_rr = float(np.median(rr))
        relative = rr / max(median_rr, 1e-6)
        if np.any(relative < 0.55):
            messages.append(f"possible_double_detection_rr_ratio:{float(relative.min()):.3f}")
        if np.any(relative > 1.60):
            messages.append(f"possible_missed_detection_rr_ratio:{float(relative.max()):.3f}")
    qc_passed = not messages
    detection = RPeakDetection(
        indices=peaks,
        rr_intervals_s=rr,
        heart_rate_bpm=heart_rate,
        filtered_ecg=filtered,
        envelope=envelope,
        qc_passed=qc_passed,
        qc_messages=tuple(messages),
    )
    if strict and not qc_passed:
        raise RPeakQualityError("; ".join(messages))
    return detection
