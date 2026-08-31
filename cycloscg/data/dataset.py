from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .cardiac_phase import normalize_aligned_pair
from .mixing import DynamicMixer
from .records import (
    DEFAULT_ECG_COLUMNS,
    DEFAULT_SIGNAL_COLUMNS,
    DatabaseIndex,
    SignalRecord,
)
from .rpeak import RPeakDetection, detect_rpeaks


class _DynamicDatasetBase(Dataset[dict[str, object]]):
    def __init__(
        self,
        database: DatabaseIndex,
        clean_records: Sequence[SignalRecord],
        noise_records: Sequence[SignalRecord],
        samples_per_epoch: int,
        mixer: DynamicMixer,
        seed: int,
        signal_columns: Sequence[str] = DEFAULT_SIGNAL_COLUMNS,
        ecg_columns: Sequence[str] = DEFAULT_ECG_COLUMNS,
        normalize_by_clean_rms: bool = True,
    ):
        if not clean_records or not noise_records:
            raise ValueError("clean_records and noise_records must not be empty")
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self.database = database
        self.clean_records = list(clean_records)
        self.noise_records = list(noise_records)
        self.samples_per_epoch = int(samples_per_epoch)
        self.mixer = mixer
        self.seed = int(seed)
        self.signal_columns = tuple(signal_columns)
        self.ecg_columns = tuple(ecg_columns)
        self.normalize_by_clean_rms = bool(normalize_by_clean_rms)
        self.epoch = 0
        self._record_cache: dict[str, dict[str, object]] = {}
        self._rpeak_cache: dict[str, RPeakDetection] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + self.epoch * 1_000_003 + int(index))

    def _load(self, record: SignalRecord, include_ecg: bool = False) -> dict[str, object]:
        cache_key = f"{record.record_id}:{int(include_ecg)}"
        if cache_key not in self._record_cache:
            self._record_cache[cache_key] = self.database.load_record(
                record,
                signal_columns=self.signal_columns,
                ecg_columns=self.ecg_columns,
                include_ecg=include_ecg,
            )
        return self._record_cache[cache_key]

    @staticmethod
    def _normalize_pair(noisy: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        rms = float(np.sqrt(np.mean(np.asarray(clean, dtype=np.float64) ** 2)))
        scale = max(rms, 1e-6)
        return noisy / scale, clean / scale, scale


class DynamicWaveformDataset(_DynamicDatasetBase):
    """Direct-waveform dynamic mixing dataset for the conventional baseline."""

    def __init__(self, *args: object, window_samples: int = 2048, **kwargs: object):
        super().__init__(*args, **kwargs)
        if window_samples <= 0:
            raise ValueError("window_samples must be positive")
        self.window_samples = int(window_samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        rng = self._rng(index)
        clean_record = self.clean_records[int(rng.integers(len(self.clean_records)))]
        noise_record = self.noise_records[int(rng.integers(len(self.noise_records)))]
        clean_full = np.asarray(self._load(clean_record)["signal"], dtype=np.float32)
        noise_full = np.asarray(self._load(noise_record)["signal"], dtype=np.float32)
        if len(clean_full) < self.window_samples:
            raise ValueError(f"Record {clean_record.record_id} is shorter than waveform window")
        clean_start = int(rng.integers(0, len(clean_full) - self.window_samples + 1))
        clean = clean_full[clean_start : clean_start + self.window_samples]
        noisy, mix = self.mixer(clean, noise_full, rng)
        normalization_scale = 1.0
        if self.normalize_by_clean_rms:
            noisy, clean, normalization_scale = self._normalize_pair(noisy, clean)
        return {
            "noisy": torch.from_numpy(np.asarray(noisy, dtype=np.float32)),
            "clean": torch.from_numpy(np.asarray(clean, dtype=np.float32)),
            "target_snr_db": torch.tensor(mix.target_snr_db, dtype=torch.float32),
            "identity": torch.tensor(mix.identity, dtype=torch.bool),
            "normalization_scale": torch.tensor(normalization_scale, dtype=torch.float32),
            "clean_record_id": clean_record.record_id,
            "noise_record_id": noise_record.record_id,
        }


class DynamicCycleDataset(_DynamicDatasetBase):
    """Dynamic synthetic contamination in cardiac-phase matrix form.

    Output/target tensors have shape ``[K, L]``. ECG R peaks are detected from
    the clean resting record and the same alignment is applied to the synthetic
    noisy input and clean target. R-peak QC is strict by default.
    """

    def __init__(
        self,
        *args: object,
        beats_per_window: int = 12,
        phase_length: int = 256,
        strict_rpeak_qc: bool = True,
        **kwargs: object,
    ):
        super().__init__(*args, **kwargs)
        self.beats_per_window = int(beats_per_window)
        self.phase_length = int(phase_length)
        self.strict_rpeak_qc = bool(strict_rpeak_qc)
        if self.beats_per_window <= 0 or self.phase_length < 8:
            raise ValueError("Invalid beats_per_window or phase_length")

    def _detection(self, record: SignalRecord, ecg: np.ndarray) -> RPeakDetection:
        if record.record_id not in self._rpeak_cache:
            self._rpeak_cache[record.record_id] = detect_rpeaks(
                ecg,
                sampling_rate_hz=record.sampling_rate_hz,
                strict=self.strict_rpeak_qc,
            )
        return self._rpeak_cache[record.record_id]

    def __getitem__(self, index: int) -> dict[str, object]:
        rng = self._rng(index)
        clean_record = self.clean_records[int(rng.integers(len(self.clean_records)))]
        noise_record = self.noise_records[int(rng.integers(len(self.noise_records)))]
        clean_loaded = self._load(clean_record, include_ecg=True)
        clean_full = np.asarray(clean_loaded["signal"], dtype=np.float32)
        ecg = np.asarray(clean_loaded["ecg"], dtype=np.float32)
        detection = self._detection(clean_record, ecg)
        peaks = detection.indices
        if len(peaks) < self.beats_per_window + 1:
            raise ValueError(
                f"Record {clean_record.record_id} has too few detected beats for K={self.beats_per_window}"
            )
        beat_start = int(rng.integers(0, len(peaks) - self.beats_per_window))
        left = int(peaks[beat_start])
        right = int(peaks[beat_start + self.beats_per_window])
        clean_segment = clean_full[left:right]
        noise_full = np.asarray(self._load(noise_record)["signal"], dtype=np.float32)
        noisy_segment, mix = self.mixer(clean_segment, noise_full, rng)
        local_peaks = peaks[beat_start : beat_start + self.beats_per_window + 1] - left
        noisy_cycles, clean_cycles = normalize_aligned_pair(
            noisy_segment, clean_segment, local_peaks, self.phase_length
        )
        normalization_scale = 1.0
        if self.normalize_by_clean_rms:
            noisy_cycles, clean_cycles, normalization_scale = self._normalize_pair(
                noisy_cycles, clean_cycles
            )
        return {
            "noisy": torch.from_numpy(np.asarray(noisy_cycles, dtype=np.float32)),
            "clean": torch.from_numpy(np.asarray(clean_cycles, dtype=np.float32)),
            "target_snr_db": torch.tensor(mix.target_snr_db, dtype=torch.float32),
            "identity": torch.tensor(mix.identity, dtype=torch.bool),
            "normalization_scale": torch.tensor(normalization_scale, dtype=torch.float32),
            "clean_record_id": clean_record.record_id,
            "noise_record_id": noise_record.record_id,
            "heart_rate_bpm": torch.tensor(detection.heart_rate_bpm, dtype=torch.float32),
        }

