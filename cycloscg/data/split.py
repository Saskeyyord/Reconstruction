from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ParticipantSplit:
    seed: int
    train_clean_subjects: list[str]
    val_clean_subjects: list[str]
    test_clean_subjects: list[str]
    train_noise_subjects: list[str]
    val_noise_subjects: list[str]
    test_noise_subjects: list[str]

    def validate(self) -> None:
        for prefix in ("clean", "noise"):
            train = set(getattr(self, f"train_{prefix}_subjects"))
            val = set(getattr(self, f"val_{prefix}_subjects"))
            test = set(getattr(self, f"test_{prefix}_subjects"))
            if train & val or train & test or val & test:
                raise ValueError(f"Participant leakage detected in {prefix} split")
            if not train or not val or not test:
                raise ValueError(f"Every {prefix} split must contain at least one participant")

    def save(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ParticipantSplit":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        split = cls(**payload)
        split.validate()
        return split


def _allocate_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if n < 3:
        raise ValueError("At least three participants are required for train/val/test")
    raw = np.asarray(ratios, dtype=np.float64) * n
    counts = np.floor(raw).astype(int)
    counts = np.maximum(counts, 1)
    while counts.sum() > n:
        candidates = np.where(counts > 1)[0]
        index = int(candidates[np.argmin(raw[candidates] - counts[candidates])])
        counts[index] -= 1
    remainder_order = np.argsort(-(raw - np.floor(raw)))
    cursor = 0
    while counts.sum() < n:
        counts[int(remainder_order[cursor % 3])] += 1
        cursor += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def _split_ids(
    participant_ids: Iterable[str], rng: np.random.Generator, ratios: tuple[float, float, float]
) -> tuple[list[str], list[str], list[str]]:
    ids = np.asarray(sorted(set(participant_ids)), dtype=object)
    rng.shuffle(ids)
    n_train, n_val, _ = _allocate_counts(len(ids), ratios)
    train = sorted(ids[:n_train].tolist())
    val = sorted(ids[n_train : n_train + n_val].tolist())
    test = sorted(ids[n_train + n_val :].tolist())
    return train, val, test


def create_participant_split(
    clean_subjects: Iterable[str],
    noise_subjects: Iterable[str],
    seed: int = 20260830,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> ParticipantSplit:
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0) or min(ratios) <= 0:
        raise ValueError("ratios must be three positive values summing to one")
    clean_rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 1)
    train_clean, val_clean, test_clean = _split_ids(clean_subjects, clean_rng, ratios)
    train_noise, val_noise, test_noise = _split_ids(noise_subjects, noise_rng, ratios)
    split = ParticipantSplit(
        seed=seed,
        train_clean_subjects=train_clean,
        val_clean_subjects=val_clean,
        test_clean_subjects=test_clean,
        train_noise_subjects=train_noise,
        val_noise_subjects=val_noise,
        test_noise_subjects=test_noise,
    )
    split.validate()
    return split

