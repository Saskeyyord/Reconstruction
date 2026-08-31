from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


MANIFEST_NAME = "dataset_manifest.csv"
DEFAULT_SIGNAL_COLUMNS = (
    "scg_z_bandpass_8_32Hz_m_s2",
    "accel_z_centered_m_s2",
    "accel_z_m_s2",
)
DEFAULT_ECG_COLUMNS = (
    "ecg_LA_RA_centered_mV",
    "ecg_LA_RA_mV",
)


@dataclass(frozen=True)
class SignalRecord:
    record_id: str
    participant_id: str
    cohort: str
    category: str
    condition: str
    standardized_relative_path: str
    sampling_rate_hz: float
    standard_samples: int
    ecg_available: bool
    sensor_location: str | None = None
    known_step_rate_hz: float | None = None


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return float(value)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def resolve_database_root(root: str | Path) -> Path:
    """Resolve the database directory from either it or one of its parents."""
    root_path = Path(root).expanduser().resolve()
    direct = root_path / "04_\u7d22\u5f15\u4e0e\u8bf4\u660e" / MANIFEST_NAME
    if direct.is_file():
        return root_path

    candidates = sorted(root_path.rglob(MANIFEST_NAME))
    candidates = [p for p in candidates if p.parent.name == "04_\u7d22\u5f15\u4e0e\u8bf4\u660e"]
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate {MANIFEST_NAME} below the supplied database root"
        )
    database_roots = sorted({p.parent.parent.resolve() for p in candidates})
    if len(database_roots) != 1:
        raise RuntimeError(
            "Multiple dataset manifests were found; pass the exact database directory"
        )
    return database_roots[0]


def select_signal_column(
    columns: Iterable[str], preferences: Sequence[str] = DEFAULT_SIGNAL_COLUMNS
) -> str:
    """Select an SCG channel using inspected preferences plus safe fallbacks.

    The function deliberately does not assume that every CSV uses one fixed
    header. It first honors configured names, then ranks semantically compatible
    Z-axis channels while excluding ECG/time/index fields.
    """
    available = [str(column) for column in columns]
    for preferred in preferences:
        if preferred in available:
            return preferred

    def score(name: str) -> int:
        normalized = re.sub(r"[^a-z0-9]+", "_", name.lower())
        if any(token in normalized for token in ("ecg", "time", "index")):
            return -100
        value = 0
        if "scg" in normalized:
            value += 8
        if re.search(r"(^|_)z(_|$)", normalized):
            value += 5
        if "bandpass" in normalized or "filtered" in normalized:
            value += 4
        if "centered" in normalized:
            value += 3
        if "accel" in normalized or "acceler" in normalized:
            value += 2
        return value

    ranked = sorted(((score(name), name) for name in available), reverse=True)
    if not ranked or ranked[0][0] <= 0:
        raise KeyError(
            "No compatible SCG/accelerometer Z-axis column was found. "
            f"Available columns: {available}"
        )
    return ranked[0][1]


def select_ecg_column(
    columns: Iterable[str], preferences: Sequence[str] = DEFAULT_ECG_COLUMNS
) -> str:
    available = [str(column) for column in columns]
    for preferred in preferences:
        if preferred in available:
            return preferred
    ranked = [
        name
        for name in available
        if "ecg" in re.sub(r"[^a-z0-9]+", "_", name.lower())
    ]
    if not ranked:
        raise KeyError(f"No ECG column was found. Available columns: {available}")
    ranked.sort(key=lambda name: ("center" in name.lower(), "la" in name.lower()), reverse=True)
    return ranked[0]


class DatabaseIndex:
    """Read-only view over the database manifest and standardized recordings."""

    def __init__(self, database_root: str | Path):
        self.root = resolve_database_root(database_root)
        self.manifest_path = self.root / "04_\u7d22\u5f15\u4e0e\u8bf4\u660e" / MANIFEST_NAME
        self.frame = pd.read_csv(self.manifest_path)
        required = {
            "RecordID",
            "ParticipantID",
            "Cohort",
            "Category",
            "Condition",
            "StandardizedRelativePath",
            "SamplingRate_Hz",
            "StandardSamples",
            "ECGAvailable",
        }
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Dataset manifest is missing fields: {sorted(missing)}")

    def records(
        self,
        category: str | None = None,
        participant_ids: Iterable[str] | None = None,
    ) -> list[SignalRecord]:
        frame = self.frame
        if category is not None:
            frame = frame[frame["Category"] == category]
        if participant_ids is not None:
            allowed = set(participant_ids)
            frame = frame[frame["ParticipantID"].isin(allowed)]
        records: list[SignalRecord] = []
        for row in frame.to_dict(orient="records"):
            records.append(
                SignalRecord(
                    record_id=str(row["RecordID"]),
                    participant_id=str(row["ParticipantID"]),
                    cohort=str(row["Cohort"]),
                    category=str(row["Category"]),
                    condition=str(row["Condition"]),
                    standardized_relative_path=str(row["StandardizedRelativePath"]),
                    sampling_rate_hz=float(row["SamplingRate_Hz"]),
                    standard_samples=int(row["StandardSamples"]),
                    ecg_available=_as_bool(row["ECGAvailable"]),
                    sensor_location=(
                        None
                        if "SensorLocation" not in row or pd.isna(row["SensorLocation"])
                        else str(row["SensorLocation"])
                    ),
                    known_step_rate_hz=_optional_float(row.get("KnownStepRate_Hz")),
                )
            )
        return records

    def record_path(self, record: SignalRecord) -> Path:
        path = (self.root / Path(record.standardized_relative_path)).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Manifest path escapes the database root") from exc
        if not path.is_file():
            raise FileNotFoundError(
                f"Standardized recording is missing: {record.standardized_relative_path}"
            )
        return path

    def load_record(
        self,
        record: SignalRecord,
        signal_columns: Sequence[str] = DEFAULT_SIGNAL_COLUMNS,
        ecg_columns: Sequence[str] = DEFAULT_ECG_COLUMNS,
        include_ecg: bool = True,
    ) -> dict[str, object]:
        frame = pd.read_csv(self.record_path(record))
        signal_column = select_signal_column(frame.columns, signal_columns)
        signal = pd.to_numeric(frame[signal_column], errors="coerce").to_numpy(np.float32)
        if not np.isfinite(signal).all():
            raise ValueError(f"Non-finite SCG values in record {record.record_id}")
        result: dict[str, object] = {
            "signal": signal,
            "signal_column": signal_column,
            "sampling_rate_hz": record.sampling_rate_hz,
        }
        if include_ecg and record.ecg_available:
            ecg_column = select_ecg_column(frame.columns, ecg_columns)
            ecg = pd.to_numeric(frame[ecg_column], errors="coerce").to_numpy(np.float32)
            if not np.isfinite(ecg).all():
                raise ValueError(f"Non-finite ECG values in record {record.record_id}")
            result.update({"ecg": ecg, "ecg_column": ecg_column})
        return result

    def participant_ids(self, cohort: str) -> list[str]:
        frame = self.frame[self.frame["Cohort"] == cohort]
        return sorted(frame["ParticipantID"].astype(str).unique().tolist())

    def audit(self, full_scan: bool = True) -> dict[str, object]:
        """Audit standardized files without changing the source database."""
        records = self.records()
        header_counts: dict[str, int] = {}
        row_mismatches: list[dict[str, object]] = []
        errors: list[dict[str, str]] = []
        selected_signal_columns: dict[str, int] = {}
        selected_ecg_columns: dict[str, int] = {}
        scanned = records if full_scan else records[: min(12, len(records))]
        for record in scanned:
            try:
                path = self.record_path(record)
                if full_scan:
                    frame = pd.read_csv(path)
                    rows = len(frame)
                else:
                    frame = pd.read_csv(path, nrows=8)
                    rows = record.standard_samples
                header_key = "|".join(map(str, frame.columns))
                header_counts[header_key] = header_counts.get(header_key, 0) + 1
                signal_column = select_signal_column(frame.columns)
                selected_signal_columns[signal_column] = (
                    selected_signal_columns.get(signal_column, 0) + 1
                )
                if record.ecg_available:
                    ecg_column = select_ecg_column(frame.columns)
                    selected_ecg_columns[ecg_column] = selected_ecg_columns.get(ecg_column, 0) + 1
                if rows != record.standard_samples:
                    row_mismatches.append(
                        {
                            "record_id": record.record_id,
                            "expected": record.standard_samples,
                            "actual": rows,
                        }
                    )
                if full_scan:
                    numeric = frame[[signal_column]].apply(pd.to_numeric, errors="coerce")
                    if numeric.isna().any().any():
                        errors.append(
                            {"record_id": record.record_id, "error": "non-finite signal values"}
                        )
            except Exception as exc:  # audit must report every bad record
                errors.append({"record_id": record.record_id, "error": str(exc)})

        category_counts = self.frame.groupby("Category").size().astype(int).to_dict()
        cohort_participants = (
            self.frame.groupby("Cohort")["ParticipantID"].nunique().astype(int).to_dict()
        )
        return {
            "database_version": "manifest-driven",
            "record_count": len(records),
            "scanned_record_count": len(scanned),
            "category_counts": category_counts,
            "participants_by_cohort": cohort_participants,
            "header_variant_count": len(header_counts),
            "selected_signal_columns": selected_signal_columns,
            "selected_ecg_columns": selected_ecg_columns,
            "row_mismatches": row_mismatches,
            "errors": errors,
            "passed": not row_mismatches and not errors,
        }


def record_to_dict(record: SignalRecord) -> dict[str, object]:
    return asdict(record)

