from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from .models import Correction


class PreviousMonthCorrector:
    """Applies a shrunk t-1 forecast residual without using target-month outcomes."""

    def __init__(self, weight: float, max_absolute_adjustment: float) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("correction.weight must be in [0, 1]")
        if max_absolute_adjustment < 0:
            raise ValueError("correction.max_absolute_adjustment must be non-negative")
        self.weight = weight
        self.maximum = max_absolute_adjustment

    def apply(self, raw_score: float, target_month: str, history_path: Path | None) -> tuple[float, Correction]:
        previous_month = self._previous_month(target_month)
        record = self._find_record(history_path, previous_month) if history_path else None
        if record is None:
            correction = Correction(None, None, None, 0.0, 0.0)
            return raw_score, correction

        prediction, actual = record
        residual = actual - prediction
        adjustment = max(-self.maximum, min(self.maximum, self.weight * residual))
        corrected = max(0.0, min(200.0, raw_score + adjustment))
        correction = Correction(previous_month, prediction, actual, round(residual, 4), round(adjustment, 4))
        return round(corrected, 4), correction

    @staticmethod
    def _find_record(path: Path, month: str) -> tuple[float, float] | None:
        if not path.exists():
            raise FileNotFoundError(f"Correction history not found: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"month", "predicted", "actual"}
        if rows and not required.issubset(rows[0]):
            raise ValueError(f"Correction history missing columns: {', '.join(sorted(required - set(rows[0])))}")
        matches = [row for row in rows if row["month"] == month]
        if len(matches) > 1:
            raise ValueError(f"Correction history has duplicate month {month}")
        if not matches:
            return None
        return float(matches[0]["predicted"]), float(matches[0]["actual"])

    @staticmethod
    def _previous_month(month: str) -> str:
        current = date.fromisoformat(f"{month}-01")
        if current.month == 1:
            return f"{current.year - 1}-12"
        return f"{current.year}-{current.month - 1:02d}"
