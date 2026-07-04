from __future__ import annotations

import pytest

from scripts.refresh_site_data import normalize_csv


def test_normalize_csv_accepts_required_site_rows() -> None:
    csv_text = (
        "as_of,record_type,region,forecast\n"
        "2026-07-04,monthly_prediction,us,53.12\n"
        "2026-07-04,weekly_prediction,us,53.12\n"
    )

    assert normalize_csv(csv_text) == csv_text


def test_normalize_csv_rejects_missing_weekly_rows() -> None:
    csv_text = (
        "as_of,record_type,region,forecast\n"
        "2026-07-04,monthly_prediction,us,53.12\n"
    )

    with pytest.raises(ValueError, match="weekly_prediction"):
        normalize_csv(csv_text)
