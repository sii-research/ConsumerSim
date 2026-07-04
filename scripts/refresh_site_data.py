from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from consumersim_mcp_proxy.client import ConsumerSimBackend


DEFAULT_OUTPUT = ROOT / "site" / "data" / "consumersim_site_data.csv"
REQUIRED_COLUMNS = {"as_of", "record_type", "region"}
REQUIRED_RECORD_TYPES = {"monthly_prediction", "weekly_prediction"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the static ConsumerSim site data CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    csv_text = await ConsumerSimBackend().site_data_csv()
    normalized = normalize_csv(csv_text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(normalized, encoding="utf-8", newline="")
    print(f"Wrote {args.output}")


def normalize_csv(csv_text: str) -> str:
    if not csv_text.strip():
        raise ValueError("Backend returned an empty site-data CSV")

    reader = csv.DictReader(StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("Backend site-data response has no CSV header")

    missing = REQUIRED_COLUMNS - set(reader.fieldnames)
    if missing:
        raise ValueError(f"Backend site-data CSV is missing required columns: {sorted(missing)}")

    rows = list(reader)
    record_types = {row.get("record_type", "") for row in rows}
    missing_record_types = REQUIRED_RECORD_TYPES - record_types
    if missing_record_types:
        raise ValueError(f"Backend site-data CSV is missing rows: {sorted(missing_record_types)}")

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


if __name__ == "__main__":
    asyncio.run(main())
