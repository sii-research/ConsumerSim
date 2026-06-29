from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .models import Consumer, PipelineResult


def write_outputs(output_root: Path, result: PipelineResult, population: list[Consumer]) -> Path:
    run_dir = output_root / result.region.lower() / result.target_month
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, ensure_ascii=False, indent=2)
    with (run_dir / "population_responses.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "consumer_id", "region", "age_group", "income_group", "education_group",
            "location_group", "weight", "is_core", "current_finance", "durable_buying",
            "future_finance", "business_12m", "business_5y",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for consumer in population:
            writer.writerow({
                "consumer_id": consumer.consumer_id,
                "region": consumer.region.value,
                "age_group": consumer.age_group,
                "income_group": consumer.income_group,
                "education_group": consumer.education_group,
                "location_group": consumer.location_group,
                "weight": consumer.weight,
                "is_core": consumer.is_core,
                **(consumer.response or {}),
            })
    return run_dir
