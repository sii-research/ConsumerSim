from __future__ import annotations

import argparse
import calendar
import csv
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from consumer_pipeline.config import load_config
from consumer_pipeline.orchestrator import ConsumerPipeline


SITE_DATA = ROOT / "data" / "consumersim_site_data.csv"
DRIVER_EVENTS = ROOT / "data" / "forecast_driver_events.csv"
DEFAULT_TIMEZONE = ZoneInfo("Asia/Shanghai")
REGIONS = ("us", "eu", "jp")
WEEKLY_POINT_COUNT = 9
DRIVER_ROWS_PER_CADENCE = 3

REGION_LABELS = {
    "us": "US",
    "eu": "EU27",
    "jp": "Japan",
}

# The core pipeline emits an internal 0-200 harmonized confidence score. The
# public site displays the historical regional index scale used by its charts.
# These offsets preserve the current site calibration while letting the pipeline
# move the displayed forecast as new inputs change.
DISPLAY_OFFSETS = {
    "us": -46.94,
    "eu": -109.73,
    "jp": -60.60,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate static ConsumerSim site data.")
    parser.add_argument("--month", help="Target month in YYYY-MM. Defaults to the as-of month.")
    parser.add_argument("--as-of", help="Information cutoff in YYYY-MM-DD. Defaults to today in Asia/Shanghai.")
    parser.add_argument("--template", type=Path, default=SITE_DATA, help="Existing site CSV to use as template.")
    parser.add_argument("--output", type=Path, default=SITE_DATA, help="CSV path to write.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    as_of = date.fromisoformat(args.as_of) if args.as_of else datetime.now(DEFAULT_TIMEZONE).date()
    target_month = args.month or as_of.strftime("%Y-%m")

    rows, fieldnames = read_rows(args.template)
    monthly_results = {region: run_region(region, target_month, as_of.isoformat()) for region in REGIONS}
    weekly_results = {
        region: [
            run_region(region, target_month, cutoff.isoformat())
            for cutoff in weekly_cutoffs(as_of, target_month)
        ]
        for region in REGIONS
    }

    update_rows(rows, target_month, as_of, monthly_results, weekly_results)
    write_rows(args.output, fieldnames, rows)
    print(f"Wrote {args.output} for {target_month} as of {as_of.isoformat()}")


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")
        return [dict(row) for row in reader], list(reader.fieldnames)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_region(region: str, target_month: str, as_of: str) -> Any:
    config = load_config(ROOT / "configs" / f"{region}.yaml")
    provider = os.getenv("CONSUMERSIM_PREDICTION_PROVIDER", "").strip().lower()
    if provider:
        config["prediction"]["provider"] = provider
    if os.getenv("CONSUMERSIM_MODEL_NAME"):
        config["prediction"]["model"] = os.environ["CONSUMERSIM_MODEL_NAME"]
    if os.getenv("CONSUMERSIM_MODEL_ENDPOINT"):
        config["prediction"]["endpoint"] = os.environ["CONSUMERSIM_MODEL_ENDPOINT"]
    result, _ = ConsumerPipeline(config).run(target_month, as_of)
    return result


def weekly_cutoffs(as_of: date, target_month: str) -> list[date]:
    year, month = (int(part) for part in target_month.split("-"))
    month_end = date(year, month, calendar.monthrange(year, month)[1])
    effective_as_of = min(as_of, month_end)
    return [effective_as_of - timedelta(days=7 * offset) for offset in range(WEEKLY_POINT_COUNT - 1, -1, -1)]


def update_rows(
    rows: list[dict[str, str]],
    target_month: str,
    as_of: date,
    monthly_results: dict[str, Any],
    weekly_results: dict[str, list[Any]],
) -> None:
    rows[:] = ensure_weekly_rows(rows)
    rows[:] = ensure_driver_rows(rows)
    curated_driver_events = read_driver_events(DRIVER_EVENTS)
    snapshot = as_of.isoformat()
    period = month_label(target_month)
    prior_period = previous_month_label(target_month)
    next_update = next_monday(as_of).isoformat()
    monthly_values = {region: display_value(region, monthly_results[region].corrected_score) for region in REGIONS}
    weekly_values = {
        region: [display_value(region, result.corrected_score) for result in weekly_results[region]]
        for region in REGIONS
    }

    latest_series_sort: dict[str, int] = {}
    for row in rows:
        row["as_of"] = snapshot
        if row.get("record_type") == "series" and row.get("region") in REGIONS:
            latest_series_sort[row["region"]] = max(latest_series_sort.get(row["region"], -1), int(row.get("sort_order") or 0))

    for row in rows:
        record_type = row.get("record_type")
        region = row.get("region")
        if record_type == "meta":
            update_meta(row, as_of, next_update)
        elif record_type == "monthly_prediction" and region in REGIONS:
            update_monthly_prediction(row, region, period, prior_period, monthly_results[region], monthly_values[region])
        elif record_type == "weekly_prediction" and region in REGIONS:
            update_weekly_prediction(row, region, target_month, weekly_results[region], weekly_values[region])
        elif record_type == "forecast_news" and region in REGIONS:
            update_driver_note(row, region, period, monthly_results[region], curated_driver_events)
        elif record_type == "region_summary" and region in REGIONS:
            update_region_summary(row, region, period, monthly_results[region], monthly_values[region])
        elif record_type == "series" and region in REGIONS and int(row.get("sort_order") or 0) == latest_series_sort[region]:
            row["period"] = period
            row["forecast"] = f"{monthly_values[region]:.2f}"
            row["actual"] = ""


def ensure_weekly_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return ensure_repeated_rows(rows, "weekly_prediction", WEEKLY_POINT_COUNT)


def ensure_driver_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return ensure_repeated_rows(rows, "forecast_news", DRIVER_ROWS_PER_CADENCE * 2)


def ensure_repeated_rows(rows: list[dict[str, str]], record_type: str, count: int) -> list[dict[str, str]]:
    if not rows:
        return rows
    fieldnames = list(rows[0])
    output: list[dict[str, str]] = []
    emitted: set[str] = set()
    for row in rows:
        region = row.get("region")
        if row.get("record_type") == record_type and region in REGIONS:
            if region in emitted:
                continue
            template = row
            for sort_order in range(1, count + 1):
                copy = {field: template.get(field, "") for field in fieldnames}
                copy["record_type"] = record_type
                copy["region"] = region
                copy["sort_order"] = str(sort_order)
                if record_type == "forecast_news":
                    copy["key"] = "weekly" if sort_order <= DRIVER_ROWS_PER_CADENCE else "monthly"
                output.append(copy)
            emitted.add(region)
        else:
            output.append(row)
    return output


def update_meta(row: dict[str, str], as_of: date, next_update: str) -> None:
    if row.get("key") == "generatedAt":
        row["value"] = as_of.isoformat()
    elif row.get("key") == "nextUpdate":
        row["value"] = next_update


def update_monthly_prediction(
    row: dict[str, str],
    region: str,
    period: str,
    prior_period: str,
    result: Any,
    value: float,
) -> None:
    previous = result.correction.get("previous_actual") or result.correction.get("previous_prediction")
    delta = value - display_value(region, previous) if previous is not None else 0.0
    row.update(
        {
            "label": REGION_LABELS[region],
            "period": period,
            "value": f"{value:.2f}",
            "value_label": f"{value:.2f}",
            "prior_period": prior_period,
            "signal": f"{delta:+.2f} vs {prior_period}, based on the latest ConsumerSim run.",
            "interpretation": interpretation(result.environment.get("combined_score", 0.0)),
        }
    )


def update_weekly_prediction(
    row: dict[str, str],
    region: str,
    target_month: str,
    results: list[Any],
    values: list[float],
) -> None:
    index = max(0, min(int(row.get("sort_order") or 1) - 1, len(results) - 1))
    result = results[index]
    value = values[index]
    question_scores = result.question_scores
    current = display_value(region, (question_scores["current_finance"] + question_scores["durable_buying"]) / 2.0)
    expectations = display_value(
        region,
        (question_scores["future_finance"] + question_scores["business_12m"] + question_scores["business_5y"]) / 3.0,
    )
    row.update(
        {
            "label": REGION_LABELS[region],
            "period": month_label(target_month),
            "week_label": week_label(date.fromisoformat(result.as_of)),
            "cutoff_day": str(date.fromisoformat(result.as_of).day),
            "forecast": f"{value:.2f}",
            "actual": f"{current:.2f}",
            "error": f"{expectations:.2f}",
            "signal": weekly_signal(result.environment.get("combined_score", 0.0)),
            "interpretation": f"Weekly nowcast through {result.as_of} from the ConsumerSim pipeline.",
        }
    )


def read_driver_events(path: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    events: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        region = row.get("region", "")
        cadence = row.get("cadence", "")
        if region not in REGIONS or cadence not in {"weekly", "monthly"}:
            continue
        events.setdefault((region, cadence), []).append(row)
    for key, items in events.items():
        items.sort(key=lambda item: int(item.get("sort_order") or 0))
    return events


def update_driver_note(
    row: dict[str, str],
    region: str,
    period: str,
    result: Any,
    curated_driver_events: dict[tuple[str, str], list[dict[str, str]]],
) -> None:
    cadence = row.get("key") or "weekly"
    sort_order = int(row.get("sort_order") or 1)
    cadence_index = (sort_order - 1) % DRIVER_ROWS_PER_CADENCE
    events = curated_driver_events.get((region, cadence)) or driver_events(result.environment)
    if cadence_index >= len(events):
        clear_driver_note(row)
        return
    event = events[cadence_index]
    row.update(
        {
            "label": event.get("headline", ""),
            "market": event.get("source", ""),
            "period": period,
            "week_label": event.get("event_period") or period,
            "signal": event.get("tag", ""),
            "interpretation": event.get("summary", ""),
            "note": event.get("url", ""),
        }
    )


def clear_driver_note(row: dict[str, str]) -> None:
    row.update(
        {
            "label": "",
            "market": "",
            "period": "",
            "week_label": "",
            "signal": "",
            "interpretation": "",
            "note": "",
        }
    )


def driver_events(environment: dict[str, Any]) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for item in environment.get("news_items", []):
        published = str(item.get("published_at", ""))[:10]
        sentiment = float(item.get("sentiment", 0.0))
        relevance = float(item.get("relevance", 0.0))
        events.append(
            {
                "date": published,
                "tag": "News",
                "source": "Input news feed",
                "headline": str(item.get("title", "")),
                "summary": f"Published {published}; sentiment {sentiment:+.2f}, relevance {relevance:.2f}.",
                "url": "",
            }
        )
    for item in environment.get("indicators", []):
        observed = str(item.get("observed_at", ""))[:10]
        z_score = float(item.get("z_score", 0.0))
        weight = float(item.get("weight", 0.0))
        events.append(
            {
                "date": observed,
                "tag": "Indicator",
                "source": "Input indicators",
                "headline": str(item.get("name", "")).replace("_", " ").title(),
                "summary": f"Observed {observed}; z-score {z_score:+.2f}, model weight {weight:.2f}.",
                "url": "",
            }
        )
    events = [event for event in events if event["headline"]]
    return sorted(events, key=lambda event: event["date"], reverse=True)[:DRIVER_ROWS_PER_CADENCE]


def update_region_summary(row: dict[str, str], region: str, period: str, result: Any, value: float) -> None:
    row.update(
        {
            "period": period,
            "value": f"{value:.2f}",
            "forecast": f"{value:.2f}",
            "actual": "",
            "error": "",
            "status": "forecast",
            "signal": "Updated by GitHub Actions from the ConsumerSim pipeline.",
            "interpretation": interpretation(result.environment.get("combined_score", 0.0)),
        }
    )


def display_value(region: str, internal_score: float) -> float:
    return round(float(internal_score) + DISPLAY_OFFSETS[region], 2)


def month_label(month: str) -> str:
    year, month_number = (int(part) for part in month.split("-"))
    return f"{calendar.month_abbr[month_number]}-{str(year)[2:]}"


def previous_month_label(month: str) -> str:
    year, month_number = (int(part) for part in month.split("-"))
    if month_number == 1:
        year -= 1
        month_number = 12
    else:
        month_number -= 1
    return month_label(f"{year:04d}-{month_number:02d}")


def week_label(cutoff: date) -> str:
    week_number = ((cutoff.day - 1) // 7) + 1
    return f"{calendar.month_abbr[cutoff.month]} W{week_number}"


def next_monday(day: date) -> date:
    days_until_monday = (7 - day.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    return day + timedelta(days=days_until_monday)


def interpretation(signal: float) -> str:
    if signal > 0.05:
        return "The latest information environment points to firmer household sentiment."
    if signal < -0.05:
        return "The latest information environment points to softer household sentiment."
    return "The latest information environment points to a broadly stable confidence path."


def weekly_signal(signal: float) -> str:
    if signal > 0.05:
        return "Improving signal"
    if signal < -0.05:
        return "Softening signal"
    return "Stable signal"


if __name__ == "__main__":
    main()
