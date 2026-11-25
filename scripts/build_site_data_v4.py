"""Build site/data/consumersim_site_data.csv from frozen v4 results + live extrapolation.

Preserves the exact schema consumed by site/app.js AND keeps the legacy 2026-07-02
snapshot in the same CSV so visitors can switch seamlessly between:
- Live v4 (2014-2026 benchmark + real-time monthly & weekly extrapolation)
- Legacy v1 archive (as_of=2026-07-02)
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LEGACY_CSV = ROOT / "site" / "data" / "legacy_20260702_site_data.csv"
LIVE_JSON = ROOT / "live_pipeline" / "live_state" / "live_extrapolation.json"
OUT_CSV = ROOT / "site" / "data" / "consumersim_site_data.csv"

MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def fmt_period(yyyymm: int) -> str:
    y, m = divmod(int(yyyymm), 100)
    return f"{MONTH_ABBR[m]}-{y % 100:02d}"


def next_scheduled_update(as_of_iso: str) -> str:
    d = dt.date.fromisoformat(as_of_iso)
    # Next scheduled weekly/monthly refresh (next Monday, at least 1 day ahead)
    days_ahead = (7 - d.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return (d + dt.timedelta(days=days_ahead)).isoformat()


def metrics(df: pd.DataFrame, col: str) -> tuple[float, float, float]:
    y = df["Official"].to_numpy(dtype=float)
    yh = df[col].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(yh - y)))
    rmse = float(np.sqrt(np.mean((yh - y) ** 2)))
    r = float(np.corrcoef(yh, y)[0, 1])
    return mae, rmse, r


def blank_row(fieldnames: list[str], **kwargs) -> dict[str, str]:
    row = {k: "" for k in fieldnames}
    for k, v in kwargs.items():
        if k in row and v is not None:
            row[k] = str(v)
    return row


def tag_headline(title: str, trail: str) -> str:
    txt = f"{title} {trail}".lower()
    if re.search(r"inflat|price|grocery|cost of living|cpi|hicp", txt):
        return "Inflation"
    if re.search(r"fed|ecb|bank of japan|boj|interest rate|mortgage|central bank", txt):
        return "Policy"
    if re.search(r"job|unemploy|layoff|wage|payroll|labor|labour|hiring", txt):
        return "Labor"
    if re.search(r"stock|market|s&p|nikkei|stoxx|wall street|rally|sell-off|yen|euro", txt):
        return "Markets"
    if re.search(r"oil|gas|energy|fuel|brent|tariff|trade", txt):
        return "Energy & Trade"
    if re.search(r"election|congress|parliament|budget|tax|shutdown|trump|government", txt):
        return "Policy & Politics"
    return "Macro Driver"


def load_live_state() -> dict:
    if not LIVE_JSON.exists():
        return {}
    try:
        return json.loads(LIVE_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    with LEGACY_CSV.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = [f for f in (reader.fieldnames or []) if f]
        legacy_rows = [{k: row.get(k, "") for k in fieldnames} for row in reader]

    live = load_live_state()
    has_live = bool(live and live.get("us"))
    as_of = live.get("updated_at", "2026-09-30") if has_live else "2026-09-30"
    next_upd = next_scheduled_update(as_of)

    v4_rows: list[dict[str, str]] = []

    # 1. Meta
    v4_rows.append(blank_row(fieldnames, as_of=as_of, record_type="meta", sort_order=1, key="generatedAt", value=as_of))
    v4_rows.append(blank_row(fieldnames, as_of=as_of, record_type="meta", sort_order=2, key="nextUpdate", value=next_upd))

    region_specs = [
        {
            "id": "us", "order": 1, "label": "US", "market": "United States",
            "target": "Michigan ICS (native scale)",
            "window": "Jul 2014 - Sep 2026 (147 mo)" + (" + Live" if has_live else ""),
            "method": "ConsumerSim v4 (Microdata, zero-lambda)",
            "series_path": RES / "us/all_methods_series.csv",
            "leaderboard_methods": [
                ("ConsumerSim", "simulation"),
                ("SARIMAX Macro", "macro_time_series"),
                ("Auto ARIMA", "time_series"),
                ("MLP Neural (Macro+Text Emb)", "ml_text_embedding"),
                ("AR Ridge", "time_series"),
                ("Ridge Delta (Macro+Text Emb)", "ml_text_embedding"),
                ("Persistence", "benchmark"),
                ("Random Forest (Macro+Text Emb)", "ml_text_embedding"),
                ("Undated Direct-LLM (gpt-6-luna)", "direct_llm"),
                ("GBDT (LightGBM-eq, +Text Emb)", "ml_text_embedding"),
            ],
            "headline": "Unified zero-lambda microdata pipeline ranks #1 across 147 U.S. months and extrapolates live.",
        },
        {
            "id": "eu", "order": 2, "label": "EU27", "market": "European Union (27 member states)",
            "target": "Eurostat BS-CSMCI (native scale)",
            "window": "Jul 2014 - Sep 2026 (147 mo)" + (" + Live" if has_live else ""),
            "method": "ConsumerSim v4 (27-state stratified, N=1,350/mo)",
            "series_path": RES / "eu27/all_series_EU27_147m.csv",
            "leaderboard_methods": [
                ("ConsumerSim", "simulation"),
                ("Persistence", "benchmark"),
                ("Theta Model", "time_series"),
                ("GBDT (LightGBM-eq, Delta)", "ml_baseline"),
                ("AR Ridge", "time_series"),
                ("SARIMAX Macro", "macro_time_series"),
                ("Auto ARIMA", "time_series"),
                ("ETS Damped Trend", "time_series"),
            ],
            "headline": "Country-specific EU27 stratified panel ranks #1 over 147 months and reaches 0.67 OOS change correlation.",
        },
        {
            "id": "jp", "order": 3, "label": "Japan", "market": "Japan (8 regions)",
            "target": "Cabinet Office ESRI CCI (native scale)",
            "window": "Jul 2014 - Sep 2026 (147 mo)" + (" + Live" if has_live else ""),
            "method": "ConsumerSim v4 (8-region stratified, N=1,200/mo)",
            "series_path": RES / "japan/all_series_Japan_147m.csv",
            "leaderboard_methods": [
                ("ConsumerSim", "simulation"),
                ("Persistence", "benchmark"),
                ("Theta Model", "time_series"),
                ("Auto ARIMA", "time_series"),
                ("GBDT (LightGBM-eq, Delta)", "ml_baseline"),
                ("AR Ridge", "time_series"),
                ("ETS Damped Trend", "time_series"),
                ("SARIMAX Macro", "macro_time_series"),
            ],
            "headline": "Regional Japan panel ranks #1 over 147 months and reaches 0.64 OOS monthly-change correlation.",
        },
    ]

    for spec in region_specs:
        reg = spec["id"]
        df = pd.read_csv(spec["series_path"])
        mcol = df.columns[0]
        df[mcol] = df[mcol].astype(int)
        mae, rmse, pearson = metrics(df, "ConsumerSim")

        # Build combined series list: frozen OOS (2023-01..2026-09) + live extrapolated months
        tail_df = df[df[mcol] >= 202301].reset_index(drop=True)
        series_points: list[dict] = []
        for _, r in tail_df.iterrows():
            ym_i = int(r[mcol])
            series_points.append({
                "ym": ym_i,
                "period": fmt_period(ym_i),
                "forecast": float(r["ConsumerSim"]),
                "actual": float(r["Official"]),
            })

        live_reg = live.get(reg, {}) if has_live else {}
        for ym_s in sorted(live_reg.keys(), key=int):
            ym_i = int(ym_s)
            rec = live_reg[ym_s]
            if series_points and ym_i <= series_points[-1]["ym"]:
                continue
            series_points.append({
                "ym": ym_i,
                "period": fmt_period(ym_i),
                "forecast": float(rec["forecast"]),
                "actual": float(rec["official"]) if rec.get("official") is not None else None,
            })

        latest_pt = series_points[-1]
        prior_pt = series_points[-2]
        latest_period = latest_pt["period"]
        prior_period = prior_pt["period"]
        cs_val = latest_pt["forecast"]
        obs_val = latest_pt["actual"]
        prior_base = prior_pt["actual"] if prior_pt["actual"] is not None else prior_pt["forecast"]
        delta = cs_val - prior_base

        realized_note = f" (official {obs_val:.2f} realized)" if obs_val is not None else " (official release pending)"
        v4_rows.append(blank_row(
            fieldnames,
            as_of=as_of, record_type="monthly_prediction", region=reg, sort_order=spec["order"],
            label=spec["label"], period=latest_period,
            value=f"{cs_val:.2f}", value_label=f"{cs_val:.2f}", prior_period=prior_period,
            signal=f"{delta:+.2f} vs {prior_period} official ({prior_base:.2f}){realized_note}.",
            interpretation=spec["headline"],
        ))

        # Weekly prediction rows (strictly matching /^([A-Za-z]{3}) W\d+$/ and numeric cutoff_day)
        live_weekly = (live.get("weekly", {}).get(reg, {}) if has_live else {})
        if live_weekly:
            sorted_weeks = sorted(live_weekly.values(), key=lambda x: (int(x["month"]), int(x["cutoff_day"])))
            for idx, wrec in enumerate(sorted_weeks, 1):
                w_fc = float(wrec["forecast"])
                w_delta = float(wrec.get("delta", 0.0))
                w_sig = "Improving" if w_delta > 0.35 else ("Softening" if w_delta < -0.35 else "Stable")
                v4_rows.append(blank_row(
                    fieldnames,
                    as_of=as_of, record_type="weekly_prediction", region=reg, sort_order=idx,
                    label=spec["label"], period=fmt_period(int(wrec["month"])),
                    week_label=wrec["week_label"], cutoff_day=int(wrec["cutoff_day"]),
                    signal=w_sig,
                    interpretation=(
                        f"Live ConsumerSim v4 nowcast through {wrec['as_of']} "
                        f"(N={wrec.get('n', 200)} households, {w_delta:+.2f} vs {prior_period} anchor)."
                    ),
                    forecast=f"{w_fc:.2f}",
                    actual=f"{obs_val:.2f}" if obs_val is not None else "",
                    error=f"{w_fc - obs_val:+.2f}" if obs_val is not None else "",
                ))
        else:
            # Fallback if live_extrapolation.json hasn't been generated yet
            m_abbr = latest_period.split("-")[0]
            for idx, day in enumerate((7, 14, 21, 28), 1):
                v4_rows.append(blank_row(
                    fieldnames,
                    as_of=as_of, record_type="weekly_prediction", region=reg, sort_order=idx,
                    label=spec["label"], period=latest_period,
                    week_label=f"{m_abbr} W{idx}", cutoff_day=day,
                    signal="Frozen benchmark",
                    interpretation=f"ConsumerSim v4 benchmark evaluation for {latest_period}.",
                    forecast=f"{cs_val:.2f}",
                    actual=f"{obs_val:.2f}" if obs_val is not None else "",
                    error=f"{cs_val - obs_val:+.2f}" if obs_val is not None else "",
                ))

        # Region summary (benchmark metrics stay strictly 147-month apples-to-apples)
        v4_rows.append(blank_row(
            fieldnames,
            as_of=as_of, record_type="region_summary", region=reg, sort_order=spec["order"],
            label=spec["label"], market=spec["market"], target=spec["target"],
            window=spec["window"], method=spec["method"], period=latest_period,
            signal=f"Rank #1 of {len(spec['leaderboard_methods'])} methods (147-month benchmark)",
            interpretation=spec["headline"],
            months=len(df), mae=f"{mae:.3f}", rmse=f"{rmse:.3f}", pearson=f"{pearson:.3f}",
            forecast=f"{cs_val:.2f}",
            actual=f"{obs_val:.2f}" if obs_val is not None else "",
            error=f"{cs_val - obs_val:+.2f}" if obs_val is not None else "",
        ))

        # Leaderboard sorted by RMSE over the 147-month benchmark
        lb_entries = []
        for m_name, fam in spec["leaderboard_methods"]:
            if m_name in df.columns:
                m_mae, m_rmse, m_r = metrics(df, m_name)
                lb_entries.append((m_rmse, m_mae, m_r, m_name, fam))
        lb_entries.sort(key=lambda x: x[0])
        for rk, (m_rmse, m_mae, m_r, m_name, fam) in enumerate(lb_entries, 1):
            v4_rows.append(blank_row(
                fieldnames,
                as_of=as_of, record_type="leaderboard", region=reg, sort_order=rk,
                label=m_name, method=m_name, family=fam,
                rank=rk, months=len(df), mae=f"{m_mae:.3f}", rmse=f"{m_rmse:.3f}", pearson=f"{m_r:.3f}",
            ))

        # Series rows (2023-01 through latest live extrapolated month)
        for idx, pt in enumerate(series_points, 1):
            act_s = f"{pt['actual']:.2f}" if pt["actual"] is not None else ""
            err_s = f"{pt['forecast'] - pt['actual']:+.2f}" if pt["actual"] is not None else ""
            v4_rows.append(blank_row(
                fieldnames,
                as_of=as_of, record_type="series", region=reg, sort_order=idx,
                period=pt["period"],
                forecast=f"{pt['forecast']:.2f}",
                actual=act_s,
                error=err_s,
            ))

        # Real live news drivers (weekly + monthly)
        latest_ym_s = str(latest_pt["ym"])
        monthly_heads = (live.get("headlines", {}).get(reg, {}).get(latest_ym_s, []) if has_live else [])
        weekly_heads = (live.get("weekly_headlines", {}).get(reg, []) if has_live else []) or monthly_heads
        latest_w_label = (
            sorted(live_weekly.values(), key=lambda x: (int(x["month"]), int(x["cutoff_day"])))[-1]["week_label"]
            if live_weekly else latest_period
        )

        if weekly_heads:
            for idx, h in enumerate(weekly_heads[:3], 1):
                v4_rows.append(blank_row(
                    fieldnames,
                    as_of=as_of, record_type="forecast_news", region=reg, sort_order=idx,
                    key="weekly", label=h["title"], market=h.get("source", "News"),
                    period=latest_w_label, week_label=latest_w_label,
                    signal=tag_headline(h["title"], h.get("trail", "")),
                    interpretation=h.get("trail") or h["title"],
                    note=h.get("url", ""),
                ))
        if monthly_heads:
            for idx, h in enumerate(monthly_heads[:3], 4):
                v4_rows.append(blank_row(
                    fieldnames,
                    as_of=as_of, record_type="forecast_news", region=reg, sort_order=idx,
                    key="monthly", label=h["title"], market=h.get("source", "News"),
                    period=latest_period, week_label=latest_period,
                    signal=tag_headline(h["title"], h.get("trail", "")),
                    interpretation=h.get("trail") or h["title"],
                    note=h.get("url", ""),
                ))

    # Carry over illustrative agent_profile / agent_monthly_prediction (and fallback news only if no live headlines)
    have_live_news = any(r["record_type"] == "forecast_news" for r in v4_rows)
    for row in legacy_rows:
        rt = row["record_type"]
        if rt in ("agent_profile", "agent_monthly_prediction") or (rt == "forecast_news" and not have_live_news):
            copied = dict(row)
            copied["as_of"] = as_of
            v4_rows.append(copied)

    # Write v4 live snapshot cleanly
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(v4_rows)
    print(f"Wrote {OUT_CSV} (as_of={as_of}; {len(v4_rows)} v4 rows)")


if __name__ == "__main__":
    main()
