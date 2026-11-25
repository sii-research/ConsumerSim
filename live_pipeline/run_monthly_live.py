"""ConsumerSim v4 Live Monthly Extrapolation Pipeline (US, EU27, Japan).

Runs frozen v4 zero-lambda household-inertia inference for post-paper months
(2026-10 onward), updates live_state/live_extrapolation.json, and rebuilds
site/data/consumersim_site_data.csv ready for GitHub Pages sync.

Features:
- 100% frozen v4 calibration constants (no refitting, zero aggregate lambda)
- Live official CCI fetch (Michigan SCA, Eurostat, FRED/ESRI) + macro + news (NYT/Guardian)
- Resumable, idempotent JSONL cache per region (already-computed months are skipped)
- Zero hardcoded secrets (reads OPENAI_API_KEY / LLM_API_KEY / FRED / NYT / GUARDIAN from env or .env)
- Optional --push flag to invoke live_pipeline/sync_to_github.sh after validation
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = Path(__file__).resolve().parent
STATE_DIR = LIVE_DIR / "live_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "consumersim"))
sys.path.insert(0, str(ROOT / "experiments" / "eu_jp"))

from llm_client import make_payload  # noqa: E402
from prompts import Q as US_Q, TOPICS as US_TOPICS, PERSPECTIVE as US_PERSP, SYSTEM as US_SYSTEM, parse_contrast  # noqa: E402
from run_eu_jp import EU_Q, JP_Q, PERSPECTIVE_REG, build_local_personas, parse_reg_contrast, clean_headline_dates  # noqa: E402

# Frozen v4 calibration constants (Appendix F / Table F3)
US_NORMS_CELL = {
    "PAGO": 0.4150193651,
    "PEXP": 0.4193558036,
    "BUS12": 0.4023107369,
    "BUS5": 0.2995138569,
    "DUR": 0.3576303238,
}
US_P_LARGE, US_P_SMALL = 0.30, 0.0225
EU_P_LARGE, EU_P_SMALL, EU_USE_CONS = 0.02, 0.005, False
JP_P_LARGE, JP_P_SMALL, JP_USE_CONS, JP_NORM_CONS = 0.12, 0.0, True, 0.24687500

SCORE_MAP = {"F": 1.0, "N": 0.0, "U": -1.0, "D": 0.0}
REGION_NAME = {1: "the West", 2: "the Midwest", 3: "the Northeast", 4: "the South"}
REGION_CPI = {3: "CUUR0100SA0", 2: "CUUR0200SA0", 4: "CUUR0300SA0", 1: "CUUR0400SA0"}
EDUC_TXT = {
    1: "less than a high school diploma",
    2: "a high school diploma",
    3: "some college",
    4: "a college degree",
    5: "a graduate degree",
    6: "a graduate degree",
}
PARTY_TXT = {1: "a Republican", 2: "a Democrat", 3: "an independent"}
INC_TXT = {1: "the bottom third", 2: "the middle third", 3: "the top third"}


def load_env_config() -> dict[str, str]:
    cfg: dict[str, str] = {}
    env_candidates = [
        Path(os.environ["CONSUMERSIM_ENV_FILE"]) if os.environ.get("CONSUMERSIM_ENV_FILE") else None,
        ROOT / ".env",
        ROOT.parent / ".env",
    ]
    for p in env_candidates:
        if p and p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.removeprefix("export ").split("=", 1)
                    cfg[k.strip().upper()] = v.strip().strip('"').strip("'")
            break
    for k, v in os.environ.items():
        cfg[k.upper()] = v
    return cfg


def prev_ym(ym: int, k: int = 1) -> int:
    y, m = ym // 100, ym % 100 - k
    while m <= 0:
        y -= 1
        m += 12
    return y * 100 + m


def ym_end_date(ym: int) -> dt.date:
    y, m = ym // 100, ym % 100
    today = dt.date.today()
    if y == today.year and m == today.month:
        return today
    nxt = dt.date(y + (1 if m == 12 else 0), 1 if m == 12 else m + 1, 1)
    return nxt - dt.timedelta(days=1)


# ---------------------------------------------------------------------------
# Live Official Targets + Macro + Market + News Fetchers
# ---------------------------------------------------------------------------

def fetch_fred_series(cli: httpx.Client, sid: str, api_key: str, start: str = "2024-01-01") -> pd.Series:
    r = cli.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": sid, "api_key": api_key, "file_type": "json", "observation_start": start},
    )
    r.raise_for_status()
    rows = []
    for o in r.json().get("observations", []):
        if o["value"] not in (".", ""):
            rows.append((pd.Timestamp(o["date"]), float(o["value"])))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(dict(rows)).sort_index()
    return s


def fetch_official_targets(cli: httpx.Client, fred_key: str) -> dict[str, dict[int, float]]:
    targets: dict[str, dict[int, float]] = {"us": {}, "eu": {}, "jp": {}}

    # Seed from frozen results first
    for reg, fn in [
        ("us", ROOT / "results/us/all_methods_series.csv"),
        ("eu", ROOT / "results/eu27/all_series_EU27_147m.csv"),
        ("jp", ROOT / "results/japan/all_series_Japan_147m.csv"),
    ]:
        df = pd.read_csv(fn)
        mcol = df.columns[0]
        for ym, val in zip(df[mcol].astype(int), df["Official"].astype(float)):
            targets[reg][int(ym)] = float(val)

    # US live: Michigan tbmics.csv (fastest official release) + FRED UMCSENT
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
    }
    try:
        r = cli.get("https://www.sca.isr.umich.edu/files/tbmics.csv", follow_redirects=True)
        if r.status_code == 200:
            for line in r.text.splitlines()[1:]:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3 and parts[0] in month_map:
                    try:
                        ym = int(parts[1]) * 100 + month_map[parts[0]]
                        targets["us"][ym] = float(parts[2])
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[WARN] Michigan tbmics.csv fetch failed: {e}")

    if fred_key:
        try:
            s = fetch_fred_series(cli, "UMCSENT", fred_key, start="2025-01-01")
            for d, v in s.items():
                targets["us"].setdefault(d.year * 100 + d.month, float(v))
        except Exception as e:
            print(f"[WARN] FRED UMCSENT failed: {e}")

        # Japan live: FRED CSCICP02JPM460S
        try:
            s = fetch_fred_series(cli, "CSCICP02JPM460S", fred_key, start="2025-01-01")
            for d, v in s.items():
                targets["jp"][d.year * 100 + d.month] = float(v)
        except Exception as e:
            print(f"[WARN] FRED Japan CCI failed: {e}")

    # EU27 live: Eurostat ei_bsco_m BS-CSMCI SA
    try:
        r = cli.get(
            "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ei_bsco_m",
            params={"geo": "EU27_2020", "indic": "BS-CSMCI", "s_adj": "SA", "sinceTimePeriod": "2025-01"},
        )
        if r.status_code == 200:
            j = r.json()
            t_idx = j.get("dimension", {}).get("time", {}).get("category", {}).get("index", {})
            vals = j.get("value", {})
            for ym_str, pos in t_idx.items():
                v = vals.get(str(pos))
                if v is not None and re.match(r"^\d{4}-\d{2}$", ym_str):
                    targets["eu"][int(ym_str.replace("-", ""))] = float(v)
    except Exception as e:
        print(f"[WARN] Eurostat EU27 CCI failed: {e}")

    return targets


def fetch_yahoo_series(cli: httpx.Client, symbol: str) -> pd.Series:
    r = cli.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={"range": "2y", "interval": "1d"},
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    rows = [
        (pd.Timestamp(dt.datetime.fromtimestamp(t, dt.timezone.utc).date()), float(c))
        for t, c in zip(ts, closes)
        if c is not None
    ]
    return pd.Series(dict(rows)).sort_index()


def fetch_eurostat_unemp(cli: httpx.Client) -> pd.Series:
    r = cli.get(
        "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/une_rt_m",
        params={"geo": "EU27_2020", "s_adj": "SA", "age": "TOTAL", "sex": "T", "unit": "PC_ACT", "sinceTimePeriod": "2024-01"},
    )
    r.raise_for_status()
    j = r.json()
    t_idx = j.get("dimension", {}).get("time", {}).get("category", {}).get("index", {})
    vals = j.get("value", {})
    rows = []
    for ym_str, pos in t_idx.items():
        v = vals.get(str(pos))
        if v is not None and re.match(r"^\d{4}-\d{2}$", ym_str):
            rows.append((pd.Timestamp(f"{ym_str}-15"), float(v)))
    return pd.Series(dict(rows)).sort_index()


def fetch_guardian_news(cli: httpx.Client, api_key: str, query_params: dict, lo: dt.date, hi: dt.date,
                        exclude_sections: set[str] | None = None) -> list[dict]:
    arts = []
    if not api_key:
        return arts
    excl = exclude_sections or {"sport", "football", "culture", "lifeandstyle", "commentisfree", "stage", "film", "music", "books"}
    # Fetch month-by-month so recent days never crowd out earlier months
    cur_y, cur_m = lo.year, lo.month
    seen_urls: set[str] = set()
    while (cur_y, cur_m) <= (hi.year, hi.month):
        m_lo = max(lo, dt.date(cur_y, cur_m, 1))
        nxt_y, nxt_m = (cur_y + 1, 1) if cur_m == 12 else (cur_y, cur_m + 1)
        m_hi = min(hi, dt.date(nxt_y, nxt_m, 1) - dt.timedelta(days=1))
        try:
            params = {
                "api-key": api_key,
                "from-date": m_lo.isoformat(),
                "to-date": m_hi.isoformat(),
                "show-fields": "headline,trailText",
                "page-size": 50,
                "order-by": "newest",
                **query_params,
            }
            r = cli.get("https://content.guardianapis.com/search", params=params)
            if r.status_code == 200:
                for a in r.json().get("response", {}).get("results", []):
                    sec = (a.get("sectionId") or "").lower()
                    if sec in excl:
                        continue
                    url = a.get("webUrl", "")
                    if url in seen_urls:
                        continue
                    title = ((a.get("fields") or {}).get("headline") or a.get("webTitle") or "").strip()
                    if "| letters" in title.lower() or "letters:" in title.lower():
                        continue
                    seen_urls.add(url)
                    d = dt.date.fromisoformat(a["webPublicationDate"][:10])
                    trail = re.sub(r"<[^>]+>", "", (a.get("fields") or {}).get("trailText") or "").strip()
                    arts.append({
                        "date": d, "title": title, "trail": trail, "url": url,
                        "source": "The Guardian", "is_biz": sec == "business",
                    })
        except Exception as e:
            print(f"[WARN] Guardian fetch {cur_y}-{cur_m:02d} failed: {e}")
        cur_y, cur_m = nxt_y, nxt_m
    return sorted(arts, key=lambda x: (x["date"], x.get("is_biz", False)), reverse=True)


def fetch_nyt_news(cli: httpx.Client, api_key: str, months_needed: list[tuple[int, int]]) -> list[dict]:
    arts = []
    if not api_key:
        return arts
    econ_re = re.compile(
        r"econom|inflat|price|tariff|job|unemploy|fed|rate|stock|market|wall street|"
        r"gas|oil|hous|mortgage|rent|wage|recession|gdp|consumer|tax|shutdown|budget|debt|election|congress|trump",
        re.I,
    )
    for y, m in sorted(set(months_needed)):
        try:
            r = cli.get(f"https://api.nytimes.com/svc/archive/v1/{y}/{m}.json", params={"api-key": api_key}, timeout=30)
            if r.status_code != 200:
                continue
            for doc in r.json().get("response", {}).get("docs", []):
                pub = (doc.get("pub_date") or "")[:10]
                if not re.match(r"^\d{4}-\d{2}-\d{2}$", pub):
                    continue
                headline = (doc.get("headline") or {}).get("main") or ""
                abstract = doc.get("abstract") or doc.get("snippet") or ""
                sec = (doc.get("section_name") or "").lower()
                is_front = str(doc.get("print_page") or "") == "1" and str(doc.get("print_section") or "").upper() == "A"
                if not (is_front or ("business" in sec or "economy" in sec or "u.s." in sec)):
                    continue
                if not econ_re.search(headline + " " + abstract):
                    continue
                arts.append({
                    "date": dt.date.fromisoformat(pub),
                    "title": headline,
                    "trail": abstract,
                    "url": doc.get("web_url", ""),
                    "source": "New York Times",
                    "front": is_front,
                })
        except Exception as e:
            print(f"[WARN] NYT Archive {y}/{m} failed: {e}")
    return sorted(arts, key=lambda x: (x.get("front", False), x["date"]), reverse=True)


def last_before(s: pd.Series, day: dt.date, default: float = 0.0) -> float:
    if s is None or len(s) == 0:
        return default
    sub = s[s.index <= pd.Timestamp(day)]
    return float(sub.iloc[-1]) if len(sub) else float(s.iloc[0])


def yoy_before(s: pd.Series, day: dt.date, default: float = 2.5) -> float:
    if s is None or len(s) < 13:
        return default
    sub = s[s.index <= pd.Timestamp(day)]
    if len(sub) == 0:
        return default
    last_d = sub.index[-1]
    prev_d = last_d - pd.DateOffset(years=1)
    prev_sub = sub[sub.index <= prev_d]
    if len(prev_sub) == 0:
        return default
    return float(100.0 * (sub.iloc[-1] / prev_sub.iloc[-1] - 1.0))


def select_news_lines(arts: list[dict], asof: dt.date, n: int = 12, label: str = "Major economic news over the past month:") -> tuple[str, list[dict]]:
    lo = asof - dt.timedelta(days=29)
    cand = [a for a in arts if lo <= a["date"] <= asof]
    picked_lines, picked_meta = [], []
    per_day: dict[dt.date, int] = {}
    seen_t: set[str] = set()
    for a in cand:
        if per_day.get(a["date"], 0) >= 2:
            continue
        tk = a["title"].strip().lower()[:60]
        if not tk or tk in seen_t:
            continue
        seen_t.add(tk)
        t = clean_headline_dates(a["title"])
        tr = clean_headline_dates(a["trail"])[:160]
        picked_lines.append(f"- {t}. {tr}".rstrip(". ") + ".")
        picked_meta.append({"title": a["title"], "trail": a["trail"][:200], "source": a["source"], "url": a["url"], "date": a["date"].isoformat()})
        per_day[a["date"]] = per_day.get(a["date"], 0) + 1
        if len(picked_lines) >= n:
            break
    if not picked_lines:
        return f"{label}\n- (no major headlines)", []
    return f"{label}\n" + "\n".join(picked_lines), picked_meta


# ---------------------------------------------------------------------------
# Live U.S. Cohort Builder + Frozen Scoring
# ---------------------------------------------------------------------------

def load_us_micro_pool(ym: int, n_cap: int = 1800) -> pd.DataFrame:
    """Load rolling quarter of 3 Michigan waves >= 12 months before ym (Methods 4.2.1)."""
    candidates = [
        Path(os.environ["CONSUMERSIM_DATA_DIR"]) / "micro_test.pkl" if os.environ.get("CONSUMERSIM_DATA_DIR") else None,
        ROOT / "data/micro_test.pkl",
        ROOT.parent / "data/micro_test.pkl",
    ]
    pkl = next((p for p in candidates if p and p.exists()), None)
    if pkl is None:
        raise FileNotFoundError("micro_test.pkl not found; set CONSUMERSIM_DATA_DIR")
    df = pickle.load(pkl.open("rb"))
    df = df[df["POLAFF"].isin([1, 2, 3]) & df["YTL3"].isin([1, 2, 3])].copy()
    cut_hi = prev_ym(ym, 12)
    cut_lo = prev_ym(ym, 14)
    sub = df[(df["YYYYMM"] >= cut_lo) & (df["YYYYMM"] <= cut_hi)].copy()
    if len(sub) < 200:
        # Fallback if future date exceeds local Michigan archive by >12 months: use latest 3 waves
        top3 = sorted(df["YYYYMM"].unique())[-3:]
        sub = df[df["YYYYMM"].isin(top3)].copy()
    if len(sub) > n_cap:
        sub = sub.sample(n=n_cap, random_state=int(ym)).copy()
    sub = sub.reset_index(drop=True)
    # Normalize weights and preserve Michigan S_* codes (1/2->F, 3->N, 4/5->U, else->D)
    w_col = "WT" if "WT" in sub.columns else ("WT_AD" if "WT_AD" in sub.columns else None)
    sub["w"] = sub[w_col].astype(float) if w_col else 1.0
    sub["w"] = sub["w"] / sub["w"].sum()
    code_map = {1: "F", 2: "F", 3: "N", 4: "U", 5: "U"}
    for q in US_Q:
        if "S_" + q not in sub.columns:
            sub["S_" + q] = sub[q].map(code_map).fillna("D")
    sub["pid"] = [f"{ym}-{i:05d}" for i in range(len(sub))]
    return sub


def anchor_us_pool_to_ics(df: pd.DataFrame, prev_ics: float, seed: int = 0) -> pd.DataFrame:
    """Anchor pool weighted relative scores minimally to prev_ics (same logic as assim.py)."""
    df = df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    cur_rs = {q: 100.0 * (df.loc[df["S_" + q] == "F", "w"].sum() - df.loc[df["S_" + q] == "U", "w"].sum()) + 100.0 for q in US_Q}
    cur_ics = sum(cur_rs.values()) / 6.7558 + 2.0
    needed_total_rs = (prev_ics - cur_ics) * 6.7558
    per_q_delta = needed_total_rs / 5.0

    jitter = rng.normal(0, 0.5, len(df))
    for q in US_Q:
        others = [x for x in US_Q if x != q]
        lean = sum(df["S_" + x].map(SCORE_MAP) for x in others) + jitter
        delta = per_q_delta / 100.0
        if abs(delta) < 1e-6:
            continue
        steps = [("U", "N", False), ("N", "F", False), ("U", "F", False)] if delta > 0 else [("F", "N", True), ("N", "U", True), ("F", "U", True)]
        rem = abs(delta)
        for src, dst, asc in steps:
            if rem <= 1e-9:
                break
            idx = df.index[df["S_" + q] == src]
            if len(idx) == 0:
                continue
            order = lean.loc[idx].sort_values(ascending=asc).index
            step_val = 2.0 if (src, dst) in (("U", "F"), ("F", "U")) else 1.0
            for i in order:
                if rem <= 1e-9:
                    break
                w = float(df.at[i, "w"])
                if w * step_val <= rem + 1e-9:
                    df.at[i, "S_" + q] = dst
                    rem -= w * step_val
    return df


def build_us_persona_text(row: pd.Series) -> str:
    age = int(row["AGE"]) if pd.notna(row.get("AGE")) else 45
    sex_w = "a man" if row.get("SEX") == 1 else "a woman"
    reg = int(row["REGION"]) if pd.notna(row.get("REGION")) else 4
    edu = int(row["EDUC"]) if pd.notna(row.get("EDUC")) else 3
    inc3 = int(row["YTL3"]) if pd.notna(row.get("YTL3")) else 2
    own = row.get("HOMEOWN") == 1
    stocks = row.get("INVEST") == 1
    party = int(row["POLAFF"]) if pd.notna(row.get("POLAFF")) else 3
    college = edu >= 4

    facts = [
        f"About you: You are {age} years old, {sex_w}.",
        f"You live in {REGION_NAME.get(reg, 'the South')}.",
        f"Your education: {EDUC_TXT.get(edu, 'some college')}.",
        f"Your household income is in {INC_TXT.get(inc3, 'the middle third')} of U.S. households.",
        "You own your home." if own else "You do not own your home.",
        "Your family has investments in the stock market." if stocks else "Your family has no investments in the stock market.",
        f"Politically, you usually think of yourself as {PARTY_TXT.get(party, 'an independent')}.",
    ]
    exp = []
    if party == 1:
        exp.append("Like many Republicans, you see the economy partly through a partisan lens: when Republicans gain or hold power you tend to feel more confident, and when Democrats win power you tend to be more skeptical and worried about where the economy is headed.")
    elif party == 2:
        exp.append("Like many Democrats, you see the economy partly through a partisan lens: when Democrats gain power or check a Republican administration you tend to feel more confident, and when Republicans hold unchecked power you tend to be more skeptical and worried.")
    else:
        exp.append("You don't strongly identify with either party, so who is in charge matters less to you than how things are actually going.")
    exp.append("Because your family owns stocks, swings in the stock market directly affect your household's wealth." if stocks
               else "Your family owns no stocks, so the stock market does not directly affect your finances.")
    if age >= 65:
        exp.append("At your age you are likely retired, so your income depends more on savings, pensions and Social Security than on a paycheck.")
    elif (not college) or age < 35:
        exp.append("People in jobs like yours are often among the first to be laid off when the economy weakens.")
    else:
        exp.append("Jobs like yours are usually less exposed to layoffs when the economy weakens.")
    if inc3 == 1:
        exp.append("With a household income in the bottom third, higher prices for essentials like food, gas and rent squeeze your budget hard.")
    elif inc3 == 2:
        exp.append("With a household income in the middle third, you notice higher prices for essentials but can usually absorb them.")
    else:
        exp.append("With a household income in the top third, you have some cushion against higher prices.")
    exp.append("As a homeowner, home values and mortgage rates matter to you." if own else "As a renter, rents and everyday prices hit you directly.")
    return " ".join(facts) + "\n\n" + " ".join(exp)


def build_us_snapshot(row: pd.Series, env_d: dict, diet: int) -> str:
    ud = lambda x: f"{'up' if x >= 0 else 'down'} {abs(x):.1f}%"
    hh = []
    if row.get("INVEST") == 1:
        hh.append(f"Your family's stock investments: {ud(env_d['spx_ret'])} over the past month")
    age = int(row["AGE"]) if pd.notna(row.get("AGE")) else 45
    college = (int(row["EDUC"]) if pd.notna(row.get("EDUC")) else 3) >= 4
    if age >= 65:
        f = env_d["fed"]
        hh.append(f"What savings earn (Federal Reserve's target rate): {max(f - 0.25, 0):.2f}% to {f:.2f}%")
    elif (not college) or age < 35:
        hh.append(f"New claims for unemployment benefits in the latest week: {round(env_d['icsa'], -3):,.0f}")
    if (int(row["YTL3"]) if pd.notna(row.get("YTL3")) else 2) in (1, 2):
        hh.append(f"Grocery prices: {ud(env_d['groc_yoy'])} from a year earlier")
        hh.append(f"Regular gasoline: ${env_d['gas']:.2f} a gallon")
    if row.get("HOMEOWN") == 1:
        hh.append(f"Home prices (Case-Shiller): {ud(env_d['home_yoy'])} from a year earlier")
        hh.append(f"30-year fixed mortgage rate: {env_d['mort']:.2f}%")
    else:
        hh.append(f"Rents: {ud(env_d['rent_yoy'])} from a year earlier")
    reg = int(row["REGION"]) if pd.notna(row.get("REGION")) else 4
    hh.append(f"Consumer prices in {REGION_NAME.get(reg, 'the South')}: {ud(env_d['reg_cpi'].get(reg, env_d['cpi_yoy']))} from a year earlier")

    parts = [env_d["pol_txt"]]
    if diet in (0, 1):
        parts.append(env_d["macro_txt"])
    if diet in (0, 2):
        parts.append(env_d["news_txt"])
    parts.append("For your household:\n" + "\n".join("- " + x for x in hh))
    return "\n\n".join(parts)


def step_us_state(s0: str, d: int, q: str) -> str:
    if d == 0:
        return s0
    if q in ("PAGO", "PEXP"):
        order = ["U", "N", "F"]
        idx = order.index(s0) if s0 in order else 1
        return order[min(2, idx + 1) if d > 0 else max(0, idx - 1)]
    return "F" if d > 0 else "U"


# ---------------------------------------------------------------------------
# Async LLM Runner (Unified for US, EU27, Japan)
# ---------------------------------------------------------------------------

async def run_us_month(ym: int, prev_ics: float, env_prev: dict, env_now: dict,
                       cfg: dict[str, str], n_cap: int = 1800, concurrency: int = 96,
                       cache_suffix: str = "") -> tuple[float, dict]:
    cache_p = STATE_DIR / f"sim_live_us_{ym}{cache_suffix}.jsonl"
    pool = anchor_us_pool_to_ics(load_us_micro_pool(ym, n_cap=n_cap), prev_ics, seed=ym)
    done: dict[str, dict] = {}
    if cache_p.exists():
        for line in cache_p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                done[obj["pid"]] = obj

    todo = [row for _, row in pool.iterrows() if row["pid"] not in done]
    print(f"[US {ym}{cache_suffix}] cached={len(done)}/{len(pool)}, todo={len(todo)}", flush=True)

    if todo:
        base_url = cfg.get("OPENAI_BASE_URL", cfg.get("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        api_key = cfg.get("OPENAI_API_KEY", cfg.get("LLM_API_KEY", ""))
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        q_block = "\n".join(f"{q}: {US_TOPICS[q]}" for q in US_Q)

        async def one(cli: httpx.AsyncClient, row: pd.Series):
            pid = row["pid"]
            diet = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16) % 3
            swap = (int(hashlib.md5(f"us_{ym}_{pid}".encode()).hexdigest()[:8], 16) % 2) == 1
            s_now = build_us_snapshot(row, env_now, diet)
            s_prev = build_us_snapshot(row, env_prev, diet)
            snap_a, snap_b = (s_now, s_prev) if swap else (s_prev, s_now)
            now_label = "A" if swap else "B"
            user_msg = "\n\n".join([
                build_us_persona_text(row),
                "Below are two undated snapshots of U.S. economic conditions and news about a month apart. "
                "They are labeled A and B in no particular order.",
                "SNAPSHOT A\n" + snap_a,
                "SNAPSHOT B\n" + snap_b,
                US_PERSP,
                q_block,
                'Return only a JSON object with exactly the keys "PAGO", "PEXP", "BUS12", "BUS5", "DUR". '
                'Each value is an object {"better": "A" or "B" or "Same", "size": "large" or "small" or "none"}.',
            ])
            payload = make_payload("gpt-6-luna", [{"role": "system", "content": US_SYSTEM}, {"role": "user", "content": user_msg}], max_tokens=240)
            async with sem:
                for att in range(4):
                    try:
                        r = await cli.post(url, headers=headers, json=payload)
                        r.raise_for_status()
                        txt = r.json()["choices"][0]["message"]["content"]
                        parsed, err = parse_contrast(txt, now_label)
                        if parsed is None:
                            raise ValueError(f"parse_fail:{err}")
                        rec = {"month": ym, "pid": pid, "parsed": parsed}
                        async with lock:
                            with cache_p.open("a", encoding="utf-8") as fp:
                                fp.write(json.dumps(rec) + "\n")
                        return rec
                    except Exception as e:
                        if att == 3:
                            rec = {"month": ym, "pid": pid, "parsed": {q: {"dir": 0, "size": 0} for q in US_Q}, "err": str(e)}
                            async with lock:
                                with cache_p.open("a", encoding="utf-8") as fp:
                                    fp.write(json.dumps(rec) + "\n")
                            return rec
                        await asyncio.sleep(1.0 * (att + 1))

        limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
        async with httpx.AsyncClient(timeout=60, limits=limits) as cli:
            res = await asyncio.gather(*(one(cli, row) for row in todo))
            for rec in res:
                done[rec["pid"]] = rec

    # Frozen v4 U.S. household-inertia scoring (Methods 4.5-4.6, zero aggregate lambda)
    cells = [f"{int(r['POLAFF'])}_{int(r['YTL3'])}" for _, r in pool.iterrows()]
    pool["cell"] = cells
    total_drs = 0.0
    q_drs = {}
    for q in US_Q:
        dirs = np.array([done[pid]["parsed"][q]["dir"] for pid in pool["pid"]], dtype=float)
        szs = np.array([done[pid]["parsed"][q].get("size") or 0 for pid in pool["pid"]], dtype=int)
        w = pool["w"].to_numpy(dtype=float)
        base_h = np.where(dirs == 0, 0.0, np.where(szs == 2, US_P_LARGE, np.where(szs == 1, US_P_SMALL, 0.0)))
        # Cell-weighted directional agreement (Party x Income third)
        tmp = pd.DataFrame({"cell": pool["cell"].values, "wd": w * dirs, "w": w})
        cell_mean = tmp.groupby("cell")["wd"].transform("sum") / tmp.groupby("cell")["w"].transform("sum")
        mult = np.minimum(1.5, 0.75 * np.abs(cell_mean.to_numpy()) / US_NORMS_CELL[q])
        h = np.minimum(0.95, base_h * mult)
        ds = np.array([SCORE_MAP[step_us_state(s0, int(d0), q)] - SCORE_MAP[s0] for s0, d0 in zip(pool["S_" + q], dirs)])
        drs = float(100.0 * np.sum(w * h * ds))
        q_drs[q] = drs
        total_drs += drs

    pred_ics = float(prev_ics + total_drs / 6.7558)
    return pred_ics, {"n": len(pool), "delta_ics": total_drs / 6.7558, "q_drs": q_drs}


async def run_regional_month(region: str, ym: int, prev_cci: float, env_prev: dict, env_now: dict,
                             cfg: dict[str, str], n_total: int, concurrency: int = 96,
                             cache_suffix: str = "") -> tuple[float, dict]:
    reg_key = "eu" if region == "EU27" else "jp"
    cache_p = STATE_DIR / f"sim_live_{reg_key}_{ym}{cache_suffix}.jsonl"
    personas = build_local_personas(region, n_total=n_total)
    q_dict = EU_Q if region == "EU27" else JP_Q
    q_keys = list(q_dict.keys())
    done: dict[str, dict] = {}
    if cache_p.exists():
        for line in cache_p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                done[obj["pid"]] = obj

    todo = [p for p in personas if p["pid"] not in done]
    print(f"[{region} {ym}{cache_suffix}] cached={len(done)}/{len(personas)}, todo={len(todo)}", flush=True)

    def reg_snap(p: dict, ed: dict) -> str:
        hh = []
        if region == "EU27":
            if p["eq"]:
                hh.append(f"Your household stock investments: {'up' if ed['eq_ch'] >= 0 else 'down'} {abs(ed['eq_ch']):.1f}% over the past month")
            if p["inc3"] in (1, 2):
                hh.append(f"Everyday consumer inflation: {ed['infl']:.1f}% year-on-year; energy oil benchmark ${ed['brent']:.1f}/bbl")
            if p["own"]:
                hh.append(f"ECB benchmark interest rate (affects mortgages): {ed['rate']:.2f}%")
            hh.append(f"EU unemployment rate: {ed['unemp']:.1f}%")
        else:
            if p["eq"]:
                hh.append(f"Your household stock investments (Nikkei): {'up' if ed['eq_ch'] >= 0 else 'down'} {abs(ed['eq_ch']):.1f}% over the past month")
            if p["inc3"] in (1, 2):
                hh.append(f"Yen exchange rate (import & grocery/energy cost driver): {ed['fx']:.1f} JPY/USD; oil ${ed['brent']:.1f}/bbl")
            hh.append(f"Japan unemployment rate: {ed['unemp']:.1f}%; BoJ policy rate: {ed['rate']:.2f}%")
        parts = []
        if p["diet"] in (0, 1):
            parts.append(ed["macro_txt"])
        if p["diet"] in (0, 2):
            parts.append(ed["news_txt"])
        parts.append("For your household:\n" + "\n".join("- " + x for x in hh))
        return "\n\n".join(parts)

    if todo:
        base_url = cfg.get("OPENAI_BASE_URL", cfg.get("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        api_key = cfg.get("OPENAI_API_KEY", cfg.get("LLM_API_KEY", ""))
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        sys_msg = (
            f"You are role-playing a specific {'European Union' if region == 'EU27' else 'Japanese'} adult answering the official "
            "monthly consumer confidence survey. Answer according to this person's situation and views. Do not say you are an AI."
        )
        keys_str = ", ".join(f'"{k}"' for k in q_keys)
        q_block = "\n".join(f"{k}: {v}" for k, v in q_dict.items())

        async def one(cli: httpx.AsyncClient, p: dict):
            h = int(hashlib.md5(f"{region}_{ym}_{p['pid']}".encode()).hexdigest()[:8], 16)
            swap = (h % 2) == 1
            s_now = reg_snap(p, env_now)
            s_prev = reg_snap(p, env_prev)
            snap_a, snap_b = (s_now, s_prev) if swap else (s_prev, s_now)
            now_label = "A" if swap else "B"
            user_msg = "\n\n".join([
                p["text"],
                "Below are two undated snapshots of economic conditions and regional news about a month apart. "
                "They are labeled A and B in no particular order.",
                "SNAPSHOT A\n" + snap_a,
                "SNAPSHOT B\n" + snap_b,
                PERSPECTIVE_REG,
                q_block,
                f"Return only a JSON object with exactly the keys {keys_str}. "
                'Each value is an object {"better": "A" or "B" or "Same", "size": "large" or "small" or "none"}.',
            ])
            payload = make_payload("gpt-6-luna", [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}], max_tokens=220)
            async with sem:
                for att in range(4):
                    try:
                        r = await cli.post(url, headers=headers, json=payload)
                        r.raise_for_status()
                        txt = r.json()["choices"][0]["message"]["content"]
                        parsed = parse_reg_contrast(txt, q_keys, now_label)
                        if parsed is None:
                            raise ValueError("parse_fail")
                        rec = {"month": ym, "pid": p["pid"], "parsed": parsed}
                        async with lock:
                            with cache_p.open("a", encoding="utf-8") as fp:
                                fp.write(json.dumps(rec) + "\n")
                        return rec
                    except Exception as e:
                        if att == 3:
                            rec = {"month": ym, "pid": p["pid"], "parsed": {q: {"dir": 0, "size": 0} for q in q_keys}, "err": str(e)}
                            async with lock:
                                with cache_p.open("a", encoding="utf-8") as fp:
                                    fp.write(json.dumps(rec) + "\n")
                            return rec
                        await asyncio.sleep(1.0 * (att + 1))

        limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
        async with httpx.AsyncClient(timeout=60, limits=limits) as cli:
            res = await asyncio.gather(*(one(cli, p) for p in todo))
            for rec in res:
                done[rec["pid"]] = rec

    # Frozen v4 EU27 / Japan household-inertia scoring
    l_sum, s_sum, d_all, n_votes = 0.0, 0.0, [], 0
    for p in personas:
        rec = done[p["pid"]]
        for q in q_keys:
            d = rec["parsed"][q]["dir"]
            sz = rec["parsed"][q]["size"]
            d_all.append(d)
            if sz == 2:
                l_sum += d
            elif sz == 1:
                s_sum += d
            n_votes += 1
    ml = l_sum / n_votes
    ms = s_sum / n_votes
    ac = abs(float(np.mean(d_all)))

    if region == "EU27":
        mult = 1.0
        delta = 100.0 * mult * (EU_P_LARGE * ml + EU_P_SMALL * ms)
    else:
        mult = min(1.5, 0.75 * (ac / (JP_NORM_CONS + 1e-6)))
        delta = 50.0 * mult * (JP_P_LARGE * ml + JP_P_SMALL * ms)

    return float(prev_cci + delta), {"n": len(personas), "delta": delta, "mean_l": ml, "mean_s": ms, "mult": mult}


# ---------------------------------------------------------------------------
# Main Live Orchestrator
# ---------------------------------------------------------------------------

def build_env_bundle(cli: httpx.Client, cfg: dict[str, str], months: list[int]) -> dict:
    fred_key = cfg.get("FRED_API_KEY", "")
    guard_key = cfg.get("GUARDIAN_API_KEY", "")
    nyt_key = cfg.get("NYT_API_KEY", "")

    print("[DATA] Fetching official US/EU27/JP targets and live FRED/Eurostat/Yahoo series...", flush=True)
    targets = fetch_official_targets(cli, fred_key)

    fred_ids = [
        "GASREGW", "SP500", "MORTGAGE30US", "DFEDTARU", "UNRATE", "CPIAUCSL",
        "ICSA", "CUUR0000SAF11", "CUUR0000SEHA", "CSUSHPINSA",
        "CUUR0100SA0", "CUUR0200SA0", "CUUR0300SA0", "CUUR0400SA0",
        "DCOILBRENTEU", "VIXCLS", "CP0000EZ19M086NEST", "ECBDFR", "DEXUSEU",
        "LRUNTTTTJPM156S", "IRSTCI01JPM156N", "DEXJPUS",
    ]
    fred = {}
    for sid in fred_ids:
        try:
            fred[sid] = fetch_fred_series(cli, sid, fred_key, start="2024-01-01")
        except Exception as e:
            print(f"[WARN] FRED {sid} failed: {e}")
            fred[sid] = pd.Series(dtype=float)

    try:
        eu_unemp = fetch_eurostat_unemp(cli)
    except Exception:
        eu_unemp = pd.Series({pd.Timestamp("2026-09-15"): 5.9})

    try:
        stoxx = fetch_yahoo_series(cli, "^STOXX50E")
    except Exception:
        stoxx = pd.Series({pd.Timestamp("2026-09-15"): 5000.0})

    try:
        nikkei = fetch_yahoo_series(cli, "^N225")
    except Exception:
        nikkei = pd.Series({pd.Timestamp("2026-09-15"): 39000.0})

    # News range covering all requested months and 45 days prior
    min_ym, max_ym = min(months), max(months)
    lo_date = dt.date(prev_ym(min_ym, 2) // 100, prev_ym(min_ym, 2) % 100, 1)
    hi_date = ym_end_date(max_ym)

    nyt_months = []
    cur = prev_ym(min_ym, 2)
    while cur <= max_ym:
        nyt_months.append((cur // 100, cur % 100))
        cur = (cur // 100 + 1) * 100 + 1 if cur % 100 == 12 else cur + 1

    print("[DATA] Fetching live NYT Archive & Guardian regional news...", flush=True)
    us_arts = fetch_nyt_news(cli, nyt_key, nyt_months)
    # Supplement with live Guardian US economy/market/policy news (covers current month before NYT Archive settles)
    us_arts.extend(fetch_guardian_news(
        cli, guard_key,
        {"tag": "business/us-economy|us-news/us-economy|business/stock-markets|us-news/us-politics"},
        lo_date, hi_date,
        exclude_sections={"sport", "football", "culture", "lifeandstyle", "commentisfree", "uk-news", "australia-news"},
    ))
    us_arts = sorted(us_arts, key=lambda x: (x["date"], x.get("front", False), x.get("is_biz", False)), reverse=True)

    eu_arts = fetch_guardian_news(
        cli, guard_key,
        {"tag": "business/eurozone|business/european-central-bank|world/eu|world/europe-news"},
        lo_date, hi_date,
        exclude_sections={"sport", "football", "culture", "lifeandstyle", "commentisfree", "uk-news", "australia-news", "politics"},
    )
    jp_arts = fetch_guardian_news(
        cli, guard_key,
        {"tag": "world/japan"},
        lo_date, hi_date,
        exclude_sections={"sport", "football", "culture", "lifeandstyle", "commentisfree", "uk-news", "australia-news"},
    )
    jp_extra = fetch_guardian_news(
        cli, guard_key,
        {"q": 'Japan OR yen OR Nikkei OR "Bank of Japan"', "query-fields": "headline,trailText"},
        lo_date, hi_date,
        exclude_sections={"sport", "football", "culture", "lifeandstyle", "commentisfree", "uk-news", "australia-news"},
    )
    seen_jp = {a["url"] for a in jp_arts}
    jp_arts.extend(a for a in jp_extra if a["url"] not in seen_jp)
    jp_arts = sorted(jp_arts, key=lambda x: (x["date"], x.get("is_biz", False)), reverse=True)
    print(f"[DATA] News loaded: US={len(us_arts)}, EU27={len(eu_arts)}, Japan={len(jp_arts)}", flush=True)

    return {
        "targets": targets, "fred": fred, "eu_unemp": eu_unemp,
        "stoxx": stoxx, "nikkei": nikkei,
        "us_arts": us_arts, "eu_arts": eu_arts, "jp_arts": jp_arts,
    }


def make_us_env_at(bundle: dict, asof: dt.date) -> tuple[dict, list[dict]]:
    f = bundle["fred"]
    gas = last_before(f["GASREGW"], asof, 3.20)
    spx = last_before(f["SP500"], asof, 6000.0)
    spx_p = last_before(f["SP500"], asof - dt.timedelta(days=30), spx)
    spx_ret = 100.0 * (spx / spx_p - 1.0)
    mort = last_before(f["MORTGAGE30US"], asof, 6.50)
    fed = last_before(f["DFEDTARU"], asof, 4.00)
    unrate = last_before(f["UNRATE"], asof - dt.timedelta(days=15), 4.2)
    cpi_yoy = yoy_before(f["CPIAUCSL"], asof - dt.timedelta(days=15), 2.5)
    icsa = last_before(f["ICSA"], asof, 225000.0)
    groc_yoy = yoy_before(f["CUUR0000SAF11"], asof - dt.timedelta(days=15), 2.0)
    rent_yoy = yoy_before(f["CUUR0000SEHA"], asof - dt.timedelta(days=15), 3.5)
    home_yoy = yoy_before(f["CSUSHPINSA"], asof - dt.timedelta(days=45), 3.0)
    reg_cpi = {r: yoy_before(f[sid], asof - dt.timedelta(days=15), cpi_yoy) for r, sid in REGION_CPI.items()}

    macro_lines = [
        f"Regular gasoline: ${gas:.2f} a gallon",
        f"S&P 500 stock index: {spx:,.0f}",
        f"Unemployment rate in the latest monthly jobs report: {unrate:.1f}%",
        f"Consumer prices in the latest report: up {cpi_yoy:.1f}% from a year earlier",
        f"30-year fixed mortgage rate: {mort:.2f}%",
        f"Federal Reserve's target interest rate: {max(fed - 0.25, 0):.2f}% to {fed:.2f}%",
    ]
    # Exact frozen v4 political context (matching consumersim/env.py; all election news comes from headlines)
    pol_txt = "Political situation:\n- President: Donald Trump (Republican)."
    news_txt, meta = select_news_lines(bundle["us_arts"], asof, n=15, label="Major national economic and political news stories over the past month:")
    return {
        "macro_txt": "Economic figures in the news:\n" + "\n".join("- " + x for x in macro_lines),
        "pol_txt": pol_txt, "news_txt": news_txt,
        "gas": gas, "spx": spx, "spx_ret": spx_ret, "mort": mort, "fed": fed,
        "unrate": unrate, "cpi_yoy": cpi_yoy, "icsa": icsa,
        "groc_yoy": groc_yoy, "rent_yoy": rent_yoy, "home_yoy": home_yoy, "reg_cpi": reg_cpi,
    }, meta


def make_reg_env_at(bundle: dict, region: str, asof: dt.date) -> tuple[dict, list[dict]]:
    f = bundle["fred"]
    lag_m = asof - dt.timedelta(days=25)
    brent = last_before(f["DCOILBRENTEU"], asof, 72.0)
    brent_p = last_before(f["DCOILBRENTEU"], asof - dt.timedelta(days=30), brent)
    brent_ch = 100.0 * (brent / brent_p - 1.0)
    vix = last_before(f["VIXCLS"], asof, 18.0)

    if region == "EU27":
        eq = last_before(bundle["stoxx"], asof, 5000.0)
        eq_p = last_before(bundle["stoxx"], asof - dt.timedelta(days=30), eq)
        eq_ch = 100.0 * (eq / eq_p - 1.0)
        fx = last_before(f["DEXUSEU"], asof, 1.08)
        unemp = last_before(bundle["eu_unemp"], lag_m, 5.9)
        rate = last_before(f["ECBDFR"], asof, 2.5)
        infl = yoy_before(f["CP0000EZ19M086NEST"], lag_m, 2.2)
        macro_lines = [
            f"Euro Stoxx 50 stock index: {eq:,.0f} ({'up' if eq_ch >= 0 else 'down'} {abs(eq_ch):.1f}% over the past month)",
            f"Consumer price inflation (HICP, latest report): {infl:.1f}% year-on-year",
            f"EU27 unemployment rate (latest report): {unemp:.1f}%",
            f"European Central Bank deposit interest rate: {rate:.2f}%",
            f"Brent crude oil (energy cost benchmark): ${brent:.1f}/barrel ({'up' if brent_ch >= 0 else 'down'} {abs(brent_ch):.1f}% over the past month)",
            f"Euro exchange rate: ${fx:.3f} per EUR; global market volatility (VIX): {vix:.1f}",
        ]
        news_txt, meta = select_news_lines(bundle["eu_arts"], asof, n=12, label="Regional economic news over the past month:")
        return {
            "macro_txt": "Economic figures in the news:\n" + "\n".join("- " + x for x in macro_lines),
            "news_txt": news_txt, "eq_ch": eq_ch, "infl": infl, "brent": brent, "rate": rate, "unemp": unemp, "fx": fx,
        }, meta
    else:
        eq = last_before(bundle["nikkei"], asof, 39000.0)
        eq_p = last_before(bundle["nikkei"], asof - dt.timedelta(days=30), eq)
        eq_ch = 100.0 * (eq / eq_p - 1.0)
        fx = last_before(f["DEXJPUS"], asof, 150.0)
        fx_p = last_before(f["DEXJPUS"], asof - dt.timedelta(days=30), fx)
        fx_ch = 100.0 * (fx / fx_p - 1.0)
        unemp = last_before(f["LRUNTTTTJPM156S"], lag_m, 2.5)
        rate = last_before(f["IRSTCI01JPM156N"], lag_m, 0.5)
        macro_lines = [
            f"Nikkei 225 stock index: {eq:,.0f} ({'up' if eq_ch >= 0 else 'down'} {abs(eq_ch):.1f}% over the past month)",
            f"Japanese yen exchange rate: {fx:.1f} JPY per USD ({'weakened' if fx_ch > 0 else 'strengthened'} {abs(fx_ch):.1f}% over the past month; weaker yen raises import & food/energy prices)",
            f"Japan unemployment rate (latest report): {unemp:.1f}%",
            f"Bank of Japan short-term policy/call rate: {rate:.2f}%",
            f"Brent crude oil (imported energy cost): ${brent:.1f}/barrel ({'up' if brent_ch >= 0 else 'down'} {abs(brent_ch):.1f}% over the past month)",
            f"Global financial market volatility (VIX): {vix:.1f}",
        ]
        news_txt, meta = select_news_lines(bundle["jp_arts"], asof, n=12, label="Regional economic news over the past month:")
        return {
            "macro_txt": "Economic figures in the news:\n" + "\n".join("- " + x for x in macro_lines),
            "news_txt": news_txt, "eq_ch": eq_ch, "infl": None, "brent": brent, "rate": rate, "unemp": unemp, "fx": fx,
        }, meta


async def run_pipeline(args: argparse.Namespace) -> None:
    cfg = load_env_config()
    for k in ("ALL_PROXY", "all_proxy"):
        os.environ.pop(k, None)

    today = dt.date.today()
    cur_ym = today.year * 100 + today.month
    if args.months:
        months = sorted(int(m) for m in args.months)
    else:
        months = []
        ym = 202610
        while ym <= cur_ym:
            months.append(ym)
            ym = (ym // 100 + 1) * 100 + 1 if ym % 100 == 12 else ym + 1

    with httpx.Client(timeout=30, follow_redirects=True) as cli:
        bundle = build_env_bundle(cli, cfg, months)

    state_file = STATE_DIR / "live_extrapolation.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    for k in ("us", "eu", "jp", "headlines", "weekly", "weekly_headlines"):
        state.setdefault(k, {})

    def resolve_prev(reg: str, pm: int) -> float:
        if pm in bundle["targets"][reg]:
            return float(bundle["targets"][reg][pm])
        prev_rec = state[reg].get(str(pm))
        if prev_rec is not None and prev_rec.get("forecast") is not None:
            return float(prev_rec["forecast"])
        raise KeyError(f"Missing anchor target for {reg} {pm}")

    for ym in months:
        pm = prev_ym(ym, 1)
        asof_now = min(dt.date(ym // 100, ym % 100, 15), today)
        asof_prev = asof_now - dt.timedelta(days=30)

        # --- US ---
        prev_us = resolve_prev("us", pm)
        env_us_prev, _ = make_us_env_at(bundle, asof_prev)
        env_us_now, us_heads = make_us_env_at(bundle, asof_now)
        pred_us, diag_us = await run_us_month(ym, prev_us, env_us_prev, env_us_now, cfg, n_cap=args.us_n, concurrency=args.concurrency)
        obs_us = bundle["targets"]["us"].get(ym)
        state["us"][str(ym)] = {
            "month": ym, "as_of": asof_now.isoformat(), "prev_official": prev_us,
            "forecast": round(pred_us, 4), "official": obs_us,
            "error": round(pred_us - obs_us, 4) if obs_us is not None else None,
            "n_households": diag_us["n"],
            "macro": {
                "spx": round(env_us_now["spx"], 1), "spx_ret": round(env_us_now["spx_ret"], 2),
                "gas": round(env_us_now["gas"], 2), "cpi_yoy": round(env_us_now["cpi_yoy"], 2),
                "unrate": round(env_us_now["unrate"], 2), "mort": round(env_us_now["mort"], 2),
            },
        }
        state["headlines"].setdefault("us", {})[str(ym)] = us_heads[:5]
        print(f"  => [US {ym}] prev={prev_us:.2f} | ConsumerSim={pred_us:.2f} | Official={obs_us}", flush=True)

        # --- EU27 ---
        prev_eu = resolve_prev("eu", pm)
        env_eu_prev, _ = make_reg_env_at(bundle, "EU27", asof_prev)
        env_eu_now, eu_heads = make_reg_env_at(bundle, "EU27", asof_now)
        pred_eu, diag_eu = await run_regional_month("EU27", ym, prev_eu, env_eu_prev, env_eu_now, cfg, n_total=args.eu_n, concurrency=args.concurrency)
        obs_eu = bundle["targets"]["eu"].get(ym)
        state["eu"][str(ym)] = {
            "month": ym, "as_of": asof_now.isoformat(), "prev_official": prev_eu,
            "forecast": round(pred_eu, 4), "official": obs_eu,
            "error": round(pred_eu - obs_eu, 4) if obs_eu is not None else None,
            "n_households": diag_eu["n"],
            "macro": {
                "eq_ch": round(env_eu_now["eq_ch"], 2), "infl": round(env_eu_now["infl"], 2),
                "unemp": round(env_eu_now["unemp"], 2), "brent": round(env_eu_now["brent"], 1),
                "rate": round(env_eu_now["rate"], 2), "fx": round(env_eu_now["fx"], 3),
            },
        }
        state["headlines"].setdefault("eu", {})[str(ym)] = eu_heads[:5]
        print(f"  => [EU27 {ym}] prev={prev_eu:.2f} | ConsumerSim={pred_eu:.2f} | Official={obs_eu}", flush=True)

        # --- Japan ---
        prev_jp = resolve_prev("jp", pm)
        env_jp_prev, _ = make_reg_env_at(bundle, "Japan", asof_prev)
        env_jp_now, jp_heads = make_reg_env_at(bundle, "Japan", asof_now)
        pred_jp, diag_jp = await run_regional_month("Japan", ym, prev_jp, env_jp_prev, env_jp_now, cfg, n_total=args.jp_n, concurrency=args.concurrency)
        obs_jp = bundle["targets"]["jp"].get(ym)
        state["jp"][str(ym)] = {
            "month": ym, "as_of": asof_now.isoformat(), "prev_official": prev_jp,
            "forecast": round(pred_jp, 4), "official": obs_jp,
            "error": round(pred_jp - obs_jp, 4) if obs_jp is not None else None,
            "n_households": diag_jp["n"],
            "macro": {
                "eq_ch": round(env_jp_now["eq_ch"], 2), "unemp": round(env_jp_now["unemp"], 2),
                "brent": round(env_jp_now["brent"], 1), "rate": round(env_jp_now["rate"], 2),
                "fx": round(env_jp_now["fx"], 1),
            },
        }
        state["headlines"].setdefault("jp", {})[str(ym)] = jp_heads[:5]
        print(f"  => [Japan {ym}] prev={prev_jp:.2f} | ConsumerSim={pred_jp:.2f} | Official={obs_jp}", flush=True)

    # --- Genuine Weekly Nowcast Cutoffs for the Latest Target Month ---
    latest_ym = max(months)
    ly, lm = divmod(latest_ym, 100)
    month_abbr = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][lm]
    month_last = ym_end_date(latest_ym)
    slot_days = []
    for w_idx, target_day in enumerate((7, 14, 21, 28), 1):
        d_cut = min(dt.date(ly, lm, min(target_day, month_last.day)), today)
        if not slot_days or d_cut > slot_days[-1][1]:
            slot_days.append((w_idx, d_cut))
    if not slot_days:
        slot_days = [(1, today)]

    pm_latest = prev_ym(latest_ym, 1)
    prev_map = {reg: resolve_prev(reg, pm_latest) for reg in ("us", "eu", "jp")}
    for reg in ("us", "eu", "jp"):
        state["weekly"][reg] = {}
    for w_idx, d_cut in slot_days:
        w_label = f"{month_abbr} W{w_idx}"
        d_prev = d_cut - dt.timedelta(days=30)
        sfx = f"_w{w_idx}"

        env_u_p, _ = make_us_env_at(bundle, d_prev)
        env_u_n, h_u = make_us_env_at(bundle, d_cut)
        p_u, d_u = await run_us_month(latest_ym, prev_map["us"], env_u_p, env_u_n, cfg, n_cap=args.weekly_n, concurrency=args.concurrency, cache_suffix=sfx)
        state["weekly"]["us"][w_label] = {"month": latest_ym, "week_label": w_label, "cutoff_day": d_cut.day, "as_of": d_cut.isoformat(), "forecast": round(p_u, 4), "delta": round(p_u - prev_map["us"], 4), "n": d_u["n"]}
        state["weekly_headlines"]["us"] = h_u[:5]

        env_e_p, _ = make_reg_env_at(bundle, "EU27", d_prev)
        env_e_n, h_e = make_reg_env_at(bundle, "EU27", d_cut)
        p_e, d_e = await run_regional_month("EU27", latest_ym, prev_map["eu"], env_e_p, env_e_n, cfg, n_total=args.weekly_n, concurrency=args.concurrency, cache_suffix=sfx)
        state["weekly"]["eu"][w_label] = {"month": latest_ym, "week_label": w_label, "cutoff_day": d_cut.day, "as_of": d_cut.isoformat(), "forecast": round(p_e, 4), "delta": round(p_e - prev_map["eu"], 4), "n": d_e["n"]}
        state["weekly_headlines"]["eu"] = h_e[:5]

        env_j_p, _ = make_reg_env_at(bundle, "Japan", d_prev)
        env_j_n, h_j = make_reg_env_at(bundle, "Japan", d_cut)
        p_j, d_j = await run_regional_month("Japan", latest_ym, prev_map["jp"], env_j_p, env_j_n, cfg, n_total=args.weekly_n, concurrency=args.concurrency, cache_suffix=sfx)
        state["weekly"]["jp"][w_label] = {"month": latest_ym, "week_label": w_label, "cutoff_day": d_cut.day, "as_of": d_cut.isoformat(), "forecast": round(p_j, 4), "delta": round(p_j - prev_map["jp"], 4), "n": d_j["n"]}
        state["weekly_headlines"]["jp"] = h_j[:5]

        print(f"  => [WEEKLY {w_label} ({d_cut})] US={p_u:.2f} | EU27={p_e:.2f} | JP={p_j:.2f}", flush=True)

    # Refresh any newly released official actuals for previously extrapolated months
    for reg in ("us", "eu", "jp"):
        for ym_s, rec in state[reg].items():
            obs = bundle["targets"][reg].get(int(ym_s))
            if obs is not None:
                rec["official"] = float(obs)
                rec["error"] = round(rec["forecast"] - float(obs), 4)

    state["updated_at"] = today.isoformat()
    state["latest_month"] = latest_ym
    state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Saved live state to {state_file}", flush=True)

    # Rebuild and validate site/data/consumersim_site_data.csv
    subprocess.run([sys.executable, str(ROOT / "scripts/build_site_data_v4.py")], check=True)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/refresh_site_data.py"),
        "--input", str(ROOT / "site/data/consumersim_site_data.csv"),
        "--output", str(ROOT / "site/data/consumersim_site_data.csv"),
    ], check=True)

    if args.push:
        subprocess.run(["bash", str(LIVE_DIR / "sync_to_github.sh"), str(ROOT / "site/data/consumersim_site_data.csv")], check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ConsumerSim v4 monthly live extrapolation.")
    p.add_argument("--months", nargs="*", type=int, help="Target YYYYMM months (default: 202610..current)")
    p.add_argument("--us-n", type=int, default=600, help="U.S. monthly cohort size (default 600; 1800 full paper scale)")
    p.add_argument("--eu-n", type=int, default=600, help="EU27 monthly cohort size (default 600; 1350 full paper scale)")
    p.add_argument("--jp-n", type=int, default=600, help="Japan monthly cohort size (default 600; 1200 full paper scale)")
    p.add_argument("--weekly-n", type=int, default=200, help="Weekly nowcast cutoff cohort size per region (default 200)")
    p.add_argument("--concurrency", type=int, default=96)
    p.add_argument("--push", action="store_true", help="Push updated CSV to origin/live-data after validation")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_pipeline(parse_args()))
