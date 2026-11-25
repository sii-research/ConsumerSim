"""Data access for ConsumerSim v4. Reads v3 raw/clean data read-only."""
import csv
import datetime as dt
import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V3 = Path(os.environ.get('CONSUMERSIM_DATA_DIR', str(ROOT / 'data')))
MICRO = V3 / 'raw/michigan/micro/csv_file.csv'
CACHE = ROOT / ('data/micro_test.pkl' if os.environ.get('CS_TEST') == '1' else 'data/micro.pkl')
Q = ['PAGO', 'PEXP', 'BUS12', 'BUS5', 'DUR']
ICC_Q = ['PAGO', 'DUR']
ICE_Q = ['PEXP', 'BUS12', 'BUS5']
FAV = {1, 2}
UNF = {4, 5}
NEU = {3}
# Fields never allowed into prompts (target answers of month t, reasons, news)
FORBIDDEN_IN_PROMPT = ['NEWS1', 'NEWS2', 'PAGOR1', 'PAGOR2', 'DURRN1', 'DURRN2', 'ICS', 'ICC', 'ICE']
FACT_COLS = ['AGE', 'SEX', 'REGION', 'EDUC', 'YTL3', 'HOMEOWN', 'INVEST', 'POLAFF', 'MARRY', 'NUMKID']
PRE_COVID = None  # set below
SHOCKS = [201611, 201901, 201908, 202003, 202004]  # T3 training-set shock months (k=2)


TEST_MODE = os.environ.get('CS_TEST') == '1'   # set only for the frozen-design test run (2023-01 to 2026-09)
MAX_MONTH = 202609 if TEST_MODE else 202212


def ym_add(m, n):
    z = m // 100 * 12 + m % 100 - 1 + n
    return z // 12 * 100 + z % 12 + 1


def month_range(a, b):
    out, m = [], a
    while m <= b:
        out.append(m)
        m = ym_add(m, 1)
    return out


PRE_COVID = month_range(201407, 202002)  # 68 scored months
COVID = month_range(202003, 202012)      # 10 scored months
ALL_SCORED = PRE_COVID + COVID
VALID = month_range(202101, 202212)      # validation set (plan A1); 2023+ is the test set, loaded only in TEST_MODE
TEST = month_range(202301, 202609)       # test set: micro data and Final Results PDFs end 2026-09
VAL_SHOCKS = [202108]                    # T3 k=2 shock month in the validation set


def state_of(code):
    if code in FAV:
        return 'F'
    if code in UNF:
        return 'U'
    if code in NEU:
        return 'N'
    return 'D'


@lru_cache(maxsize=1)
def micro():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    d = pd.read_csv(MICRO, usecols=['CASEID', 'YYYYMM', 'WT', *Q, *FACT_COLS], dtype=str)
    for c in ['YYYYMM', *Q]:
        d[c] = pd.to_numeric(d[c].str.strip(), errors='coerce').astype('Int64')
    d['WT'] = pd.to_numeric(d.WT, errors='coerce')
    for c in FACT_COLS:
        d[c] = pd.to_numeric(d[c].str.strip(), errors='coerce')
    d = d[d.YYYYMM.between(201201, MAX_MONTH)].copy()
    d['YYYYMM'] = d.YYYYMM.astype(int)
    for q in Q:
        d['S_' + q] = d[q].map(lambda c: state_of(int(c)) if pd.notna(c) else 'D')
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(d))
    return d


ATT_COLS = ['GOVT', 'BEXP', 'UNEMP', 'PX1', 'HOM', 'CAR', 'PEXP5', 'PINC', 'PJOB']


@lru_cache(maxsize=1)
def attitudes():
    """Other answers each respondent gave in their own interview (used only for personas drawn from months <= t-12)."""
    path = ROOT / 'data/attitudes.pkl'
    if path.exists():
        return pickle.loads(path.read_bytes())
    d = pd.read_csv(MICRO, usecols=['CASEID', 'YYYYMM', *ATT_COLS], dtype=str)
    d['CASEID'] = d.CASEID.str.strip()
    for c in ['YYYYMM', *ATT_COLS]:
        d[c] = pd.to_numeric(d[c].str.strip(), errors='coerce')
    d = d[d.YYYYMM.between(201001, MAX_MONTH - 100)]   # only months that can serve as t-12 pools
    out = {(r.CASEID, int(r.YYYYMM)): {c: getattr(r, c) for c in ATT_COLS} for r in d.itertuples()}
    path.write_bytes(pickle.dumps(out))
    return out


@lru_cache(maxsize=1)
def components():
    """Published Final Results values per month (RS for 5 questions, ICS/ICC/ICE, release date)."""
    c = pd.read_csv(ROOT / 'data/components.csv').set_index('month')
    if TEST_MODE:
        t = pd.read_csv(ROOT / 'data/components_test.csv').set_index('month')
        c = pd.concat([c, t[~t.index.isin(c.index)]])
    return c


@lru_cache(maxsize=1)
def official():
    """Official (final) monthly indices used as scoring target (same table as T3/T9)."""
    o = pd.read_csv(V3 / 'clean/T3_20261113/official/official_indices.csv').dropna(subset=['ICS'])
    o['month'] = o.month.astype(int)
    return o.set_index('month')[['ICS', 'ICC', 'ICE']]


@lru_cache(maxsize=1)
def dates():
    rows = {}
    with open(V3 / 'clean/T2_20261113/C2/monthly_dates_final.csv') as f:
        for r in csv.DictReader(f):
            rows[int(r['month'].replace('-', ''))] = r
    return rows


def cutoff(m):
    """Information cutoff for survey month m: last interview day (inclusive)."""
    return dt.date.fromisoformat(dates()[m]['cutoff_date'])


def interview_start(m):
    """Interview start; for the known inverted calendar entry (2020-11: start after end) use the day after the
    previous month's interview end."""
    s = dt.date.fromisoformat(dates()[m]['interview_start'])
    e = dt.date.fromisoformat(dates()[m]['interview_end'])
    if s > e:
        s = dt.date.fromisoformat(dates()[ym_add(m, -1)]['interview_end']) + dt.timedelta(days=1)
    return s


def published_last_month(t):
    """Components of month t-1 as printed in its Final Results report; checks availability before cutoff(t)."""
    p = ym_add(t, -1)
    c = components().loc[p]
    rel = dt.date.fromisoformat(c.final_release)
    assert rel <= cutoff(t), (t, rel, cutoff(t))
    return c


def rs(states, weights, q=None):
    """Relative score from arrays of states ('F','N','U','D') and weights."""
    w = np.asarray(weights, float)
    s = np.asarray(states)
    return 100.0 + 100.0 * (w[s == 'F'].sum() - w[s == 'U'].sum()) / w.sum()


def indices_from_rs(r):
    ics = sum(r[q] for q in Q) / 6.7558 + 2.0
    icc = sum(r[q] for q in ICC_Q) / 2.6424 + 2.0
    ice = sum(r[q] for q in ICE_Q) / 4.1134 + 2.0
    return {'ICS': ics, 'ICC': icc, 'ICE': ice}


def pool_months(t, k=1, need_party=False):
    """The k most recent survey months m <= t-12 (persona eligibility rule).
    need_party: only months where party affiliation was asked. T56 fix (2026-11-19): the threshold was 90% coverage, which
    excluded months 2024-08..2025-03 (asked, 87-90% answered); party is asked at 87-99% coverage and not asked at 0%."""
    d = micro()
    last = ym_add(t, -12)
    cands = sorted({m for m in d.YYYYMM.unique() if m <= last}, reverse=True)
    if need_party:
        cov = d.assign(ok=d.POLAFF.isin([1, 2, 3])).groupby('YYYYMM').ok.mean()
        cands = [m for m in cands if cov.get(m, 0) >= 0.5]
    return cands[:k]


def pool(t, k=1, need_party=False):
    d = micro()
    ms = pool_months(t, k, need_party)
    p = d[d.YYYYMM.isin(ms)].copy()
    # each source month contributes equal total weight
    p['w'] = p.WT / p.groupby('YYYYMM').WT.transform('sum') / len(ms)
    return p.reset_index(drop=True)
