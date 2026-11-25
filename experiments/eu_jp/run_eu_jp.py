"""Module 1: Unified EU27 & Japan ConsumerSim Pipeline + Full Expanding-Window Baselines
(2020-01 to 2026-09, 81 months; also reports 2020-01..2026-03 [75m] and 2023-01..2026-09 [45m]).

Architecture is 100% unified with U.S. ConsumerSim:
- Country-specific stratified demographic personas (Eurostat / Japan e-Stat distributions)
- Undated A/B contrastive snapshots (30 days apart, randomized A/B order, 3 information diets)
- Household-level behavioral inertia (CCI_t = CCI_{t-1} + delta_CCI_t; ZERO aggregate lambda blending!)
- Calibration of household update-share strictly on development window (2020-01..2022-12), held fixed for 2023-01..2026-09.

All outputs live strictly in experiments/01_eu_jp/.
"""
import asyncio
import datetime as dt
import hashlib
import json
import os
import re
import sys
import warnings
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'consumersim'))
sys.path.insert(0, str(ROOT / 'consumersim'))
import engine
from llm_client import make_payload

OUT = ROOT / 'consumersim' / '01_eu_jp'
RAW = OUT / 'raw_data'
LEDGER = OUT / 'cost_ledger.jsonl'

EVAL_MONTHS = [y * 100 + m for y in range(2014, 2027) for m in range(1, 13) if 201407 <= y * 100 + m <= 202609]
DEV_MONTHS = [m for m in EVAL_MONTHS if m <= 202212]
OOS_MONTHS = [m for m in EVAL_MONTHS if m >= 202301]
POST_MONTHS = [m for m in EVAL_MONTHS if m >= 202601]

# Official survey question batteries
EU_Q = {
    'FS_LY': 'How the financial situation of your household has changed over the last 12 months',
    'FS_NY': 'How the financial position of your household will change over the next 12 months',
    'GES_NY': 'How the general economic situation in your country will develop over the next 12 months',
    'MP_PR': 'Whether you expect to spend more or less money on major purchases (furniture, electrical/electronic devices) over the next 12 months',
}
JP_Q = {
    'LIV': 'How your overall livelihood will change over the next 6 months',
    'INC': 'How the rate of increase in your household income will change over the next 6 months',
    'EMP': 'How the employment environment will change over the next 6 months',
    'DUR': 'Whether the next 6 months will be a good time to buy durable consumer goods (appliances, furniture, vehicles)',
}

PERSPECTIVE_REG = (
    "People in different economic situations often react quite differently to the same economic news. "
    "Compare the two snapshots from your own household's point of view: think about what the differences mean "
    "for your prices, job security, savings, housing costs, and outlook. "
    'For each topic below, say which snapshot you feel better about ("A", "B", or "Same"), '
    'and how big the difference is for your household ("large", "small", or "none").'
)


def build_local_personas(region, n_total=None):
    """Build Plan-B country-specific stratified household panel:
    - EU27: N=1,350 households/month covering ALL 27 EU member states (Eurostat 2024 population-weighted,
            averaging 50 households per member state) with Eurostat EU-SILC conditional demographics.
    - Japan: N=1,200 households/month covering ALL 8 Japanese regions (e-Stat 2024 population-weighted),
             matching Fig. 6's 1,200-household empirical saturation scale."""
    if n_total is None:
        n_total = 1350 if region == 'EU27' else 1200
    rng = np.random.default_rng(20261123 if region == 'EU27' else 20261124)
    personas = []
    if region == 'EU27':
        # Exact Eurostat 2024 population shares across ALL 27 EU member states (~449M total)
        countries = [
            ('Germany', 83.4), ('France', 68.4), ('Italy', 58.9), ('Spain', 48.6),
            ('Poland', 36.6), ('Romania', 19.1), ('Netherlands', 17.9), ('Belgium', 11.8),
            ('Czechia', 10.9), ('Sweden', 10.6), ('Portugal', 10.6), ('Greece', 10.4),
            ('Hungary', 9.6), ('Austria', 9.2), ('Bulgaria', 6.4), ('Denmark', 6.0),
            ('Finland', 5.6), ('Slovakia', 5.4), ('Ireland', 5.3), ('Croatia', 3.9),
            ('Lithuania', 2.9), ('Slovenia', 2.1), ('Latvia', 1.9), ('Estonia', 1.4),
            ('Cyprus', 0.9), ('Luxembourg', 0.7), ('Malta', 0.5),
        ]
        c_names, c_pops = zip(*countries)
        c_probs = np.array(c_pops) / sum(c_pops)
        for i in range(n_total):
            inc3 = (i % 3) + 1  # balanced 1/3 each income tercile
            age_band = ((i // 3) % 3)  # 0: 20-34, 1: 35-54, 2: 55-75
            age = int(rng.integers(20, 35) if age_band == 0 else (rng.integers(35, 55) if age_band == 1 else rng.integers(55, 76)))
            sex = 'man' if (i % 2 == 0) else 'woman'
            country = str(rng.choice(c_names, p=c_probs))
            # Realistic joint correlations (Eurostat EU-SILC structure)
            p_own = {1: 0.48, 2: 0.72, 3: 0.86}[inc3]
            p_eq = {1: 0.10, 2: 0.30, 3: 0.62}[inc3]
            p_col = {1: 0.18, 2: 0.38, 3: 0.65}[inc3]
            own = bool(rng.random() < p_own)
            eq = bool(rng.random() < p_eq)
            col = bool(rng.random() < p_col)
            diet = i % 3  # 0: macro+news, 1: macro, 2: news
            inc_str = {1: 'bottom third', 2: 'middle third', 3: 'top third'}[inc3]
            edu_str = 'university degree' if col else 'secondary school education'
            facts = (
                f"About you: You are a {age}-year-old {sex} living in {country} (European Union). "
                f"Your education: {edu_str}. Your household income is in the {inc_str} of households in your country. "
                f"{'You own your home.' if own else 'You rent your home.'} "
                f"{'Your household holds stocks or investment funds.' if eq else 'Your household holds no stocks or market investments.'}"
            )
            exp = []
            if eq:
                exp.append("Because your household owns investments, stock market swings affect your financial wealth.")
            if age >= 65:
                exp.append("You are retired, so your livelihood depends mainly on pensions, savings, and living costs.")
            elif not col or age < 35:
                exp.append("In economic downturns, jobs like yours face higher risk of layoffs or reduced hours.")
            if inc3 == 1:
                exp.append("With income in the bottom third, food, energy, and heating prices squeeze your monthly budget hard.")
            elif inc3 == 2:
                exp.append("With middle-tier income, you feel energy, food, and borrowing costs closely.")
            else:
                exp.append("With top-third income, you have more cushion against everyday price increases.")
            exp.append("As a homeowner, mortgage interest rates and energy bills matter to you." if own else "As a renter, rent and everyday living costs hit your budget directly.")
            personas.append({
                'pid': f'EU_{i:04d}', 'inc3': inc3, 'age': age, 'own': own, 'eq': eq, 'col': col, 'diet': diet,
                'cell': f'{inc3}_{int(own)}', 'text': facts + '\n\n' + ' '.join(exp),
            })
    else:
        # Exact Japan e-Stat 2024 population shares across ALL 8 geographic regions (~124M total)
        regions_jp = [
            ('Kanto (Tokyo metro)', 35.1), ('Chubu (Nagoya / central)', 17.0),
            ('Kansai (Osaka / Kyoto)', 16.5), ('Kyushu-Okinawa', 11.2),
            ('Tohoku', 6.7), ('Chugoku', 5.7), ('Hokkaido', 4.1), ('Shikoku', 2.9),
        ]
        r_names, r_pops = zip(*regions_jp)
        r_probs = np.array(r_pops) / sum(r_pops)
        for i in range(n_total):
            inc3 = (i % 3) + 1
            age_band = ((i // 3) % 3)  # 0: 22-39, 1: 40-59, 2: 60-78
            age = int(rng.integers(22, 40) if age_band == 0 else (rng.integers(40, 60) if age_band == 1 else rng.integers(60, 79)))
            sex = 'man' if (i % 2 == 0) else 'woman'
            reg_j = str(rng.choice(r_names, p=r_probs))
            p_own = {1: 0.45, 2: 0.65, 3: 0.80}[inc3]
            p_eq = {1: 0.08, 2: 0.25, 3: 0.55}[inc3]
            p_col = {1: 0.25, 2: 0.48, 3: 0.72}[inc3]
            own = bool(rng.random() < p_own)
            eq = bool(rng.random() < p_eq)
            col = bool(rng.random() < p_col)
            diet = i % 3
            inc_str = {1: 'bottom third', 2: 'middle third', 3: 'top third'}[inc3]
            edu_str = 'university degree' if col else 'high school or vocational education'
            facts = (
                f"About you: You are a {age}-year-old {sex} living in {reg_j}, Japan. "
                f"Your education: {edu_str}. Your household income is in the {inc_str} of Japanese households. "
                f"{'You own your home.' if own else 'You rent your home.'} "
                f"{'Your household holds Japanese stocks or NISA investment funds.' if eq else 'Your household keeps savings mainly in bank deposits with no stock investments.'}"
            )
            exp = []
            if eq:
                exp.append("Because your household owns stocks, Nikkei market moves directly affect your savings.")
            if age >= 65:
                exp.append("You are retired on pension and savings, so rising food, utility, and imported goods prices erode your purchasing power.")
            elif not col or age < 35:
                exp.append("Your livelihood depends closely on wage growth, bonuses, and job stability.")
            if inc3 == 1:
                exp.append("With income in the bottom third, food prices, electricity bills, and yen-driven import inflation squeeze your daily budget hard.")
            elif inc3 == 2:
                exp.append("With middle-tier income, you watch real wages versus everyday prices and exchange-rate costs closely.")
            else:
                exp.append("With top-third income, you have more buffer against everyday price increases and watch investments and durable-goods prices.")
            exp.append("As a homeowner, mortgage rates and household maintenance costs matter to you." if own else "As a renter, rent and monthly living costs matter most.")
            personas.append({
                'pid': f'JP_{i:04d}', 'inc3': inc3, 'age': age, 'own': own, 'eq': eq, 'col': col, 'diet': diet,
                'cell': f'{inc3}_{int(own)}', 'text': facts + '\n\n' + ' '.join(exp),
            })
    return personas


def load_series_csv(path, col=None):
    df = pd.read_csv(path, index_col=0)
    s = df[col] if col else df.iloc[:, 0]
    s = pd.to_numeric(s, errors='coerce').dropna()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def last_before(s, day):
    sub = s[s.index <= pd.Timestamp(day)]
    return float(sub.iloc[-1]) if len(sub) else np.nan


def load_guardian_articles(region):
    reg_code = 'eu' if region == 'EU27' else 'jp'
    p14 = RAW / 'guardian_news' / f'guardian_{reg_code}_2014_2026.json'
    p19 = RAW / 'guardian_news' / f'guardian_{reg_code}_2019_2026.json'
    raw = json.loads((p14 if p14.exists() else p19).read_text())
    arts = []
    for a in raw:
        d = dt.date.fromisoformat(a['webPublicationDate'][:10])
        title = (a.get('fields') or {}).get('headline') or a.get('webTitle') or ''
        trail = re.sub(r'<[^>]+>', '', (a.get('fields') or {}).get('trailText') or '').strip()
        arts.append({'date': d, 'title': title, 'trail': trail})
    return sorted(arts, key=lambda x: x['date'], reverse=True)


def clean_headline_dates(txt):
    txt = re.sub(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b', 'recently', txt)
    txt = re.sub(r'\b20[12]\d\b', 'this year', txt)
    return txt


def news_snapshot(arts, asof, n=12):
    lo = asof - dt.timedelta(days=29)
    cand = [a for a in arts if lo <= a['date'] <= asof]
    picked = []
    per_day = {}
    for a in cand:
        if per_day.get(a['date'], 0) >= 2:
            continue
        t = clean_headline_dates(a['title'])
        tr = clean_headline_dates(a['trail'])[:160]
        picked.append(f"- {t}. {tr}".rstrip('. ') + '.')
        per_day[a['date']] = per_day.get(a['date'], 0) + 1
        if len(picked) >= n:
            break
    if not picked:
        return 'Regional economic news over the past month:\n- (no major headlines)'
    return 'Regional economic news over the past month:\n' + '\n'.join(picked)


class RegionalEnv:
    def __init__(self, region):
        self.region = region
        self._asof_cache = {}
        self.arts = load_guardian_articles(region)
        self.brent = load_series_csv(RAW / 'fred/DCOILBRENTEU.csv')
        self.vix = load_series_csv(RAW / 'fred/VIXCLS.csv')
        if region == 'EU27':
            self.target = load_series_csv(RAW / 'eurostat_eu27_cci.csv', 'BS-CSMCI')
            self.hicp_idx = load_series_csv(RAW / 'fred/CP0000EZ19M086NEST.csv')
            self.hicp_yoy = (self.hicp_idx / self.hicp_idx.shift(12) - 1) * 100
            self.unemp = load_series_csv(RAW / 'eurostat_eu27_unemp.csv')
            self.rate = load_series_csv(RAW / 'fred/ECBDFR.csv')
            self.fx = load_series_csv(RAW / 'fred/DEXUSEU.csv')
            self.eq = load_series_csv(RAW / 'market/stoxx50e.csv')
        else:
            self.target = load_series_csv(RAW / 'jp_fred_CSCICP02JPM460S.csv')
            self.unemp = load_series_csv(RAW / 'fred/LRUNTTTTJPM156S.csv')
            self.rate = load_series_csv(RAW / 'fred/IRSTCI01JPM156N.csv')
            self.fx = load_series_csv(RAW / 'fred/DEXJPUS.csv')
            self.eq = load_series_csv(RAW / 'market/nikkei225.csv')

    def target_ym(self):
        return {d.year * 100 + d.month: float(v) for d, v in self.target.items()}

    def _get_asof(self, asof):
        if asof in self._asof_cache:
            return self._asof_cache[asof]
        lag_m = asof - dt.timedelta(days=25)
        brent = last_before(self.brent, asof)
        brent_30 = last_before(self.brent, asof - dt.timedelta(days=30))
        brent_ch = 100 * (brent / brent_30 - 1)
        vix = last_before(self.vix, asof)
        eq = last_before(self.eq, asof)
        eq_30 = last_before(self.eq, asof - dt.timedelta(days=30))
        eq_ch = 100 * (eq / eq_30 - 1)
        fx = last_before(self.fx, asof)
        fx_30 = last_before(self.fx, asof - dt.timedelta(days=30))
        fx_ch = 100 * (fx / fx_30 - 1)
        unemp = last_before(self.unemp, lag_m)
        rate = last_before(self.rate, asof if self.region == 'EU27' else lag_m)
        infl = last_before(self.hicp_yoy.dropna(), lag_m) if self.region == 'EU27' else None
        ntxt = news_snapshot(self.arts, asof, n=12)
        if self.region == 'EU27':
            macro_lines = [
                f"Euro Stoxx 50 stock index: {eq:,.0f} ({'up' if eq_ch >= 0 else 'down'} {abs(eq_ch):.1f}% over the past month)",
                f"Consumer price inflation (HICP, latest report): {infl:.1f}% year-on-year",
                f"EU27 unemployment rate (latest report): {unemp:.1f}%",
                f"European Central Bank deposit interest rate: {rate:.2f}%",
                f"Brent crude oil (energy cost benchmark): ${brent:.1f}/barrel ({'up' if brent_ch >= 0 else 'down'} {abs(brent_ch):.1f}% over the past month)",
                f"Euro exchange rate: ${fx:.3f} per EUR; global market volatility (VIX): {vix:.1f}",
            ]
        else:
            macro_lines = [
                f"Nikkei 225 stock index: {eq:,.0f} ({'up' if eq_ch >= 0 else 'down'} {abs(eq_ch):.1f}% over the past month)",
                f"Japanese yen exchange rate: {fx:.1f} JPY per USD ({'weakened' if fx_ch > 0 else 'strengthened'} {abs(fx_ch):.1f}% over the past month; weaker yen raises import & food/energy prices)",
                f"Japan unemployment rate (latest report): {unemp:.1f}%",
                f"Bank of Japan short-term policy/call rate: {rate:.2f}%",
                f"Brent crude oil (imported energy cost): ${brent:.1f}/barrel ({'up' if brent_ch >= 0 else 'down'} {abs(brent_ch):.1f}% over the past month)",
                f"Global financial market volatility (VIX): {vix:.1f}",
            ]
        val = {
            'macro_txt': 'Economic figures in the news:\n' + '\n'.join('- ' + x for x in macro_lines),
            'news_txt': ntxt, 'eq_ch': eq_ch, 'infl': infl, 'brent': brent, 'rate': rate, 'unemp': unemp, 'fx': fx,
        }
        self._asof_cache[asof] = val
        return val

    def snapshot_text(self, asof, persona):
        c = self._get_asof(asof)
        hh = []
        if self.region == 'EU27':
            if persona['eq']:
                hh.append(f"Your household stock investments: {'up' if c['eq_ch'] >= 0 else 'down'} {abs(c['eq_ch']):.1f}% over the past month")
            if persona['inc3'] in (1, 2):
                hh.append(f"Everyday consumer inflation: {c['infl']:.1f}% year-on-year; energy oil benchmark ${c['brent']:.1f}/bbl")
            if persona['own']:
                hh.append(f"ECB benchmark interest rate (affects mortgages): {c['rate']:.2f}%")
            hh.append(f"EU unemployment rate: {c['unemp']:.1f}%")
        else:
            if persona['eq']:
                hh.append(f"Your household stock investments (Nikkei): {'up' if c['eq_ch'] >= 0 else 'down'} {abs(c['eq_ch']):.1f}% over the past month")
            if persona['inc3'] in (1, 2):
                hh.append(f"Yen exchange rate (import & grocery/energy cost driver): {c['fx']:.1f} JPY/USD; oil ${c['brent']:.1f}/bbl")
            hh.append(f"Japan unemployment rate: {c['unemp']:.1f}%; BoJ policy rate: {c['rate']:.2f}%")

        parts = []
        if persona['diet'] in (0, 1):
            parts.append(c['macro_txt'])
        if persona['diet'] in (0, 2):
            parts.append(c['news_txt'])
        parts.append('For your household:\n' + '\n'.join('- ' + x for x in hh))
        return '\n\n'.join(parts)

    def feat_row(self, ym):
        # Mid-month cutoff day 15 (EU/JP surveys conducted roughly days 1-20 of month)
        d = dt.date(ym // 100, ym % 100, 15)
        lag_m = d - dt.timedelta(days=25)
        brent = last_before(self.brent, d)
        brent_p = last_before(self.brent, d - dt.timedelta(days=30))
        eq = last_before(self.eq, d)
        eq_p = last_before(self.eq, d - dt.timedelta(days=30))
        fx = last_before(self.fx, d)
        fx_p = last_before(self.fx, d - dt.timedelta(days=30))
        u = last_before(self.unemp, lag_m)
        u_p = last_before(self.unemp, lag_m - dt.timedelta(days=32))
        r = last_before(self.rate, d if self.region == 'EU27' else lag_m)
        r_p = last_before(self.rate, (d if self.region == 'EU27' else lag_m) - dt.timedelta(days=32))
        vix = last_before(self.vix, d)
        row = {
            'eq_log': np.log(max(eq, 1.0)), 'eq_ret': np.log(max(eq, 1.0) / max(eq_p, 1.0)),
            'brent_log': np.log(max(brent, 1.0)), 'brent_ret': np.log(max(brent, 1.0) / max(brent_p, 1.0)),
            'fx': fx, 'fx_ret': np.log(max(fx, 1e-3) / max(fx_p, 1e-3)),
            'unemp': u, 'd_unemp': u - u_p,
            'rate': r, 'd_rate': r - r_p,
            'vix': vix,
        }
        if self.region == 'EU27':
            infl = last_before(self.hicp_yoy.dropna(), lag_m)
            infl_p = last_before(self.hicp_yoy.dropna(), lag_m - dt.timedelta(days=32))
            row['infl'] = infl
            row['d_infl'] = infl - infl_p
        return row


def parse_reg_contrast(text, q_keys, now_label):
    if not text:
        return None
    s = text.strip()
    i, j = s.find('{'), s.rfind('}')
    if i < 0 or j < 0:
        return None
    try:
        obj = json.loads(s[i:j + 1])
    except Exception:
        return None
    out = {}
    for q in q_keys:
        v = obj.get(q)
        if not isinstance(v, dict):
            return None
        b = str(v.get('better', '')).strip().upper()
        sz = {'large': 2, 'big': 2, 'small': 1, 'none': 0}.get(str(v.get('size', '')).strip().lower(), 0)
        if b in ('SAME', 'NEITHER', 'ABOUT THE SAME'):
            d = 0
        elif b in ('A', 'B'):
            d = 1 if b == now_label else -1
        else:
            return None
        if sz == 0:
            d = 0
        out[q] = {'dir': d, 'size': sz}
    return out


async def run_region_sim(region, env_obj, personas):
    q_dict = EU_Q if region == 'EU27' else JP_Q
    q_keys = list(q_dict.keys())
    cache_p = OUT / f'sim_{region}_planB_{len(personas)}.jsonl'
    done = {}
    if cache_p.exists():
        for line in cache_p.read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            if 'err' not in o:
                done[(o['month'], o['pid'])] = o

    tasks_todo = [(m, p) for m in EVAL_MONTHS for p in personas if (m, p['pid']) not in done]
    print(f'[{region} N={len(personas)}] cached={len(done)}, todo={len(tasks_todo)}', flush=True)
    if not tasks_todo:
        return done

    key = engine._key()
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    url = f'{engine.ROUTE}/chat/completions'
    sem = asyncio.Semaphore(96)
    f_lock = asyncio.Lock()
    sys_msg = (
        f"You are the {'European Union' if region == 'EU27' else 'Japanese'} resident described below, "
        f"taking part in the official {'European Commission / Eurostat' if region == 'EU27' else 'Cabinet Office ESRI'} "
        "monthly consumer confidence survey. Answer according to this person's situation and views. Do not say you are an AI."
    )
    keys_str = ', '.join(f'"{k}"' for k in q_keys)
    q_block = '\n'.join(f'{k}: {v}' for k, v in q_dict.items())

    # Pre-cache macro+news text per (month, diet, persona_signature) to speed up prompt building
    async def call_one(cli, m, p):
        asof_now = dt.date(m // 100, m % 100, 15)
        asof_prev = asof_now - dt.timedelta(days=30)
        h = int(hashlib.md5(f'{region}_{m}_{p["pid"]}'.encode()).hexdigest()[:8], 16)
        swap = (h % 2) == 1
        s_now = env_obj.snapshot_text(asof_now, p)
        s_prev = env_obj.snapshot_text(asof_prev, p)
        snap_a, snap_b = (s_now, s_prev) if swap else (s_prev, s_now)
        now_label = 'A' if swap else 'B'
        user_msg = '\n\n'.join([
            p['text'],
            'Below are two undated snapshots of economic conditions and regional news about a month apart. '
            'They are labeled A and B in no particular order.',
            'SNAPSHOT A\n' + snap_a,
            'SNAPSHOT B\n' + snap_b,
            PERSPECTIVE_REG,
            q_block,
            f'Return only a JSON object with exactly the keys {keys_str}. '
            'Each value is an object {"better": "A" or "B" or "Same", "size": "large" or "small" or "none"}.'
        ])
        async with sem:
            for attempt in range(4):
                try:
                    r = await cli.post(url, headers=headers, json=make_payload(
                        'gpt-6-luna', [{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': user_msg}], max_tokens=220
                    ))
                    r.raise_for_status()
                    data = r.json()
                    txt = data['choices'][0]['message']['content']
                    parsed = parse_reg_contrast(txt, q_keys, now_label)
                    if parsed is None:
                        raise ValueError('parse_fail')
                    u = data.get('usage', {})
                    rec = {'month': m, 'pid': p['pid'], 'cell': p['cell'], 'parsed': parsed}
                    async with f_lock:
                        with open(cache_p, 'a') as fp:
                            fp.write(json.dumps(rec) + '\n')
                        with open(LEDGER, 'a') as fp:
                            fp.write(json.dumps({'region': region, 'in': u.get('prompt_tokens', 0),
                                                 'out': u.get('completion_tokens', 0)}) + '\n')
                    return rec
                except Exception as e:
                    if attempt == 3:
                        rec = {'month': m, 'pid': p['pid'], 'cell': p['cell'],
                               'parsed': {q: {'dir': 0, 'size': 0} for q in q_keys}, 'err': str(e)}
                        async with f_lock:
                            with open(cache_p, 'a') as fp:
                                fp.write(json.dumps(rec) + '\n')
                        return rec
                    await asyncio.sleep(1.0 * (attempt + 1))

    limits = httpx.Limits(max_connections=96, max_keepalive_connections=96)
    async with httpx.AsyncClient(timeout=45, limits=limits) as cli:
        # Process in chunks of 5000 so progress is visible
        for i in range(0, len(tasks_todo), 5000):
            chunk = tasks_todo[i:i + 5000]
            res = await asyncio.gather(*(call_one(cli, m, p) for m, p in chunk))
            for rec in res:
                done[(rec['month'], rec['pid'])] = rec
            print(f'[{region} N={len(personas)}] progress: {len(done)} / {len(EVAL_MONTHS) * len(personas)}', flush=True)
    return done


def score_region_sim(region, done, target_ym, personas):
    """Exact unified household-level inertia scoring (NO aggregate lambda blending!).
    Calibrate household update step (p_large, p_small) strictly on DEV_MONTHS (2020-01..2022-12)
    on a fixed grid, then hold fixed across all 81 months (2020-01..2026-09)."""
    q_keys = list((EU_Q if region == 'EU27' else JP_Q).keys())

    # Aggregate raw household directional votes per month: large_net_m, small_net_m, consensus_m
    # Also compute cell-level consensus multiplier exactly like U.S.
    m_stats = {}
    for m in EVAL_MONTHS:
        l_sum, s_sum, d_all = 0.0, 0.0, []
        n = 0
        for p in personas:
            rec = done[(m, p['pid'])]
            for q in q_keys:
                d = rec['parsed'][q]['dir']
                sz = rec['parsed'][q]['size']
                d_all.append(d)
                if sz == 2:
                    l_sum += d
                elif sz == 1:
                    s_sum += d
                n += 1
        mean_l = l_sum / n
        mean_s = s_sum / n
        abs_cons = abs(np.mean(d_all))
        m_stats[m] = (mean_l, mean_s, abs_cons)

    # Normalizing constant of consensus measured ONLY on DEV_MONTHS
    norm_cons = float(np.median([m_stats[m][2] for m in DEV_MONTHS])) + 1e-6

    def build_series(step_large, step_small, use_cons=True):
        # Official scale: EU balance is [-100, +100] (100pt step); JP index is [0, 100] (50pt half-step)
        scale = 100.0 if region == 'EU27' else 50.0
        out = {}
        for m in EVAL_MONTHS:
            pm = m - 1 if m % 100 > 1 else (m // 100 - 1) * 100 + 12
            prev = target_ym[pm]
            ml, ms, ac = m_stats[m]
            mult = min(1.5, 0.75 * (ac / norm_cons)) if use_cons else 1.0
            delta = scale * mult * (step_large * ml + step_small * ms)
            out[m] = prev + delta
        return out

    # Fixed grid calibration on DEV_MONTHS ONLY (minimizing dev RMSE)
    best = None
    for sl in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
        for ss in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]:
            if ss > sl * 0.5:
                continue
            for uc in (True, False):
                ser = build_series(sl, ss, uc)
                rmse_dev = float(np.sqrt(np.mean([(ser[m] - target_ym[m]) ** 2 for m in DEV_MONTHS])))
                if best is None or rmse_dev < best[0]:
                    best = (rmse_dev, sl, ss, uc, ser)
    rmse_dev, sl, ss, uc, best_ser = best
    print(f'[{region}] Dev-calibrated household inertia (2020-2022): p_large={sl}, p_small={ss}, consensus={uc}, dev_RMSE={rmse_dev:.3f}')
    return best_ser, {'p_large': sl, 'p_small': ss, 'use_consensus': uc, 'dev_rmse': rmse_dev}


def run_regional_baselines(region, env_obj, target_ym):
    """Run exact same expanding-window baselines as U.S. Table 1 (history starting 2010-04)."""
    hist = [y * 100 + m for y in range(2010, 2027) for m in range(1, 13) if 201004 <= y * 100 + m <= 202609]
    F = pd.DataFrame({m: env_obj.feat_row(m) for m in hist}).T.ffill().bfill()
    y = pd.Series({m: target_ym[m] for m in hist})

    def lag(k):
        def shift_m(m, kk):
            yy, mm = m // 100, m % 100 - kk
            while mm <= 0:
                yy -= 1; mm += 12
            return yy * 100 + mm
        return pd.Series({m: target_ym[shift_m(m, k)] for m in hist})

    L1, L2, L3 = lag(1), lag(2), lag(3)
    sx_cols = ['d_unemp', 'eq_ret', 'brent_ret', 'd_rate'] + (['infl'] if region == 'EU27' else ['fx_ret'])
    all_cols = list(F.columns)
    stable_cols = ['unemp', 'rate'] + (['infl'] if region == 'EU27' else ['fx'])

    def ridge_pred(X, yy, xt):
        sc = StandardScaler().fit(X)
        r = RidgeCV(alphas=np.logspace(-2, 4, 15)).fit(sc.transform(X), yy)
        return float(r.predict(sc.transform(xt[None, :]))[0])

    methods = [
        'Persistence', 'AR Ridge', 'SARIMAX Macro', 'Auto ARIMA', 'ETS Damped Trend',
        'Theta Model', '3-Month Mean', 'Macro-Market Ridge', 'Macro Stable Ridge',
        'GBDT (LightGBM-eq, Delta)', 'Prophet Univ.', 'Prophet Macro',
    ]
    res = {k: {} for k in methods}
    orders = {}
    for t in EVAL_MONTHS:
        tr = [m for m in hist if m < t]
        yy = y[tr].values
        res['Persistence'][t] = float(L1[t])
        res['3-Month Mean'][t] = float(np.mean([L1[t], L2[t], L3[t]]))
        res['AR Ridge'][t] = ridge_pred(np.c_[L1[tr], L2[tr], L3[tr]], yy, np.array([L1[t], L2[t], L3[t]]))
        res['Macro-Market Ridge'][t] = ridge_pred(F.loc[tr, all_cols].values, yy, F.loc[t, all_cols].values)
        res['Macro Stable Ridge'][t] = ridge_pred(F.loc[tr, stable_cols].values, yy, F.loc[t, stable_cols].values)

        # GBDT Delta (LightGBM-equivalent)
        dy_tr = (y[tr] - L1[tr]).values
        X_gb = np.c_[L1[tr], L1[tr] - L2[tr], F.loc[tr, all_cols].values]
        x_gb_t = np.r_[L1[t], L1[t] - L2[t], F.loc[t, all_cols].values]
        sc_g = StandardScaler().fit(X_gb)
        gb = HistGradientBoostingRegressor(max_iter=80, max_depth=3, learning_rate=0.05, min_samples_leaf=5,
                                           l2_regularization=5.0, random_state=42)
        gb.fit(sc_g.transform(X_gb), dy_tr)
        res['GBDT (LightGBM-eq, Delta)'][t] = float(L1[t]) + float(gb.predict(sc_g.transform(x_gb_t[None, :]))[0])

        # SARIMAX & Auto ARIMA (AIC selection each January)
        ex_tr = F.loc[tr, sx_cols].values
        ex_t = F.loc[[t], sx_cols].values
        if t % 100 == 1 or 'sx' not in orders:
            for kind in ('sx', 'ar'):
                cands = []
                for p in (0, 1, 2):
                    for dd in (0, 1):
                        for q in (0, 1, 2):
                            if p + q == 0:
                                continue
                            try:
                                fit = SARIMAX(yy, exog=ex_tr if kind == 'sx' else None, order=(p, dd, q),
                                              trend='c' if dd == 0 else 'n').fit(disp=False)
                                cands.append((fit.aic, (p, dd, q)))
                            except Exception:
                                pass
                orders[kind] = min(cands)[1] if cands else (1, 1, 0)
        try:
            o = orders['sx']
            res['SARIMAX Macro'][t] = float(SARIMAX(yy, exog=ex_tr, order=o, trend='c' if o[1] == 0 else 'n').fit(disp=False).forecast(1, exog=ex_t)[0])
        except Exception:
            res['SARIMAX Macro'][t] = float(L1[t])
        try:
            o = orders['ar']
            res['Auto ARIMA'][t] = float(SARIMAX(yy, order=o, trend='c' if o[1] == 0 else 'n').fit(disp=False).forecast(1)[0])
        except Exception:
            res['Auto ARIMA'][t] = float(L1[t])
        try:
            res['ETS Damped Trend'][t] = float(ExponentialSmoothing(yy, trend='add', damped_trend=True).fit().forecast(1)[0])
        except Exception:
            res['ETS Damped Trend'][t] = float(L1[t])
        try:
            res['Theta Model'][t] = float(ThetaModel(yy, period=12, deseasonalize=False).fit().forecast(1).iloc[0])
        except Exception:
            res['Theta Model'][t] = float(L1[t])

    # Prophet baselines
    try:
        from prophet import Prophet
        import logging
        logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
        logging.getLogger('prophet').setLevel(logging.ERROR)
        ds = pd.Series({m: pd.Timestamp(f'{m // 100}-{m % 100:02d}-01') for m in hist})
        for t in EVAL_MONTHS:
            tr = [m for m in hist if m < t]
            dfp = pd.DataFrame({'ds': ds[tr].values, 'y': y[tr].values})
            p1 = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False).fit(dfp)
            res['Prophet Univ.'][t] = float(p1.predict(pd.DataFrame({'ds': [ds[t]]})).yhat.iloc[0])
            dfp2 = dfp.copy()
            for c in sx_cols:
                dfp2[c] = F.loc[tr, c].values
            p2 = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            for c in sx_cols:
                p2.add_regressor(c)
            p2.fit(dfp2)
            nt = pd.DataFrame({'ds': [ds[t]]} | {c: [float(F.loc[t, c])] for c in sx_cols})
            res['Prophet Macro'][t] = float(p2.predict(nt).yhat.iloc[0])
    except Exception as e:
        print(f'[{region}] Prophet skipped:', e)
        res.pop('Prophet Univ.', None)
        res.pop('Prophet Macro', None)

    return pd.DataFrame(res)


def eval_table(region, sim_ser, base_df, target_ym, months, tag):
    df = base_df.loc[months].copy()
    df.insert(0, 'ConsumerSim', [sim_ser[m] for m in months])
    off = np.array([target_ym[m] for m in months])
    pers = df['Persistence'].values
    d_off = off - pers
    rows = []
    for col in df.columns:
        pred = df[col].values
        e = pred - off
        dp = pred - pers
        mae = float(np.mean(np.abs(e)))
        rmse = float(np.sqrt(np.mean(e ** 2)))
        r = float(np.corrcoef(pred, off)[0, 1]) if np.std(pred) > 1e-9 else np.nan
        rho = float(pd.Series(pred).corr(pd.Series(off), method='spearman')) if np.std(pred) > 1e-9 else np.nan
        dr = float(np.corrcoef(dp, d_off)[0, 1]) if np.std(dp) > 1e-9 else np.nan
        rows.append({'Method': col, 'MAE': mae, 'RMSE': rmse, 'Pearson r': r, 'Spearman rho': rho, 'Delta r': dr})
    tbl = pd.DataFrame(rows).sort_values('RMSE')
    tbl.to_csv(OUT / f'summary_{region}_{tag}.csv', index=False)
    return tbl


async def main():
    md = ['# Module 1: Unified EU27 & Japan ConsumerSim Benchmark (Table 1 Extension)\n']
    for region in ('EU27', 'Japan'):
        env_obj = RegionalEnv(region)
        target_ym = env_obj.target_ym()
        personas = build_local_personas(region)
        done = await run_region_sim(region, env_obj, personas)
        sim_ser, cal_info = score_region_sim(region, done, target_ym, personas)

        b_path = OUT / f'baselines_{region}_147m.csv'
        if b_path.exists():
            base_df = pd.read_csv(b_path, index_col=0)
            base_df.index = base_df.index.astype(int)
        else:
            base_df = run_regional_baselines(region, env_obj, target_ym)
            base_df.index.name = 'month'
            base_df.to_csv(b_path)

        all_s = base_df.copy()
        all_s.insert(0, 'ConsumerSim', [sim_ser[m] for m in EVAL_MONTHS])
        all_s.insert(0, 'Official', [target_ym[m] for m in EVAL_MONTHS])
        all_s.to_csv(OUT / f'all_series_{region}_147m.csv')

        md.append(f'## {region} (Dev calibration 2014-07..2022-12: `{cal_info}`)\n')
        for tag, ms, title in [
            ('147m_201407_202609', EVAL_MONTHS, 'Full Period (Aligned with U.S.): July 2014 to September 2026 (147 months)'),
            ('45m_202301_202609', OOS_MONTHS, 'Recent / Out-of-Sample (Aligned with U.S.): January 2023 to September 2026 (45 months)'),
            ('9m_202601_202609', POST_MONTHS, 'Post-Cutoff: January 2026 to September 2026 (9 months)'),
        ]:
            tbl = eval_table(region, sim_ser, base_df, target_ym, ms, tag)
            md.append(f'### {region} — {title}\n')
            md.append(tbl.round(3).to_markdown(index=False))
            md.append('')
        print(f'\n=== {region} FULL 147m (2014-07..2026-09) ===')
        print(pd.read_csv(OUT / f'summary_{region}_147m_201407_202609.csv').round(3).to_string(index=False))
        print(f'\n=== {region} RECENT 45m (2023-01..2026-09) ===')
        print(pd.read_csv(OUT / f'summary_{region}_45m_202301_202609.csv').round(3).to_string(index=False))

    (OUT / 'REPORT_147m.md').write_text('\n'.join(md))


if __name__ == '__main__':
    asyncio.run(main())
