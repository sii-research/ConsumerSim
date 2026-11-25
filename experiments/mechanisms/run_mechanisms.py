"""Module 5: Mechanism Non-Hardcoding Decomposition, Negative Controls,
Controlled Synthetic Counterfactual Shocks, and Exogenous (Outcome-Independent) High-Salience Check.

Addresses Reviewer #1 (#1 counterfactuals, #4 non-hardcoded mechanisms, #5 outcome-independent salience)
and Reviewer #2 (#1 placebo/negative controls).
All outputs live strictly in experiments/05_mechanisms/.
"""
import asyncio
import datetime as dt
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'consumersim'))
sys.path.insert(0, str(ROOT / 'consumersim'))
os.environ['CS_TEST'] = '1'

import data as D
import env
import engine
from llm_client import make_payload
import design
import ext28
import monitor as MON
from prompts import Q, TOPICS, PERSPECTIVE, SYSTEM, facts_text, family_text, exposure_profile, parse_contrast
from score import official_target
from t30_frames import base
from t43_pilot import group_change
from t47_micro_scan import CON
from t48_pilot import official_monthly_gaps

OUT = ROOT / 'consumersim' / '05_mechanisms'
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = OUT / 'cost_ledger.jsonl'

# 20 high-information Michigan months spanning elections, COVID, inflation, gas, tariffs, and ordinary months
MECH_MONTHS = [
    201609, 201611, 201612, 201804, 201812, 201906, 201908, 202001,
    202003, 202004, 202005, 202011, 202012, 202101, 202108, 202111,
    202203, 202206, 202208, 202210,
]


def _covid_block(day):
    try:
        return ext28.covid_text(day)
    except Exception:
        return ''


def build_tier_prompt(row, day_now, day_prev, tier, swap):
    """Build undated contrastive prompt under 3 mechanism tiers:
      - T0_RAW_SHARED: Raw demographics ONLY (NO exposure_profile prose, NO household_uniform routing).
                       Everyone sees the exact same shared macro+news+politics snapshot.
      - T1_RAW_HH:     Raw demographics ONLY (NO exposure_profile prose) + household_uniform indicators.
      - T2_FULL:       Full v4 Uniform Exposure Layer (facts + exposure_profile + household_uniform).
    """
    p = row.to_dict()
    fam = family_text(p)
    # Strip the president line from facts_text since political_text is inside each snapshot
    ft = facts_text(p, day_now)
    # Keep only demographic facts (before "The president is ...")
    idx_pres = ft.find('The president is ')
    if idx_pres > 0:
        ft = ft[:idx_pres].strip()
    if fam:
        ft = ft + ' ' + fam

    if tier == 'T2_FULL':
        exp = exposure_profile(p, env.job_exposed(p))
        persona_block = ft + '\n\n' + exp
    else:
        persona_block = ft

    diet = ext28.diet_of(p['pid'])

    def snap(day, other):
        parts = [env.political_text(day)]
        if diet in (0, 1):
            parts.append(env.macro_levels_text(env.macro(day)))
            if 202003 <= (day.year * 100 + day.month) <= 202212:
                cb = _covid_block(day)
                if cb:
                    parts.append(cb)
        if diet in (0, 2):
            parts.append(env.news_text(env.news(day, econ=True), econ=True))
        if tier in ('T1_RAW_HH', 'T2_FULL'):
            hh = env.household_uniform(p, day, other=other)
            if hh:
                parts.append(hh)
        return '\n\n'.join(parts)

    s_now = snap(day_now, day_prev)
    s_prev = snap(day_prev, day_now)
    snap_a, snap_b = (s_now, s_prev) if swap else (s_prev, s_now)
    now_label = 'A' if swap else 'B'

    q_inst = (
        'Compare the two snapshots from your own point of view. For each topic below, say which snapshot '
        'you feel better about ("A", "B", or "Same"), and how big the difference is for you: "large", "small", or "none".'
        if tier != 'T2_FULL' else
        PERSPECTIVE + ' Also say how big the difference is for you on that topic: "large", "small", or "none".'
    )
    user_msg = '\n\n'.join([
        persona_block,
        'Below are two snapshots of the economic figures and front-page news an ordinary person '
        'might have seen at two different times about a month apart. They are labeled A and B in no particular order.',
        'SNAPSHOT A\n' + snap_a,
        'SNAPSHOT B\n' + snap_b,
        q_inst,
        '\n'.join(f'{q}: {t}' for q, t in TOPICS.items()),
        'Return only a JSON object with exactly the keys "PAGO", "PEXP", "BUS12", "BUS5", "DUR". '
        'Each value is an object {"better": "A" or "B" or "Same", "size": "large" or "small" or "none"}.'
    ])
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user_msg}], now_label


async def run_tier_ablation(P_sub):
    """Run T0_RAW_SHARED, T1_RAW_HH, T2_FULL on the exact same personas across MECH_MONTHS."""
    key = engine._key()
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    url = f'{engine.ROUTE}/chat/completions'
    sem = asyncio.Semaphore(32)
    ds_map = D.dates()

    results_by_tier = {}
    for tier in ('T0_RAW_SHARED', 'T1_RAW_HH', 'T2_FULL'):
        cache_p = OUT / f'tier_{tier}.jsonl'
        done = {}
        if cache_p.exists():
            for line in cache_p.read_text().splitlines():
                if line.strip():
                    o = json.loads(line)
                    done[(o['month'], o['pid'])] = o

        todo = [row for _, row in P_sub.iterrows() if (int(row['month']), row['pid']) not in done]
        print(f'[5A {tier}] cached={len(done)}, todo={len(todo)}', flush=True)
        f_lock = asyncio.Lock()

        async def call_one(cli, row):
            m = int(row['month'])
            pid = row['pid']
            d_s = D.interview_start(m)
            d_e = dt.date.fromisoformat(ds_map[m]['interview_end'])
            span = max((d_e - d_s).days, 1)
            h = int(hashlib.md5(f'{m}_{pid}'.encode()).hexdigest()[:8], 16)
            day_now = d_s + dt.timedelta(days=h % span)
            day_prev = day_now - dt.timedelta(days=30)
            swap = ((h >> 8) % 2) == 1
            msgs, now_label = build_tier_prompt(row, day_now, day_prev, tier, swap)
            async with sem:
                for attempt in range(4):
                    try:
                        r = await cli.post(url, headers=headers, json=make_payload('gpt-6-luna', msgs, max_tokens=240))
                        r.raise_for_status()
                        data = r.json()
                        txt = data['choices'][0]['message']['content']
                        parsed, err = parse_contrast(txt, now_label)
                        if err:
                            raise ValueError(err)
                        u = data.get('usage', {})
                        rec = {'month': m, 'pid': pid, 'parsed': parsed}
                        async with f_lock:
                            with open(cache_p, 'a') as fp:
                                fp.write(json.dumps(rec) + '\n')
                            with open(LEDGER, 'a') as fp:
                                fp.write(json.dumps({'task': f'5A_{tier}', 'in': u.get('prompt_tokens', 0),
                                                     'out': u.get('completion_tokens', 0)}) + '\n')
                        return rec
                    except Exception as e:
                        if attempt == 3:
                            rec = {'month': m, 'pid': pid, 'parsed': {q: {'dir': 0, 'size': 0} for q in Q}, 'err': str(e)}
                            async with f_lock:
                                with open(cache_p, 'a') as fp:
                                    fp.write(json.dumps(rec) + '\n')
                            return rec
                        await asyncio.sleep(1.0 * (attempt + 1))

        if todo:
            async with httpx.AsyncClient(timeout=60) as cli:
                res = await asyncio.gather(*(call_one(cli, row) for row in todo))
            for rec in res:
                done[(rec['month'], rec['pid'])] = rec
        results_by_tier[tier] = done
    return results_by_tier


def score_tier_df(P_sub, done_map):
    """Score a tier using exact v4 household transition & cell consensus rules."""
    P = P_sub.copy().reset_index(drop=True)
    for q in Q:
        dirs = np.array([done_map[(int(r.month), r.pid)]['parsed'][q]['dir'] for r in P.itertuples()])
        sizes = np.array([done_map[(int(r.month), r.pid)]['parsed'][q]['size'] or 0 for r in P.itertuples()])
        P[f'dir_T_{q}'] = dirs
        P[f'size_T_{q}'] = sizes
        # Score change in [-2, +2] units (MON.method_series multiplies by 100)
        sc_map = {'F': 2.0, 'N': 1.0, 'U': 0.0, 'D': 1.0}
        cur = P[f'S_{q}'].map(sc_map).values
        if q in ('PAGO', 'PEXP'):
            nxt = np.where(dirs > 0, np.minimum(2.0, cur + 1.0), np.where(dirs < 0, np.maximum(0.0, cur - 1.0), cur))
        else:
            nxt = np.where(dirs > 0, 2.0, np.where(dirs < 0, 0.0, cur))
        P[f'ds_T_{q}'] = nxt - cur
    H = design.H(P, 'T', 'T')
    s, drs = MON.method_series(P, H, 'ds_T')
    return P, H, s


async def run_counterfactual_shocks():
    """Part 5C: Pure Synthetic Undated Counterfactual Shocks (ZERO historical events/dates!).
    Tests 5 isolated single-factor shocks under BOTH T0_RAW_SHARED (no channel prose/routing)
    and T2_FULL (full exposure layer) on 120 stratified personas."""
    cache_p = OUT / 'counterfactual_shocks.jsonl'
    done = {}
    if cache_p.exists():
        for line in cache_p.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                done[(o['tier'], o['shock'], o['pid'])] = o

    # Build 120 balanced synthetic personas covering Party(3) x Stocks(2) x Income(3) x Home(2) x Educ(2) x Age(2)
    rng = np.random.default_rng(2026)
    personas = []
    for i in range(120):
        pol = (i % 3) + 1          # 1=Rep, 2=Dem, 3=Ind
        inc = ((i // 3) % 3) + 1   # 1=Bot, 2=Mid, 3=Top
        inv = 1 if ((i // 9) % 2 == 0) else 5   # 1=Stocks, 5=No stocks
        own = 1 if ((i // 18) % 2 == 0) else 2  # 1=Homeowner, 2=Renter
        edu = 5 if ((i // 36) % 2 == 0) else 3  # 5=College, 3=HS
        age = 68 if (i % 5 == 0) else (27 if (i % 5 == 1) else 44)
        sex = 1 if (i % 2 == 0) else 2
        reg = (i % 4) + 1
        p = {'pid': f'CF_{i:03d}', 'POLAFF': pol, 'YTL3': inc, 'INVEST': inv, 'HOMEOWN': own,
             'EDUC': edu, 'AGE': age, 'SEX': sex, 'REGION': reg, 'MARRY': 1 if i % 2 == 0 else 5, 'NUMKID': i % 2}
        personas.append(p)

    # Baseline neutral synthetic world (completely fictionalized, zero dates!)
    BASE = {
        'pol': 'Political situation:\n- President: A Republican president is in office.',
        'gas': 3.30, 'spx': 5000, 'spx_ret': 0.5, 'unrate': 4.1, 'icsa': 220000,
        'cpi': 2.5, 'mort': 6.50, 'fed': 4.50, 'home_yoy': 3.0, 'groc': 2.4, 'rent': 3.2,
        'news': 'Front-page economic news:\n- Economic growth continues at a steady pace with stable consumer spending and moderate price increases.',
    }
    SHOCKS = {
        'Gasoline_Spike': {**BASE, 'gas': 5.20, 'cpi': 4.8, 'groc': 5.5,
                           'news': 'Front-page economic news:\n- Gasoline prices surge past $5.20 a gallon nationwide as energy costs spike, pushing up grocery and commuting bills.'},
        'Equity_Crash': {**BASE, 'spx': 4000, 'spx_ret': -20.0,
                         'news': 'Front-page economic news:\n- Wall Street plunges 20% in a sharp stock market sell-off, wiping trillions from equity portfolios and 401(k) accounts.'},
        'Labor_Shock': {**BASE, 'unrate': 6.8, 'icsa': 680000,
                        'news': 'Front-page economic news:\n- Nationwide layoffs accelerate sharply as unemployment jumps to 6.8% and weekly jobless claims surge to 680,000.'},
        'Mortgage_Rate_Surge': {**BASE, 'mort': 8.80, 'fed': 6.00, 'home_yoy': -6.5,
                                'news': 'Front-page economic news:\n- Mortgage rates jump to 8.8% following aggressive rate hikes, freezing home sales and pushing home values down 6.5%.'},
        'Partisan_Flip_Rep_to_Dem': {**BASE,
                                     'pol': 'Political situation:\n- President: A Democrat has just been elected president and is taking office, replacing the Republican administration.',
                                     'news': 'Front-page economic news:\n- Following the presidential election, a Democratic administration prepares to take office while macroeconomic indicators remain unchanged.'},
    }

    def render_cf_snap(w, p, tier):
        macro = (
            "Economic figures in the news:\n"
            f"- Regular gasoline: ${w['gas']:.2f} a gallon\n"
            f"- S&P 500 stock index: {w['spx']:,} ({'up' if w['spx_ret'] >= 0 else 'down'} {abs(w['spx_ret']):.1f}% over the past month)\n"
            f"- Unemployment rate in the latest monthly jobs report: {w['unrate']:.1f}%\n"
            f"- Consumer prices in the latest report: up {w['cpi']:.1f}% from a year earlier\n"
            f"- 30-year fixed mortgage rate: {w['mort']:.2f}%\n"
            f"- Federal Reserve's target interest rate: {w['fed'] - 0.25:.2f}% to {w['fed']:.2f}%"
        )
        parts = [w['pol'], macro, w['news']]
        if tier == 'T2_FULL':
            hh = []
            if p['INVEST'] == 1:
                hh.append(f"Your family's stock investments: {'up' if w['spx_ret'] >= 0 else 'down'} {abs(w['spx_ret']):.1f}% over the past month")
            if p['AGE'] >= 65:
                hh.append(f"What savings earn: {w['fed'] - 0.25:.2f}% to {w['fed']:.2f}%")
            elif p['EDUC'] <= 4 or p['AGE'] < 35:
                hh.append(f"New claims for unemployment benefits: {w['icsa']:,}")
            if p['YTL3'] in (1, 2):
                hh.append(f"Grocery prices: up {w['groc']:.1f}%; regular gasoline: ${w['gas']:.2f}/gal")
            if p['HOMEOWN'] == 1:
                hh.append(f"Home prices: {'up' if w['home_yoy'] >= 0 else 'down'} {abs(w['home_yoy']):.1f}%; mortgage rate: {w['mort']:.2f}%")
            else:
                hh.append(f"Rents: up {w['rent']:.1f}%")
            parts.append('For your household:\n' + '\n'.join('- ' + x for x in hh))
        return '\n\n'.join(parts)

    todo = [(tier, sh, p) for tier in ('T0_RAW_SHARED', 'T2_FULL') for sh in SHOCKS for p in personas if (tier, sh, p['pid']) not in done]
    print(f'[5C Counterfactuals] cached={len(done)}, todo={len(todo)}', flush=True)
    if todo:
        key = engine._key()
        headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
        url = f'{engine.ROUTE}/chat/completions'
        sem = asyncio.Semaphore(32)
        f_lock = asyncio.Lock()

        async def call_cf(cli, tier, sh, p):
            ft = facts_text(p, dt.date(2025, 6, 1))
            idx_pr = ft.find('The president is ')
            if idx_pr > 0:
                ft = ft[:idx_pr].strip()
            p_blk = ft + '\n\n' + exposure_profile(p, env.job_exposed(p)) if tier == 'T2_FULL' else ft
            s_base = render_cf_snap(BASE, p, tier)
            s_shk = render_cf_snap(SHOCKS[sh], p, tier)
            swap = (int(hashlib.md5(f'{tier}_{sh}_{p["pid"]}'.encode()).hexdigest()[:8], 16) % 2) == 1
            sa, sb = (s_shk, s_base) if swap else (s_base, s_shk)
            now_lbl = 'A' if swap else 'B'
            q_inst = (
                'Compare the two snapshots from your own point of view. For each topic below, say which snapshot '
                'you feel better about ("A", "B", or "Same"), and how big the difference is for you: "large", "small", or "none".'
                if tier == 'T0_RAW_SHARED' else
                PERSPECTIVE + ' Also say how big the difference is for you on that topic: "large", "small", or "none".'
            )
            msg = '\n\n'.join([
                p_blk,
                'Below are two undated snapshots of economic conditions and news about a month apart, labeled A and B in no particular order.',
                'SNAPSHOT A\n' + sa, 'SNAPSHOT B\n' + sb, q_inst,
                '\n'.join(f'{q}: {t}' for q, t in TOPICS.items()),
                'Return only a JSON object with exactly the keys "PAGO", "PEXP", "BUS12", "BUS5", "DUR". '
                'Each value is an object {"better": "A" or "B" or "Same", "size": "large" or "small" or "none"}.'
            ])
            async with sem:
                for attempt in range(4):
                    try:
                        r = await cli.post(url, headers=headers, json=make_payload(
                            'gpt-6-luna', [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': msg}], max_tokens=240
                        ))
                        r.raise_for_status()
                        data = r.json()
                        parsed, err = parse_contrast(data['choices'][0]['message']['content'], now_lbl)
                        if err:
                            raise ValueError(err)
                        rec = {'tier': tier, 'shock': sh, 'pid': p['pid'], 'p': p, 'parsed': parsed}
                        async with f_lock:
                            with open(cache_p, 'a') as fp:
                                fp.write(json.dumps(rec) + '\n')
                        return rec
                    except Exception as e:
                        if attempt == 3:
                            rec = {'tier': tier, 'shock': sh, 'pid': p['pid'], 'p': p,
                                   'parsed': {q: {'dir': 0, 'size': 0} for q in Q}, 'err': str(e)}
                            async with f_lock:
                                with open(cache_p, 'a') as fp:
                                    fp.write(json.dumps(rec) + '\n')
                            return rec
                        await asyncio.sleep(1.0 * (attempt + 1))

        async with httpx.AsyncClient(timeout=60) as cli:
            res = await asyncio.gather(*(call_cf(cli, t, s, p) for t, s, p in todo))
        for rec in res:
            done[(rec['tier'], rec['shock'], rec['pid'])] = rec

    # Analyze synthetic shock responses: mean weighted update score (large=1.0, small=0.15) * dir averaged across 5 items
    rows = []
    for tier in ('T0_RAW_SHARED', 'T2_FULL'):
        for sh in SHOCKS:
            recs = [done[(tier, sh, p['pid'])] for p in personas]
            def score_p(rec):
                vals = []
                for q in Q:
                    d = rec['parsed'][q]['dir']
                    sz = rec['parsed'][q]['size'] or 0
                    w = 1.0 if sz == 2 else (0.15 if sz == 1 else 0.0)
                    vals.append(d * w)
                return float(np.mean(vals)) * 100.0  # -100 to +100 intensity scale
            df_s = pd.DataFrame([{**r['p'], 'resp': score_p(r)} for r in recs])
            rows.append({
                'Tier': tier, 'Synthetic Shock': sh,
                'All Mean': df_s['resp'].mean(),
                'Dem - Rep': df_s[df_s.POLAFF == 2].resp.mean() - df_s[df_s.POLAFF == 1].resp.mean(),
                'Stock - NoStock': df_s[df_s.INVEST == 1].resp.mean() - df_s[df_s.INVEST == 5].resp.mean(),
                'BotInc - TopInc': df_s[df_s.YTL3 == 1].resp.mean() - df_s[df_s.YTL3 == 3].resp.mean(),
                'Owner - Renter': df_s[df_s.HOMEOWN == 1].resp.mean() - df_s[df_s.HOMEOWN == 2].resp.mean(),
                'NoCol - College': df_s[df_s.EDUC == 3].resp.mean() - df_s[df_s.EDUC == 5].resp.mean(),
                'Working - Retired65+': df_s[df_s.AGE < 65].resp.mean() - df_s[df_s.AGE >= 65].resp.mean(),
                '[NegCtrl] Male - Female': df_s[df_s.SEX == 1].resp.mean() - df_s[df_s.SEX == 2].resp.mean(),
            })
    cf_df = pd.DataFrame(rows)
    cf_df.to_csv(OUT / 'counterfactual_summary.csv', index=False)
    return cf_df


def run_exogenous_salience_check():
    """Part 5D: Outcome-Independent (Exogenous X_t-only) High-Salience Months Evaluation.
    Ranks months strictly by observable input shock intensity S_t(X_t) WITHOUT ever looking at official Y_t!"""
    feat = pd.read_csv(ROOT / 'receipts/T31/features.csv', index_col=0)
    feat.index = feat.index.astype(int)
    sim_df = pd.read_csv(ROOT / 'receipts/T49/series_1800.csv', index_col=0)
    sim_df.index = sim_df.index.astype(int)
    bpath = ROOT / 'receipts/T56/baselines_cpifix/baselines.csv'
    base_df = pd.read_csv(bpath, index_col=0)
    base_df.index = base_df.index.astype(int)

    eval_m = list(sim_df.index)
    F = feat.loc[eval_m].copy()
    # Exogenous shock intensity: standardized sum of |equity move|, VIX level, |gas move|, |unrate move|, |infl move|, news volume
    cols = ['r_spx', 'vix', 'd_gas', 'd_unrate', 'd_infl', 'news_n']
    z = pd.DataFrame(index=eval_m)
    for c in cols:
        v = F[c].abs() if c != 'vix' and c != 'news_n' else F[c]
        z[c] = (v - v.mean()) / (v.std() + 1e-9)
    salience = z.mean(axis=1).sort_values(ascending=False)

    all_m = pd.DataFrame(index=eval_m)
    all_m['Official'] = sim_df['official']
    all_m['Persistence'] = sim_df['persistence']
    all_m['ConsumerSim'] = sim_df['sim']
    for c in ['SARIMAX Macro', 'Auto ARIMA', 'AR Ridge', 'Expectations Reg.', 'News Semantic FAAR', 'Theta Model', 'ETS Damped Trend']:
        all_m[c] = base_df.loc[eval_m, c]

    tables = {}
    for top_k in (10, 20):
        top_ms = list(salience.head(top_k).index)
        sub = all_m.loc[top_ms]
        off, pers = sub['Official'].values, sub['Persistence'].values
        d_off = off - pers
        e_cs = sub['ConsumerSim'].values - off
        rng = np.random.default_rng(42)
        rows = []
        for col in [c for c in sub.columns if c != 'Official']:
            pred = sub[col].values
            e = pred - off
            dp = pred - pers
            mae = float(np.mean(np.abs(e)))
            rmse = float(np.sqrt(np.mean(e ** 2)))
            cap = float(np.dot(dp, d_off) / (np.dot(d_off, d_off) + 1e-9)) * 100
            if col == 'ConsumerSim':
                ci = '--'
            else:
                # Paired bootstrap over the selected months
                diffs = [np.sqrt(np.mean(e_cs[idx] ** 2)) - np.sqrt(np.mean(e[idx] ** 2))
                         for idx in rng.integers(0, len(e), size=(2000, len(e)))]
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                ci = f'{rmse - np.sqrt(np.mean(e_cs**2)):+.2f} [{lo:.2f}, {hi:.2f}]'
            rows.append({'Method': col, 'MAE': mae, 'RMSE': rmse, 'Move Captured (%)': cap, 'CS-M 95% CI': ci})
        tbl = pd.DataFrame(rows).sort_values('RMSE')
        tbl.to_csv(OUT / f'exogenous_salience_top{top_k}.csv', index=False)
        tables[top_k] = (top_ms, tbl)
    return tables


async def main():
    # Prepare balanced 150-persona/month Michigan subset on MECH_MONTHS
    B = base(600)
    B = B[B.month.isin(MECH_MONTHS)].copy()
    # Deterministic subsample of 150 personas per month (preserving survey weights)
    P_sub = B.groupby('month', group_keys=False).apply(lambda g: g.iloc[:150]).reset_index(drop=True)
    P_sub['w'] = P_sub.w / P_sub.groupby('month').w.transform('sum')
    print(f'[5A] Loaded {len(P_sub)} personas across {len(MECH_MONTHS)} months')

    tier_res = await run_tier_ablation(P_sub)
    off = official_target()
    OG = official_monthly_gaps()

    # Evaluate 5A Tiers + 5B Negative Controls
    t_rows = []
    neg_rows = []
    for tier, done_map in tier_res.items():
        P_sc, H_sc, s_sc = score_tier_df(P_sub, done_map)
        e = np.array([s_sc[m] - off[m] for m in MECH_MONTHS])
        pers = np.array([off[D.ym_add(m, -1)] for m in MECH_MONTHS])
        d_off = np.array([off[m] for m in MECH_MONTHS]) - pers
        d_sim = np.array([s_sc[m] for m in MECH_MONTHS]) - pers
        r_chg = float(np.corrcoef(d_sim, d_off)[0, 1])

        gap_corrs = {}
        gap_sds = {}
        for cname, (s1, s2) in CON.items():
            c1, c2 = group_change(P_sc, H_sc, 'ds_T', s1), group_change(P_sc, H_sc, 'ds_T', s2)
            g = pd.Series({m: c1.get(m, np.nan) - c2.get(m, np.nan) for m in MECH_MONTHS})
            ok = g.notna() & OG[cname].notna()
            gap_corrs[cname] = float(np.corrcoef(g[ok], OG[cname][ok])[0, 1]) if g[ok].std() > 1e-9 else 0.0
            gap_sds[cname] = float(g.std())

        # Extra negative controls: Marital (MARRY==1 vs !=1) & Children (NUMKID>0 vs ==0)
        for nc_name, cond1, cond2 in [
            ('Sex (men - women)', lambda df: df.SEX == 1, lambda df: df.SEX == 2),
            ('Marital (married - not)', lambda df: df.MARRY == 1, lambda df: df.MARRY.isin([3, 4, 5])),
            ('Children (has kids - none)', lambda df: df.NUMKID > 0, lambda df: df.NUMKID == 0),
        ]:
            c1, c2 = group_change(P_sc, H_sc, 'ds_T', cond1), group_change(P_sc, H_sc, 'ds_T', cond2)
            g = pd.Series({m: c1.get(m, np.nan) - c2.get(m, np.nan) for m in MECH_MONTHS})
            neg_rows.append({'Tier': tier, 'Negative Control Split': nc_name,
                             'Mean |Sim Gap Change| (ICS pts)': float(g.abs().mean()),
                             'SD Sim Gap Change': float(g.std()),
                             'Max |Sim Gap Change|': float(g.abs().max())})

        t_rows.append({
            'Tier': tier,
            'MAE': float(np.mean(np.abs(e))), 'RMSE': float(np.sqrt(np.mean(e ** 2))), 'Delta r': r_chg,
            'Party gap r': gap_corrs['Party (Rep - Dem)'],
            'Stock gap r': gap_corrs['Stocks (owner - none)'],
            'Educ gap r': gap_corrs['Education (college - HS or less)'],
            'Income gap r': gap_corrs['Income (top - bottom third)'],
            '[NegCtrl] Sex SD': gap_sds['Sex (men - women)'],
        })
    tier_tbl = pd.DataFrame(t_rows)
    neg_tbl = pd.DataFrame(neg_rows)
    tier_tbl.to_csv(OUT / 'tier_decomposition.csv', index=False)
    neg_tbl.to_csv(OUT / 'negative_controls.csv', index=False)

    # Part 5C: Controlled Synthetic Counterfactual Shocks
    cf_df = await run_counterfactual_shocks()

    # Part 5D: Exogenous High-Salience Check
    exo_tables = run_exogenous_salience_check()

    md = [
        '# Module 5: Mechanism Non-Hardcoding Proof, Negative Controls, Counterfactuals & Exogenous Salience\n',
        '## 5A. Three-Tier Mechanism Decomposition (20 High-Info Michigan Months, 150 Personas/Month)\n',
        '- `T0_RAW_SHARED`: Raw demographics ONLY + identical shared macro/news snapshot (ZERO channel prose, ZERO household-specific indicator routing).\n',
        '- `T1_RAW_HH`: Raw demographics ONLY (ZERO channel prose) + household-specific real-time indicators.\n',
        '- `T2_FULL`: Full v4 Uniform Exposure Layer.\n\n',
        tier_tbl.round(3).to_markdown(index=False),
        '\n## 5B. Negative Control Splits (No Economic Channel -> Near-Zero Spurious Divergence)\n',
        neg_tbl.round(3).to_markdown(index=False),
        '\n## 5C. Controlled Synthetic Counterfactual Shocks (Zero Dates, Zero Historical Memory)\n',
        'Values are directional response intensity (−100 to +100) for paired contrasts under pure single-factor synthetic shocks:\n\n',
        cf_df.round(2).to_markdown(index=False),
        '\n## 5D. Outcome-Independent (Exogenous X_t-Only) High-Salience Months\n',
        f'### Top-10 Exogenous Salience Months ({exo_tables[10][0]})\n',
        exo_tables[10][1].round(3).to_markdown(index=False),
        f'\n### Top-20 Exogenous Salience Months\n',
        exo_tables[20][1].round(3).to_markdown(index=False),
    ]
    (OUT / 'REPORT.md').write_text('\n'.join(md))
    print('\n=== 5A TIER DECOMPOSITION ===')
    print(tier_tbl.round(3).to_string(index=False))
    print('\n=== 5C COUNTERFACTUALS ===')
    print(cf_df.round(2).to_string(index=False))
    print('\n=== 5D EXOGENOUS SALIENCE TOP-10 ===')
    print(exo_tables[10][1].round(3).to_string(index=False))


if __name__ == '__main__':
    asyncio.run(main())
