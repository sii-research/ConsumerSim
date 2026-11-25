"""Rerun 5C Synthetic Counterfactual Shocks with strictly orthogonal balanced Sex assignment
(60M/60F balanced within every Stock x Party x Income cell) so negative control Male-Female has zero grid confounding."""
import asyncio
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'consumersim'))
sys.path.insert(0, str(ROOT / 'consumersim'))
os.environ['CS_TEST'] = '1'

import env
import engine
from llm_client import make_payload
from prompts import Q, TOPICS, PERSPECTIVE, SYSTEM, facts_text, exposure_profile, parse_contrast

OUT = ROOT / 'consumersim' / '05_mechanisms'


async def main():
    cache_p = OUT / 'counterfactual_shocks_ortho.jsonl'
    done = {}
    if cache_p.exists():
        for line in cache_p.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                done[(o['tier'], o['shock'], o['pid'])] = o

    # 144 exact orthogonal factorial personas:
    # Party(3) x Income(3) x Stocks(2) x Home(2) x Educ(2) x Sex(2) = 144 (100% orthogonal!)
    personas = []
    idx = 0
    for pol in (1, 2, 3):
        for inc in (1, 2, 3):
            for inv in (1, 5):
                for own in (1, 2):
                    for edu in (3, 5):
                        for sex in (1, 2):
                            age = 68 if (idx % 5 == 0) else (27 if (idx % 5 == 1) else 44)
                            reg = (idx % 4) + 1
                            personas.append({
                                'pid': f'CFO_{idx:03d}', 'POLAFF': pol, 'YTL3': inc, 'INVEST': inv,
                                'HOMEOWN': own, 'EDUC': edu, 'SEX': sex, 'AGE': age, 'REGION': reg,
                                'MARRY': 1 if ((idx // 2) % 2 == 0) else 5, 'NUMKID': (idx // 4) % 2,
                            })
                            idx += 1

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

    def render_snap(w, p, tier):
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
    print(f'[5C Ortho] N=144 orthogonal factorial, cached={len(done)}, todo={len(todo)}', flush=True)
    if todo:
        key = engine._key()
        headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
        url = f'{engine.ROUTE}/chat/completions'
        sem = asyncio.Semaphore(64)
        f_lock = asyncio.Lock()

        async def call_cf(cli, tier, sh, p):
            ft = facts_text(p, dt.date(2025, 6, 1))
            idx_pr = ft.find('The president is ')
            if idx_pr > 0:
                ft = ft[:idx_pr].strip()
            p_blk = ft + '\n\n' + exposure_profile(p, env.job_exposed(p)) if tier == 'T2_FULL' else ft
            s_base = render_snap(BASE, p, tier)
            s_shk = render_snap(SHOCKS[sh], p, tier)
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
                            return {'tier': tier, 'shock': sh, 'pid': p['pid'], 'p': p,
                                    'parsed': {q: {'dir': 0, 'size': 0} for q in Q}, 'err': str(e)}
                        await asyncio.sleep(1.0 * (attempt + 1))

        limits = httpx.Limits(max_connections=64, max_keepalive_connections=64)
        async with httpx.AsyncClient(timeout=45, limits=limits) as cli:
            res = await asyncio.gather(*(call_cf(cli, t, s, p) for t, s, p in todo))
        for rec in res:
            done[(rec['tier'], rec['shock'], rec['pid'])] = rec

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
                return float(np.mean(vals)) * 100.0
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
    cf_df.to_csv(OUT / 'counterfactual_summary_ortho.csv', index=False)
    print(cf_df.round(2).to_string(index=False))


if __name__ == '__main__':
    asyncio.run(main())
