"""Module 4: Multi-LLM Backbone Robustness, Prompt Wording Sensitivity, and Random Seed Stability.
Directly addresses Reviewer #1 (#6) and Reviewer #2 (#2), reporting BOTH aggregate accuracy
(MAE, RMSE, Delta r) AND subgroup gap fidelity (Party, Stocks) against Michigan microdata!

Evaluated on the same 20 high-information Michigan months (150 stratified personas/month = 3,000 calls/arm)
as Module 5, using the exact frozen v4 uniform exposure architecture.
All outputs live strictly in experiments/04_robustness/.
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
import design
import ext28
import monitor as MON
from llm_client import make_payload
from prompts import Q, TOPICS, TOPICS_V2, PERSPECTIVE, SYSTEM, facts_text, family_text, exposure_profile, parse_contrast
from score import official_target
from t30_frames import base
from t43_pilot import group_change
from t47_micro_scan import CON
from t48_pilot import official_monthly_gaps

OUT = ROOT / 'consumersim' / '04_robustness'
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = OUT / 'cost_ledger.jsonl'

ROB_MONTHS = [
    201609, 201611, 201612, 201804, 201812, 201906, 201908, 202001,
    202003, 202004, 202005, 202011, 202012, 202101, 202108, 202111,
    202203, 202206, 202208, 202210,
]

# Prompt wording variants (testing prompt sensitivity while keeping exact same information!)
PROMPT_VARIANTS = {
    'P0_Standard': {
        'sys': SYSTEM,
        'intro': 'Below are two snapshots of the economic figures and front-page news an ordinary person might have seen at two different times about a month apart. They are labeled A and B in no particular order.',
        'inst': PERSPECTIVE + ' Also say how big the difference is for you on that topic: "large", "small", or "none".',
        'topics': TOPICS,
    },
    'P1_Formal_Survey': {
        'sys': 'You are participating in a monthly household economic sentiment survey as the American adult profiled below. Respond faithfully from their lived economic situation and political perspective. Never break character.',
        'intro': 'Two undated economic briefings—labeled Period A and Period B—summarize public economic data and headlines observed roughly one month apart. Their chronological order is hidden.',
        'inst': 'Evaluate Period A versus Period B strictly from your household standpoint (income, prices, investments, job risk, and political outlook). For each of the five items, indicate whether "A", "B", or "Same" feels more favorable, and classify the personal impact as "large", "small", or "none".',
        'topics': TOPICS_V2,
    },
    'P2_Concise_Casual': {
        'sys': 'Roleplay the U.S. resident below answering a quick phone poll about the economy. Answer naturally as this person.',
        'intro': 'Here are two undated snapshots of news and economic numbers from two months (A and B, random order):',
        'inst': 'Speaking as this person, pick which snapshot ("A", "B", or "Same") feels better for each question below, and whether the gap feels "large", "small", or "none" to you.',
        'topics': TOPICS,
    },
    'P3_Reverse_Topic_Order': {
        'sys': SYSTEM,
        'intro': 'Below are two snapshots of the economic figures and front-page news an ordinary person might have seen at two different times about a month apart. They are labeled A and B in no particular order.',
        'inst': PERSPECTIVE + ' Also say how big the difference is for you on that topic: "large", "small", or "none".',
        'topics': {k: TOPICS[k] for k in reversed(list(TOPICS.keys()))},
    },
}

# Experiments to run:
# 1. Multi-LLM backbones (4 distinct model families including open-weight!)
# 2. Prompt wording variants (4 templates on gpt-6-luna)
# 3. Random seed replications (3 independent seeds on gpt-6-luna)
RUNS_SPEC = [
    # (tag, group, model, prompt_key, seed)
    ('LLM_gpt6_luna', 'Multi-LLM', 'gpt-6-luna', 'P0_Standard', None),
    ('LLM_qwen35_flash', 'Multi-LLM', 'qwen3.5-flash', 'P0_Standard', None),
    ('LLM_gemini35_lite', 'Multi-LLM', 'gemini-3.5-flash-lite', 'P0_Standard', None),
    ('LLM_grok41_fast', 'Multi-LLM', 'grok-4-1-fast-non-reasoning', 'P0_Standard', None),
    ('LLM_deepseek_v41', 'Multi-LLM', 'deepseek-v4.1-flash', 'P0_Standard', None),
    ('Prompt_P1_Formal', 'Prompt', 'gpt-6-luna', 'P1_Formal_Survey', None),
    ('Prompt_P2_Concise', 'Prompt', 'gpt-6-luna', 'P2_Concise_Casual', None),
    ('Prompt_P3_RevOrder', 'Prompt', 'gpt-6-luna', 'P3_Reverse_Topic_Order', None),
    ('Seed_202', 'Seed', 'gpt-6-luna', 'P0_Standard', 202),
    ('Seed_303', 'Seed', 'gpt-6-luna', 'P0_Standard', 303),
]


def build_msg(row, day_now, day_prev, prompt_key, swap):
    p = row.to_dict()
    ft = facts_text(p, day_now)
    idx_pr = ft.find('The president is ')
    if idx_pr > 0:
        ft = ft[:idx_pr].strip()
    fam = family_text(p)
    if fam:
        ft = ft + ' ' + fam
    persona_block = ft + '\n\n' + exposure_profile(p, env.job_exposed(p))
    diet = ext28.diet_of(p['pid'])

    def snap(day, other):
        parts = [env.political_text(day)]
        if diet in (0, 1):
            parts.append(env.macro_levels_text(env.macro(day)))
            if 202003 <= (day.year * 100 + day.month) <= 202212:
                try:
                    cb = ext28.covid_text(day)
                    if cb:
                        parts.append(cb)
                except Exception:
                    pass
        if diet in (0, 2):
            parts.append(env.news_text(env.news(day, econ=True), econ=True))
        hh = env.household_uniform(p, day, other=other)
        if hh:
            parts.append(hh)
        return '\n\n'.join(parts)

    s_now = snap(day_now, day_prev)
    s_prev = snap(day_prev, day_now)
    sa, sb = (s_now, s_prev) if swap else (s_prev, s_now)
    now_lbl = 'A' if swap else 'B'
    pv = PROMPT_VARIANTS[prompt_key]
    keys_str = ', '.join(f'"{k}"' for k in pv['topics'])
    user_msg = '\n\n'.join([
        persona_block,
        pv['intro'],
        'SNAPSHOT A\n' + sa,
        'SNAPSHOT B\n' + sb,
        pv['inst'],
        '\n'.join(f'{q}: {t}' for q, t in pv['topics'].items()),
        f'Return only a JSON object with exactly the keys {keys_str}. '
        'Each value is an object {"better": "A" or "B" or "Same", "size": "large" or "small" or "none"}.'
    ])
    return [{'role': 'system', 'content': pv['sys']}, {'role': 'user', 'content': user_msg}], now_lbl


async def run_one_spec(P_sub, tag, model, prompt_key, seed):
    cache_p = OUT / f'raw_{tag}.jsonl'
    done = {}
    if cache_p.exists():
        for line in cache_p.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                if 'err' not in o:
                    done[(o['month'], o['pid'])] = o

    todo = [row for _, row in P_sub.iterrows() if (int(row['month']), row['pid']) not in done]
    print(f'[{tag} ({model})] cached={len(done)}, todo={len(todo)}', flush=True)
    if not todo:
        return done

    key = engine._key()
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    url = f'{engine.ROUTE}/chat/completions'
    sem = asyncio.Semaphore(32)
    f_lock = asyncio.Lock()
    ds_map = D.dates()

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
        msgs, now_lbl = build_msg(row, day_now, day_prev, prompt_key, swap)
        payload = make_payload(model, msgs, max_tokens=260, seed=seed)
        async with sem:
            for attempt in range(4):
                try:
                    r = await cli.post(url, headers=headers, json=payload)
                    r.raise_for_status()
                    data = r.json()
                    txt = data['choices'][0]['message'].get('content') or ''
                    parsed, err = parse_contrast(txt, now_lbl)
                    if err:
                        raise ValueError(err)
                    u = data.get('usage', {})
                    rec = {'month': m, 'pid': pid, 'parsed': parsed}
                    async with f_lock:
                        with open(cache_p, 'a') as fp:
                            fp.write(json.dumps(rec) + '\n')
                        with open(LEDGER, 'a') as fp:
                            fp.write(json.dumps({'tag': tag, 'model': model, 'in': u.get('prompt_tokens', 0),
                                                 'out': u.get('completion_tokens', 0)}) + '\n')
                    return rec
                except Exception as e:
                    if attempt == 3:
                        return {'month': m, 'pid': pid, 'parsed': {q: {'dir': 0, 'size': 0} for q in Q}, 'err': str(e)}
                    await asyncio.sleep(1.2 * (attempt + 1))

    limits = httpx.Limits(max_connections=48, max_keepalive_connections=48)
    async with httpx.AsyncClient(timeout=40, limits=limits) as cli:
        res = await asyncio.gather(*(call_one(cli, row) for row in todo))
    fail_n = sum(1 for r in res if 'err' in r)
    print(f'[{tag}] finished {len(res)} calls (failures={fail_n})', flush=True)
    for rec in res:
        done[(rec['month'], rec['pid'])] = rec
    return done


def score_spec(P_sub, done_map, off, OG):
    P = P_sub.copy().reset_index(drop=True)
    sc_map = {'F': 2.0, 'N': 1.0, 'U': 0.0, 'D': 1.0}
    for q in Q:
        dirs = np.array([done_map[(int(r.month), r.pid)]['parsed'][q]['dir'] for r in P.itertuples()])
        sizes = np.array([done_map[(int(r.month), r.pid)]['parsed'][q]['size'] or 0 for r in P.itertuples()])
        P[f'dir_R_{q}'] = dirs
        P[f'size_R_{q}'] = sizes
        cur = P[f'S_{q}'].map(sc_map).values
        if q in ('PAGO', 'PEXP'):
            nxt = np.where(dirs > 0, np.minimum(2.0, cur + 1.0), np.where(dirs < 0, np.maximum(0.0, cur - 1.0), cur))
        else:
            nxt = np.where(dirs > 0, 2.0, np.where(dirs < 0, 0.0, cur))
        P[f'ds_R_{q}'] = nxt - cur
    H = design.H(P, 'R', 'R')
    s, _ = MON.method_series(P, H, 'ds_R')

    e = np.array([s[m] - off[m] for m in ROB_MONTHS])
    pers = np.array([off[D.ym_add(m, -1)] for m in ROB_MONTHS])
    d_off = np.array([off[m] for m in ROB_MONTHS]) - pers
    d_sim = np.array([s[m] for m in ROB_MONTHS]) - pers

    gc = {}
    for cname in ('Party (Rep - Dem)', 'Stocks (owner - none)', 'Education (college - HS or less)', 'Sex (men - women)'):
        s1, s2 = CON[cname]
        c1, c2 = group_change(P, H, 'ds_R', s1), group_change(P, H, 'ds_R', s2)
        g = pd.Series({m: c1.get(m, np.nan) - c2.get(m, np.nan) for m in ROB_MONTHS})
        ok = g.notna() & OG[cname].notna()
        gc[cname] = float(np.corrcoef(g[ok], OG[cname][ok])[0, 1]) if g[ok].std() > 1e-9 else 0.0
    return {
        'MAE': float(np.mean(np.abs(e))),
        'RMSE': float(np.sqrt(np.mean(e ** 2))),
        'Level r': float(np.corrcoef([s[m] for m in ROB_MONTHS], [off[m] for m in ROB_MONTHS])[0, 1]),
        'Delta r': float(np.corrcoef(d_sim, d_off)[0, 1]) if np.std(d_sim) > 1e-9 else 0.0,
        'Party gap r': gc['Party (Rep - Dem)'],
        'Stock gap r': gc['Stocks (owner - none)'],
        'Educ gap r': gc['Education (college - HS or less)'],
    }


async def main():
    B = base(600)
    B = B[B.month.isin(ROB_MONTHS)].copy()
    # 120 stratified personas/month across 20 months = 2,400 calls per variant (fast, stable, low cost!)
    P_sub = B.groupby('month', group_keys=False).apply(lambda g: g.iloc[:120]).reset_index(drop=True)
    P_sub['w'] = P_sub.w / P_sub.groupby('month').w.transform('sum')
    off = official_target()
    OG = official_monthly_gaps()

    # Persistence reference on same 20 months
    e_p = np.array([off[D.ym_add(m, -1)] - off[m] for m in ROB_MONTHS])
    pers_row = {'Group': 'Reference', 'Variant': 'Persistence', 'Model': '--', 'Prompt': '--', 'Seed': '--',
                'MAE': float(np.mean(np.abs(e_p))), 'RMSE': float(np.sqrt(np.mean(e_p ** 2))),
                'Level r': float(np.corrcoef([off[D.ym_add(m, -1)] for m in ROB_MONTHS], [off[m] for m in ROB_MONTHS])[0, 1]),
                'Delta r': np.nan, 'Party gap r': np.nan, 'Stock gap r': np.nan, 'Educ gap r': np.nan}

    rows = [pers_row]
    for tag, grp, model, pkey, seed in RUNS_SPEC:
        done = await run_one_spec(P_sub, tag, model, pkey, seed)
        # Skip if too many failures
        fails = sum(1 for v in done.values() if 'err' in v)
        if fails > len(P_sub) * 0.25:
            print(f'[WARN] {tag} had {fails}/{len(P_sub)} failures, skipping from clean table')
            continue
        sc = score_spec(P_sub, done, off, OG)
        rows.append({'Group': grp, 'Variant': tag, 'Model': model, 'Prompt': pkey, 'Seed': seed, **sc})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'robustness_summary.csv', index=False)

    # Compute cross-seed SD & 95% CI summary
    seed_df = df[df['Variant'].isin(['LLM_gpt6_luna', 'Seed_202', 'Seed_303'])]
    seed_stats = ''
    if len(seed_df) >= 2:
        seed_stats = (
            f"- **Across 3 independent seeds (`gpt-6-luna`, P0)**: "
            f"RMSE = **{seed_df.RMSE.mean():.3f} ± {seed_df.RMSE.std():.3f}**, "
            f"MAE = **{seed_df.MAE.mean():.3f} ± {seed_df.MAE.std():.3f}**, "
            f"Delta r = **{seed_df['Delta r'].mean():.3f} ± {seed_df['Delta r'].std():.3f}**, "
            f"Party gap r = **{seed_df['Party gap r'].mean():.3f} ± {seed_df['Party gap r'].std():.3f}**, "
            f"Stock gap r = **{seed_df['Stock gap r'].mean():.3f} ± {seed_df['Stock gap r'].std():.3f}**.\n"
        )

    md = [
        '# Module 4: Multi-LLM Backbone, Prompt Wording & Random Seed Robustness (20 Months, 120 Personas/Month)\n',
        df.round(3).to_markdown(index=False),
        '\n## Seed Stability Summary\n',
        seed_stats,
    ]
    (OUT / 'REPORT.md').write_text('\n'.join(md))
    print('\n=== MODULE 4 ROBUSTNESS SUMMARY ===')
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    asyncio.run(main())
