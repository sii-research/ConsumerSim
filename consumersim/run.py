"""T10 driver: build assimilated populations, render prompts per arm, call the model, aggregate.

Usage:
  python3 run.py build  --run S1 --months screen --k 1
  python3 run.py call   --run S1 --arms C0,E3 --cap 40 [--limit-per-month 20] [--months ...]
  python3 run.py agg    --run S1
"""
import os
import argparse
import asyncio
import datetime as dt
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import data as D
from assim import assimilate, aggregate
from env import macro, macro_text, macro_levels_text, prices_text, news, news_text, personal_text, political_text, household_uniform, job_exposed
from prompts import render, render_contrast, facts_text, history_text, president_line, template_hash, Q, TOPICS_V2, lens_text, stakes_text, exposure_profile, family_text
import ext28
import engine

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
SEED = 20261116
ARMS = {
    'C0': {},
    'D': {'date': True},
    'E1': {'macro': True},
    'E2': {'news': True},
    'E3': {'macro': True, 'news': True},
    'H': {'macro': True, 'news': True, 'facts': 'own'},
    'Hs': {'macro': True, 'news': True, 'facts': 'shuffle'},
    # wording probes for the no-information drift
    'C0p': {'style': 'panel'},
    'C0k': {'style': 'keep'},
    'C0f': {'facts': 'own'},
    'E3k': {'macro': True, 'news': True, 'style': 'keep'},
    'Hk': {'macro': True, 'news': True, 'facts': 'own', 'style': 'keep'},
    'Hks': {'macro': True, 'news': True, 'facts': 'shuffle', 'style': 'keep'},
    'E3m': {'macro': True, 'news': True, 'memory': True, 'style': 'keep'},
    'E3e': {'macro': True, 'econ_news': True, 'style': 'keep'},
    # reasoning probes (diagnosis on training months only)
    'E3e_c': {'macro': True, 'econ_news': True, 'style': 'keep', 'cot': True, 'max_tokens': 450},
    'E3e_r': {'macro': True, 'econ_news': True, 'style': 'keep', 'reasoning': 'low', 'max_tokens': 2000},
    'Dk_c': {'date': True, 'style': 'keep', 'cot': True, 'max_tokens': 450},
    'He_c': {'macro': True, 'econ_news': True, 'facts': 'own', 'style': 'keep', 'cot': True, 'max_tokens': 450},
    'E3e_im': {'macro': True, 'econ_news': True, 'info_memory': True, 'style': 'keep'},
    # T12 contrast elicitation: direction (and the persona's own 'would change') from comparing last month vs this month
    'X': {'contrast': True, 'facts': True},
    'Xp': {'contrast': True, 'facts': True, 'placebo': True},
    'X2': {'contrast': True, 'facts': True, 'topics': 'v2'},
    'X3': {'contrast': True, 'facts': True, 'topics': 'v2', 'snap': 'both'},
    'X4': {'contrast': True, 'facts': True, 'topics': 'v2', 'snap': 'prices'},
    'Xm': {'contrast': True, 'facts': True, 'snap': 'macro'},
    'XP1': {'contrast': True, 'facts': True, 'politics': True},
    'XP2': {'contrast': True, 'facts': True, 'politics': True, 'identity': True},
    'XP2p': {'contrast': True, 'facts': True, 'politics': True, 'identity': True, 'placebo': True},
    'XP2m': {'contrast': True, 'facts': True, 'politics': True, 'identity': True, 'snap': 'macro'},
    'XP2n': {'contrast': True, 'facts': True, 'politics': True, 'identity': True, 'snap': 'news'},
    'XP1m': {'contrast': True, 'facts': True, 'politics': True, 'snap': 'macro'},
    'XP1n': {'contrast': True, 'facts': True, 'politics': True, 'snap': 'news'},
    'XP': {'contrast': True, 'facts': True, 'personal': True},
    'XG': {'contrast': True, 'facts': True, 'size': True},
    'XGm': {'contrast': True, 'facts': True, 'size': True, 'snap': 'macro'},
    'XGn': {'contrast': True, 'facts': True, 'size': True, 'snap': 'news'},
    'XPG': {'contrast': True, 'facts': True, 'personal': True, 'size': True},
    'XPGp': {'contrast': True, 'facts': True, 'personal': True, 'size': True, 'placebo': True},
    'XPGm': {'contrast': True, 'facts': True, 'personal': True, 'size': True, 'snap': 'macro'},
    'XPGn': {'contrast': True, 'facts': True, 'personal': True, 'size': True, 'snap': 'news'},
    # T22 richer persona: the persona's own answers to other questions in their earlier interview
    'XR': {'contrast': True, 'facts': True, 'history': True},
    'XRm': {'contrast': True, 'facts': True, 'history': True, 'snap': 'macro'},
    'XRn': {'contrast': True, 'facts': True, 'history': True, 'snap': 'news'},
    'XRp': {'contrast': True, 'facts': True, 'history': True, 'placebo': True},
    'XPGR': {'contrast': True, 'facts': True, 'history': True, 'personal': True, 'size': True},
    'XPGRm': {'contrast': True, 'facts': True, 'history': True, 'personal': True, 'size': True, 'snap': 'macro'},
    'XPGRn': {'contrast': True, 'facts': True, 'history': True, 'personal': True, 'size': True, 'snap': 'news'},
    # T25 interpretation: B information control, M country-direction channel, S salience instruction
    **{f'{k}{n}': {'contrast': True, 'facts': True, 'politics': True, 'snap': 'newsboth' if n else 'both', **extra}
       for k, extra in [('B', {}), ('M', {'mood': True}), ('S', {'salience': True}), ('MS', {'mood': True, 'salience': True}),
                         ('PE', {'peers': True})]
       for n in ('', 'n')},
    # T28 (direction step; each persona only in its own diet: 0 figures+news, 1 figures, 2 news)
    **{f'{fam}{suf}': {'contrast': True, 'facts': True, 'diet': d, **({'snap': snap} if snap else {}), **extra}
       for fam, extra in [('W60', {'window': 60}), ('W90', {'window': 90}), ('CV', {'covid': True}), ('HP', {'hp': True}),
                          ('REL', {'rel': True}),
                          ('SOL', {'model': 'gpt-6-sol', 'reasoning': 'medium', 'max_tokens': 4000})]
       for d, suf, snap in [(0, '', None), (1, 'm', 'macro'), (2, 'n', 'news')]},
    'HPp': {'contrast': True, 'facts': True, 'hp': True, 'placebo': True},
    # T48 (registered 2026-11-18 before any call): uniform exposure layer over every persona attribute. Persona block: all
    # attributes as facts plus one exposure sentence per attribute with an economic channel (identity lens, stock wealth,
    # work, budget, housing, place); each snapshot: who holds the presidency and a household block with each channel's
    # real-time indicator; question: which snapshot do YOU feel better about; direction and size in one call. Diet by persona
    # id, COVID panel for diets 0/1. IUp* = placebo (identical snapshots).
    **{f'{fam}{suf}': {'contrast': True, 'facts': True, 'politics': True, 'perspective': True, 'size': True, 'uniform': True,
                       'diet': d, **({'snap': snap} if snap else {}), **({'covid': True} if d < 2 else {}), **extra}
       for fam, extra in [('IU', {}), ('IUp', {'placebo': True})]
       for d, suf, snap in [(0, '', None), (1, 'm', 'macro'), (2, 'n', 'news')]},
    # T56 pilot (audit #2): the news-only diet without the household block of real-time figures
    'IUnx': {'contrast': True, 'facts': True, 'politics': True, 'perspective': True, 'size': True, 'uniform': True, 'no_household': True,
             'diet': 2, 'snap': 'news'},
    # T49 ablations of the adopted design (pilot base): everyone in one information diet (only personas whose own diet differs
    # get a new call), and no COVID case panel for diets 0/1
    **{f'IUa{d}': {'contrast': True, 'facts': True, 'politics': True, 'perspective': True, 'size': True, 'uniform': True,
                   'not_diet': d, **({'snap': snap} if snap else {}), **({'covid': True} if d < 2 else {})}
       for d, snap in [(0, None), (1, 'macro'), (2, 'news')]},
    **{f'IUc{suf}': {'contrast': True, 'facts': True, 'politics': True, 'perspective': True, 'size': True, 'uniform': True,
                     'diet': d, **({'snap': snap} if snap else {})}
       for d, suf, snap in [(0, '', None), (1, 'm', 'macro')]},
    # T43 identity-forward persona (registered 2026-11-18 before any call): partisan lens (IF only) and household stakes in
    # the persona block, who holds the presidency in each snapshot, and the question "which snapshot do YOU feel better
    # about"; direction and size in one call; diet by persona id and COVID panel as in the frozen design. IFp* = placebo.
    **{f'{fam}{suf}': {'contrast': True, 'facts': True, 'politics': True, 'perspective': True, 'size': True, 'stakes': True,
                       'diet': d, **({'snap': snap} if snap else {}), **({'covid': True} if d < 2 else {}), **extra}
       for fam, extra in [('IF', {'lens': True}), ('IS', {}), ('IFp', {'lens': True, 'placebo': True})]
       for d, suf, snap in [(0, '', None), (1, 'm', 'macro'), (2, 'n', 'news')]},
    # FROZEN DESIGN for the test (T28 re-freeze): diet fixed by persona id, COVID cases/deaths in the figures panel
    'FX': {'contrast': True, 'facts': True, 'diet': 0, 'covid': True},
    'FXm': {'contrast': True, 'facts': True, 'diet': 1, 'snap': 'macro', 'covid': True},
    'FXn': {'contrast': True, 'facts': True, 'diet': 2, 'snap': 'news'},
    'FP': {'contrast': True, 'facts': True, 'personal': True, 'size': True, 'diet': 0},
    'FPm': {'contrast': True, 'facts': True, 'personal': True, 'size': True, 'snap': 'macro', 'diet': 1},
    'FPn': {'contrast': True, 'facts': True, 'personal': True, 'size': True, 'snap': 'news', 'diet': 2},
    # secondary, registered for the test: party-relative wording (direction step), same diet rule
    'FR': {'contrast': True, 'facts': True, 'diet': 0, 'rel': True, 'covid': True},
    'FRm': {'contrast': True, 'facts': True, 'diet': 1, 'snap': 'macro', 'rel': True, 'covid': True},
    'FRn': {'contrast': True, 'facts': True, 'diet': 2, 'snap': 'news', 'rel': True},
    # T28 current method restricted to the same diet rule (re-uses X / Xm / Xn answers; no new calls needed)
    # T24 ORACLE ceiling: news picked by the topic mix month-t respondents reported hearing (national / party x income)
    'XO': {'contrast': True, 'facts': True, 'oracle': 'nat'},
    'XOn': {'contrast': True, 'facts': True, 'oracle': 'nat', 'snap': 'news'},
    'XQ': {'contrast': True, 'facts': True, 'oracle': 'grp'},
    'XQn': {'contrast': True, 'facts': True, 'oracle': 'grp', 'snap': 'news'},
    'Xn': {'contrast': True, 'facts': True, 'snap': 'news'},
    'Xs': {'contrast': True, 'facts': True, 'own_view': True},
    'XP2_grok': {'contrast': True, 'facts': True, 'politics': True, 'identity': True, 'model': 'grok-4-1-fast-non-reasoning'},
    'XP2_qwen': {'contrast': True, 'facts': True, 'politics': True, 'identity': True, 'model': 'qwen3.5-flash'},
    # same contrast prompt (X) on other model families
    **{f'X_{k}': {'contrast': True, 'facts': True, 'model': m} for k, m in [
        ('grok', 'grok-4-1-fast-non-reasoning'), ('oss', 'gpt-oss-120b'), ('gem31', 'gemini-3.1-flash-lite'),
        ('gem35', 'gemini-3.5-flash-lite'), ('qwen', 'qwen3.5-flash'), ('doubao', 'doubao-seed-1-6-flash-250615'),
        ('nano', 'gpt-5.4-nano'), ('dsv4', 'deepseek-v4-flash')]},
    'He': {'macro': True, 'econ_news': True, 'facts': 'own', 'style': 'keep'},
    'Hes': {'macro': True, 'econ_news': True, 'facts': 'shuffle', 'style': 'keep'},
    'E3k_ds': {'macro': True, 'news': True, 'style': 'keep', 'model': 'deepseek-v4.1-flash'},
    'C0k_ds': {'style': 'keep', 'model': 'deepseek-v4.1-flash'},
    'Hm': {'macro': True, 'news': True, 'memory': True, 'facts': 'own', 'style': 'keep'},
    'Hms': {'macro': True, 'news': True, 'memory': True, 'facts': 'shuffle', 'style': 'keep'},
    'E1k': {'macro': True, 'style': 'keep'},
    'E2k': {'news': True, 'style': 'keep'},
    'Dk': {'date': True, 'style': 'keep'},
    'E3a': {'macro': True, 'news': True, 'style': 'keep', 'adjacent': True},
    'Ha': {'macro': True, 'news': True, 'facts': 'own', 'style': 'keep', 'adjacent': True},
    'Has': {'macro': True, 'news': True, 'facts': 'shuffle', 'style': 'keep', 'adjacent': True},
}
T9_S_MONTHS = None


def screen_months():
    # T9 S months (every third pre-COVID month from 2014-07, 23 months) plus each shock month and the month before
    s = D.PRE_COVID[::3]
    extra = [201610, 201611, 201812, 201901, 201907, 201908, 202002, 202003, 202004]
    return sorted(set(s) | set(extra))


def git_commit():
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return None


def fact_fields(row):
    return {c: (None if pd.isna(row[c]) else int(row[c])) for c in D.FACT_COLS}


def build(run, months, k, max_n=None):
    rd = RUNS / run
    rd.mkdir(parents=True, exist_ok=True)
    pops = {}
    for t in months:
        need_party = bool(D.pool_months(t, k, need_party=True))
        pl = D.pool(t, k, need_party=need_party)
        if max_n and len(pl) > max_n:
            pl = pl.sample(max_n, random_state=SEED % 2**31 + t, weights=None)
        pop, target, pub = assimilate(pl, t, seed=SEED)
        r, ix = aggregate(pop)
        assert abs(ix['ICS'] - pub.ICS) < 0.05, (t, ix, pub.ICS)
        rng = np.random.default_rng([SEED, t, 1])
        start, end = D.interview_start(t), D.dates()[t]['interview_end']
        end = dt.date.fromisoformat(end)
        ndays = (end - start).days + 1
        pop['day'] = [start + dt.timedelta(days=int(x)) for x in rng.integers(0, ndays, len(pop))]
        perm = rng.permutation(len(pop))
        pop['shuffle_src'] = perm  # facts of persona perm[i] shown to persona i in shuffle arm
        pops[t] = {'pop': pop, 'target': target, 'pub_ics': float(pub.ICS), 'pool_months': D.pool_months(t, k, need_party),
                   'need_party': need_party}
        print(t, len(pop), 'pool', pops[t]['pool_months'], 'ICS', round(ix['ICS'], 3), flush=True)
    (rd / 'populations.pkl').write_bytes(pickle.dumps(pops))
    manifest = {'run': run, 'created': dt.datetime.now().isoformat(), 'model': engine.MODEL, 'route': engine.ROUTE,
                'template_hash': template_hash(), 'seed': SEED, 'months': months, 'k': k, 'max_n': max_n,
                'git_commit': git_commit(), 'components': str(ROOT / 'data/components.csv'),
                'components_sha': hashlib.sha256((ROOT / 'data/components.csv').read_bytes()).hexdigest()[:16]}
    (rd / 'manifest.json').write_text(json.dumps(manifest, indent=1))


def info_for(arm, persona, src, day, cache):
    cfg = ARMS[arm]
    info = {}
    if cfg.get('macro'):
        key = ('m', day)
        if key not in cache:
            cache[key] = macro_text(macro(day))
        info['macro'] = cache[key]
    if cfg.get('news'):
        key = ('n', day)
        if key not in cache:
            cache[key] = news_text(news(day))
        info['news'] = cache[key]
    if cfg.get('econ_news'):
        key = ('ne', day)
        if key not in cache:
            cache[key] = news_text(news(day, econ=True), econ=True)
        info['news'] = cache[key]
    if cfg.get('info_memory'):
        key = ('im', day)
        if key not in cache:
            prev_day = day - dt.timedelta(days=30)
            blocks = [macro_text(macro(prev_day))]
            if cfg.get('econ_news'):
                blocks.append(news_text(news(prev_day, econ=True), econ=True))
            cache[key] = 'When you gave those answers, you already knew the following:\n\n' + '\n\n'.join(blocks)
        info['memory'] = cache[key]
    if cfg.get('memory'):
        key = ('mem', day)
        if key not in cache:
            prev = news(day - dt.timedelta(days=30), n=8)
            cache[key] = news_text(prev).replace(
                'Front-page news stories over the past month (most-covered topics first):',
                'At your last interview, about a month ago, you had already heard about these front-page stories:')
        info['memory'] = cache[key]
    if cfg.get('facts') == 'own':
        info['facts'] = facts_text(persona, day)
    elif cfg.get('facts') == 'shuffle':
        info['facts'] = facts_text(src, day)
    return info


def snapshot(day, cache, kind='econ'):
    key = ('snap', kind, day)
    if key not in cache:
        if kind in ('news', 'newsboth'):
            txt = news_text(news(day, econ=True), econ=True)
        else:
            txt = macro_levels_text(macro(day))
            if kind == 'prices':
                txt += '\n\n' + prices_text(day)
            if kind != 'macro':
                txt += '\n\n' + news_text(news(day, econ=True), econ=True)
        if kind in ('both', 'newsboth'):  # add the other (non-economic) most-covered front-page stories
            econ_ids = {a['id'] for a in news(day, econ=True)}
            other = tuple(a for a in news(day) if a['id'] not in econ_ids)[:8]
            txt += '\n\n' + news_text(other).replace('Front-page news stories over the past month (most-covered topics first):',
                                                       'Other front-page news stories over the past month (most-covered topics first):')
        cache[key] = txt
    return cache[key]


def contrast_requests(run, arm, pops, months, limit=None):
    cfg = ARMS[arm]
    cache, reqs = {}, []
    only = RUNS / run / 'only_pids.json'          # T56: re-ask only the personas whose prompt a fix changes
    only = json.loads(only.read_text()) if only.exists() else None
    for t in months:
        pop = pops[t]['pop']
        rows = pop.to_dict('records')[:limit] if limit else pop.to_dict('records')
        for p in rows:
            if only is not None and p['pid'] not in set(only.get(str(t), [])):
                continue
            day = p['day']
            kind = cfg.get('snap', 'econ')
            if cfg.get('diet') is not None and ext28.diet_of(p['pid']) != cfg['diet']:
                continue   # T28: each persona answers only in its own information diet
            if cfg.get('not_diet') is not None and ext28.diet_of(p['pid']) == cfg['not_diet']:
                continue   # T49 ablation: only personas whose own diet differs need a new call
            pday = day - dt.timedelta(days=cfg.get('window', 30))
            now, prev = snapshot(day, cache, kind), snapshot(pday, cache, kind)
            if cfg.get('hp'):
                hpn = lambda d0: ext28.hp_levels_text(d0) + ('\n\n' + news_text(news(d0, econ=True), econ=True) if kind == 'econ' else '')
                now, prev = hpn(day), hpn(pday)
            if cfg.get('rel'):
                rd = ext28.reader(p)
                rn = lambda d0: ((snapshot(d0, cache, 'macro') + '\n\n') if kind != 'news' else '') + \
                    (ext28.rel_news_text(d0, rd) + '\n\n' if kind != 'macro' else '') + ext28.relative_situation(d0, rd)
                now, prev = rn(day), rn(pday)
            if cfg.get('covid'):
                both = (bool(ext28.covid_lines(day)) and bool(ext28.covid_lines(pday))) if os.environ.get('T56_SNAPFIX') == '1' else True   # T56 test, not adopted
                cl = lambda d0: ('\n' + '\n'.join('- ' + x for x in ext28.covid_lines(d0))) if both and ext28.covid_lines(d0) else ''
                now = now.replace('Economic figures in the news:\n', 'Economic figures in the news:\n', 1)
                head = 'Economic figures in the news:\n'
                def addc(txt, d0):
                    i = txt.find(head)
                    j = txt.find('\n\n', i) if i >= 0 else -1
                    return txt[:j] + cl(d0) + txt[j:] if j >= 0 else txt + cl(d0)
                now, prev = addc(now, day), addc(prev, pday)
            if cfg.get('oracle'):  # T24 ORACLE ONLY: news chosen by the topics month-t respondents reported hearing
                import oracle24
                on, op = oracle24.snapshot_pair(p, day, t, cfg['oracle'])
                if kind == 'news':
                    now, prev = on, op
                else:
                    now = snapshot(day, cache, 'macro') + '\n\n' + on
                    prev = snapshot(day - dt.timedelta(days=30), cache, 'macro') + '\n\n' + op
            if cfg.get('politics'):
                now = now + '\n\n' + political_text(day)
                prev = prev + '\n\n' + political_text(pday)
            if cfg.get('personal'):
                now = now + '\n\n' + personal_text(p, day)
                prev = prev + '\n\n' + personal_text(p, pday)
            if cfg.get('uniform') and not cfg.get('no_household'):
                hn, hp = household_uniform(p, day, other=pday), household_uniform(p, pday, other=day)
                now = now + ('\n\n' + hn if hn else '')
                prev = prev + ('\n\n' + hp if hp else '')
            if cfg.get('placebo'):
                prev = now
            rng = np.random.default_rng([SEED, t, int(p['pid'].split('-')[1])])
            now_label = 'A' if rng.random() < 0.5 else 'B'
            a, b = (now, prev) if now_label == 'A' else (prev, now)
            facts = facts_text(p, day) if cfg.get('facts') else None
            if facts and cfg.get('lens') and lens_text(p):
                facts += ' ' + lens_text(p)
            if facts and cfg.get('stakes') and stakes_text(p):
                facts += ' ' + stakes_text(p)
            if facts and cfg.get('uniform'):
                facts = ' '.join(x for x in (facts, family_text(p), exposure_profile(p, job_exposed(p))) if x)
            if cfg.get('rel') and facts:   # no names: the presidency is stated relative to the persona in each snapshot
                facts = facts.replace(' ' + president_line(day), '')
            if cfg.get('history'):
                h = history_text(D.attitudes().get((str(p['CASEID']).strip(), int(p['YYYYMM']))), int(p['YYYYMM']), t)
                facts = facts + ('\n\n' + h if h else '')
            reqs.append({'rid': f"{run}|{arm}|{t}|{p['pid']}", 'run': run, 'parser': 'contrast', 'now_label': now_label,
                         'messages': render_contrast(facts, a, b, ask_change=cfg.get('ask_change', True),
                                                     topics=TOPICS_V2 if cfg.get('topics') == 'v2' else None,
                                                     own_view=cfg.get('own_view', False), size=cfg.get('size', False),
                                                     identity=cfg.get('identity', False), mood=cfg.get('mood', False),
                                                     salience=cfg.get('salience', False), peers=cfg.get('peers', False),
                                                     perspective=cfg.get('perspective', False)),
                         'max_tokens': cfg.get('max_tokens', 300), 'model': cfg.get('model'), 'reasoning': cfg.get('reasoning')})
    return reqs


def requests_for(run, arm, pops, months, limit=None):
    if ARMS[arm].get('contrast'):
        return contrast_requests(run, arm, pops, months, limit)
    cache = {}
    reqs = []
    for t in months:
        pop = pops[t]['pop']
        rows = pop.to_dict('records')
        if limit:
            rows = rows[:limit]
        for p in rows:
            src = pop.iloc[int(p['shuffle_src'])].to_dict()
            msgs = render(p, p['day'], ARMS[arm], info_for(arm, p, src, p['day'], cache))
            reqs.append({'rid': f"{run}|{arm}|{t}|{p['pid']}", 'run': run, 'messages': msgs, 'model': ARMS[arm].get('model'),
                         'reasoning': ARMS[arm].get('reasoning'), 'max_tokens': ARMS[arm].get('max_tokens'),
                         'prev': {q: p['S_' + q] for q in Q}})
    return reqs


def call(run, arms, cap, months=None, limit=None, concurrency=96):
    rd = RUNS / run
    pops = pickle.loads((rd / 'populations.pkl').read_bytes())
    months = months or sorted(pops)
    for arm in arms:
        reqs = requests_for(run, arm, pops, months, limit)
        ex = rd / f'examples_{arm}.json'
        if not ex.exists():
            ex.write_text(json.dumps([r['messages'] for r in reqs[:3]], indent=1, ensure_ascii=False))
        print(f'[{arm}] {len(reqs)} requests', flush=True)
        asyncio.run(engine.run_requests(reqs, rd / f'results_{arm}.jsonl', cap_usd=cap, concurrency=concurrency))


def load_results(rd, arm):
    out = {}
    p = rd / f'results_{arm}.jsonl'
    if p.exists():
        for line in p.open():
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue  # interleaved write from a parallel stream; the rid is re-requested on resume
            out[x['rid']] = x
    return out


def agg(run):
    rd = RUNS / run
    pops = pickle.loads((rd / 'populations.pkl').read_bytes())
    rows = []
    for arm in ARMS:
        res = load_results(rd, arm)
        if not res:
            continue
        for t, P in sorted(pops.items()):
            pop = P['pop'].copy()
            got = 0
            changed = {q: 0.0 for q in Q}
            for q in Q:
                pop['N_' + q] = pop['S_' + q]
            for i, p in pop.iterrows():
                x = res.get(f"{run}|{arm}|{t}|{p['pid']}")
                if x is None:
                    continue
                got += 1
                if x['states']:
                    for q in Q:
                        pop.at[i, 'N_' + q] = x['states'][q]
            have = pop.pid.map(lambda pid: f"{run}|{arm}|{t}|{pid}" in res)
            if have.sum() == 0:
                continue
            sub = pop[have].copy()
            sub['w'] = sub.w / sub.w.sum()
            r0, ix0 = aggregate(sub, 'S_')
            r1, ix1 = aggregate(sub, 'N_')
            valid = sum(1 for pid in sub.pid if res[f"{run}|{arm}|{t}|{pid}"]['states'])
            rec = {'arm': arm, 'month': t, 'n': len(sub), 'valid': valid,
                   'ICS_start': ix0['ICS'], 'pub_prev': P['pub_ics'],
                   # simulated level = published t-1 ICS + simulated change on this population
                   'ICS': P['pub_ics'] + (ix1['ICS'] - ix0['ICS']),
                   'ICS_raw': ix1['ICS'], 'ICC': ix1['ICC'], 'ICE': ix1['ICE']}
            for q in Q:
                rec['dRS_' + q] = r1[q] - r0[q]
                rec['chg_' + q] = float((sub.w * (sub['N_' + q] != sub['S_' + q])).sum())
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(rd / 'monthly.csv', index=False)
    return df


def reuse(src, dst, arms):
    """Copy finished results of identical persona-months from run `src` into run `dst` (rid rewritten)."""
    S = pickle.loads((RUNS / src / 'populations.pkl').read_bytes())
    Dp = pickle.loads((RUNS / dst / 'populations.pkl').read_bytes())
    cols = ['pid', 'CASEID', 'w', 'day', *['S_' + q for q in Q]]
    same = [t for t in S if t in Dp and S[t]['pop'][cols].equals(Dp[t]['pop'][cols])]
    for arm in arms:
        res = load_results(RUNS / src, arm)
        have = load_results(RUNS / dst, arm)
        n = 0
        with (RUNS / dst / f'results_{arm}.jsonl').open('a') as f:
            for rid, x in res.items():
                _, a, t, pid = rid.split('|')
                if int(t) in same:
                    new = f'{dst}|{a}|{t}|{pid}'
                    if new not in have:
                        f.write(json.dumps({**x, 'rid': new, 'reused_from': rid}) + '\n')
                        n += 1
        print(arm, 'reused', n, 'from', len(same), 'identical months')


def drop_failed(runid, arm):
    """Remove failed results (e.g. HTTP 503 after 3 attempts) so a resumed call re-requests them."""
    p = RUNS / runid / f'results_{arm}.jsonl'
    lines = p.read_text().splitlines()

    def ok(l):
        try:
            return bool(json.loads(l)['states'])
        except Exception:      # interleaved write: drop the line, the rid is re-requested
            return False
    keep = [l for l in lines if ok(l)]
    (RUNS / runid / f'failed_{arm}.jsonl').open('a').write('\n'.join(l for l in lines if not ok(l)) + '\n')
    p.write_text('\n'.join(keep) + '\n')
    print(arm, 'dropped', len(lines) - len(keep), 'failed of', len(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--run', required=True)
    ap.add_argument('--months', default='screen')
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--max-n', type=int, default=None)
    ap.add_argument('--arms', default='C0')
    ap.add_argument('--cap', type=float, default=0)
    ap.add_argument('--limit-per-month', type=int, default=None)
    ap.add_argument('--concurrency', type=int, default=96)
    ap.add_argument('--src', default=None)
    a = ap.parse_args()
    if a.months == 'screen':
        months = screen_months()
    elif a.months == 'all':
        months = D.ALL_SCORED
    elif a.months == 'valid':
        months = D.VALID
    else:
        months = [int(x) for x in a.months.split(',')]
    if a.cmd == 'build':
        build(a.run, months, a.k, a.max_n)
    elif a.cmd == 'call':
        call(a.run, a.arms.split(','), a.cap, months if a.months not in ('screen',) else None, a.limit_per_month, a.concurrency)
    elif a.cmd == 'retry':
        for arm in a.arms.split(','):
            drop_failed(a.run, arm)
        call(a.run, a.arms.split(','), a.cap, months if a.months != 'screen' else None, None, a.concurrency)
    elif a.cmd == 'reuse':
        reuse(a.src, a.run, a.arms.split(','))
    elif a.cmd == 'agg':
        print(agg(a.run).to_string())
