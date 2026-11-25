"""Scoring: ICS error metrics and paired moving-block bootstrap (same settings as T9 F: block 6, 1000 reps)."""
import json
import math

import numpy as np

from data import V3, PRE_COVID, COVID, ym_add, SHOCKS

T9K = V3 / 'clean/T9_20261115/K'


def official_target():
    """Official final ICS. T9 table for months it covers; T3 table (identical on overlap) for 2021-2022.
    The T3 table ends at 2022-12, so no test-set target is ever loaded."""
    import pandas as pd
    off = json.loads((T9K / 'official.json').read_text())
    out = {int(k): v['ICS'] for k, v in off.items() if v.get('ICS') is not None}
    o = pd.read_csv(V3 / 'clean/T3_20261113/official/official_indices.csv').dropna(subset=['ICS'])
    for m, v in zip(o.month.astype(int), o.ICS):
        if 202101 <= m <= 202212:
            out[m] = float(v)
    from data import TEST_MODE, ROOT
    if TEST_MODE:   # test months: ICS printed in the Final Results PDF (matches the T3 table to 0.1 on 2013-2022)
        t = pd.read_csv(ROOT / 'data/components_test.csv')
        for m, v in zip(t.month.astype(int), t.ICS):
            if m >= 202301 and v == v:
                out[m] = float(v)
    return out


def _ics_2a(m):
    """2a: the month t-12 respondents' own answers, official weights (reproduces T9 baseline_F exactly)."""
    from data import micro, Q
    d = micro()
    g = d[d.YYYYMM == ym_add(m, -12)]
    w = g.WT.values
    return sum(100 + 100 * (w[g['S_' + q].values == 'F'].sum() - w[g['S_' + q].values == 'U'].sum()) / w.sum()
               for q in Q) / 6.7558 + 2


def baselines():
    from data import ALL_SCORED, VALID
    import pandas as pd
    off = official_target()
    b2a = {int(k): v['ICS'] for k, v in json.loads((T9K / 'baseline_F.json').read_text()).items()}
    for m in VALID:
        b2a[m] = _ics_2a(m)
    pers = {m: off[ym_add(m, -1)] for m in off if ym_add(m, -1) in off}
    p = pd.read_csv(V3 / 'clean/T3_20261113/S3/ar/predictions.csv')
    ar = {int(m): v for m, v in zip(p.month, p.ICS)}
    return {'2a': b2a, 'persistence': pers, 'AR': ar}


def metrics(series, months=PRE_COVID):
    off = official_target()
    ms = [m for m in months if m in series]
    e = np.array([series[m] - off[m] for m in ms])
    prev = np.array([off[ym_add(m, -1)] for m in ms])
    sim_ch = np.array([series[m] for m in ms]) - prev  # change relative to last official value
    off_ch = np.array([off[m] for m in ms]) - prev
    calm = np.array([m not in SHOCKS for m in ms])
    out = {'n': len(ms), 'rmse': float(np.sqrt(np.mean(e ** 2))), 'bias': float(e.mean()),
           'change_corr': float(np.corrcoef(sim_ch, off_ch)[0, 1]) if sim_ch.std() > 0 else None,
           'sign_acc': float(np.mean(np.sign(sim_ch) == np.sign(off_ch))),
           'sim_change_sd': float(sim_ch.std()), 'off_change_sd': float(off_ch.std()),
           'calm_sim_change_sd': float(sim_ch[calm].std()) if calm.any() else None}
    return out


def paired(a, b, months=PRE_COVID, block=6, reps=1000, seed=20261115):
    off = official_target()
    ms = [m for m in months if m in a and m in b]
    ea = np.array([a[m] - off[m] for m in ms])
    eb = np.array([b[m] - off[m] for m in ms])
    n = len(ms)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(reps):
        starts = rng.integers(0, n - block + 1, math.ceil(n / block))
        take = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        diffs.append(np.sqrt(np.mean(ea[take] ** 2)) - np.sqrt(np.mean(eb[take] ** 2)))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {'n': n, 'diff': float(np.sqrt(np.mean(ea ** 2)) - np.sqrt(np.mean(eb ** 2))),
            'ci_low': float(lo), 'ci_high': float(hi), 'significant_better': bool(hi < 0)}


GROUPS = ['all', 'a1834', 'a3554', 'a5597', 'y13', 'y23', 'y33', 'ehs', 'esc', 'ecd', 'w', 'nc', 'ne', 's', 'm', 'f']
DIMS = {'age': ['a1834', 'a3554', 'a5597'], 'income': ['y13', 'y23', 'y33'], 'education': ['ehs', 'esc', 'ecd'],
        'region': ['w', 'nc', 'ne', 's'], 'sex': ['m', 'f']}


def group_labels(p):
    """Group membership with the same definitions as T9 (common9.group)."""
    age, ed = p.get('AGE'), p.get('EDUC')
    out = {'all'}
    if age is not None and age == age:
        out.add('a1834' if 18 <= age <= 34 else 'a3554' if 35 <= age <= 54 else 'a5597' if age >= 55 else None)
    out.add({1: 'y13', 2: 'y23', 3: 'y33'}.get(p.get('YTL3')))
    out.add('ehs' if ed in (1, 2, 3) else 'esc' if ed == 4 else 'ecd' if ed in (5, 6) else None)
    out.add({1: 'w', 2: 'nc', 3: 'ne', 4: 's'}.get(p.get('REGION')))
    out.add({1: 'm', 2: 'f'}.get(p.get('SEX')))
    out.discard(None)
    return out


def group_ics(df, prefix):
    """Weighted ICS per group from a persona frame with state columns prefix+Q."""
    from data import Q, rs, indices_from_rs
    labs = [group_labels(r) for r in df[['AGE', 'EDUC', 'YTL3', 'REGION', 'SEX']].to_dict('records')]
    out = {}
    for g in GROUPS:
        m = np.array([g in l for l in labs])
        if m.sum() < 5:
            out[g] = None
            continue
        sub = df[m]
        out[g] = indices_from_rs({q: rs(sub[prefix + q], sub.w) for q in Q})['ICS']
    return out


def official_groups():
    d = json.loads((T9K / 'official_groups.json').read_text())
    return {int(k): {g: v['groups'][g]['value'] for g in GROUPS} for k, v in d.items()}


def e_grp(sim_groups, months):
    """Mean absolute error of group-minus-national levels (T9 E_grp definition)."""
    og = official_groups()
    errs = []
    for m in months:
        a, b = sim_groups[m], og[m]
        for g in GROUPS:
            if a.get(g) is not None and b.get(g) is not None:
                errs.append(abs((a[g] - a['all']) - (b[g] - b['all'])))
    return float(np.mean(errs))


def direction(series, months, big=3.0, components=None, sim_components=None):
    """Month-to-month direction accuracy against the official change (both relative to last month's official ICS).
    A predicted change of exactly 0 counts as a miss. Also reported on 'big' months (|official change| >= big)
    and, if given, per question (component RS changes)."""
    from math import comb
    off = official_target()
    ms = [m for m in months if m in series]
    o = np.array([off[m] - off[ym_add(m, -1)] for m in ms])
    s = np.array([series[m] - off[ym_add(m, -1)] for m in ms])
    hit = np.sign(s) == np.sign(o)
    bigm = np.abs(o) >= big

    def p_one_sided(k, n):  # P(X >= k) under a fair coin
        return float(sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n) if n else None
    out = {'n': len(ms), 'hit': float(hit.mean()), 'hits': int(hit.sum()), 'p_vs_coin': p_one_sided(int(hit.sum()), len(ms)),
           'n_big': int(bigm.sum()), 'hit_big': float(hit[bigm].mean()) if bigm.any() else None,
           'p_big_vs_coin': p_one_sided(int(hit[bigm].sum()), int(bigm.sum()))}
    if components is not None and sim_components is not None:
        from data import Q
        hq = []
        for q in Q:
            oq = np.array([components.loc[m, q] - components.loc[ym_add(m, -1), q] for m in ms])
            sq = np.array([sim_components.get((m, q), 0.0) for m in ms])
            keep = oq != 0
            hq.append((np.sign(sq[keep]) == np.sign(oq[keep])))
            out['hit_' + q] = float(hq[-1].mean())
        allq = np.concatenate(hq)
        out['hit_5q'] = float(allq.mean())
        out['p_5q_vs_coin'] = p_one_sided(int(allq.sum()), len(allq))
    return out
