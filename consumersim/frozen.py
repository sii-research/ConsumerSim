"""Frozen design (re-frozen at T28): diet fixed by persona id, COVID panel in the figures when data exist, two-step persona,
graded update shares 0.4/0.03 x 0.75 x consensus gain (cap 1.5). Builds the person frame from collected answers.

Train/validation: direction answers come from X/Xm/Xn (identical prompts) except COVID-era months, where diets 0/1
use the CV/CVm answers (same prompt plus the case panel). Size answers from XPG/XPGm/XPGn (identical to FP*).
Test: direction FX/FXm/FXn, size FP/FPm/FPn.
"""
import numpy as np
import pandas as pd

import data as D
import eval3 as E
import ext28
import gate
import run

SC = {'F': 1, 'N': 0, 'U': -1, 'D': 0}
COVID_ERA = (202003, 202303)


def _fill(P, arm_of, runid_of, tag):
    cache = {}
    cols = {f'{k}_{tag}_{q}': np.full(len(P), np.nan) for k in ('dir', 'ds', 'size') for q in D.Q}
    ok = np.zeros(len(P), bool)
    for i, (m, pid, d) in enumerate(zip(P.month.values, P.pid.values, P.diet.values)):
        arm, runid = arm_of(m, d), runid_of(m)
        if (runid, arm) not in cache:
            cache[(runid, arm)] = run.load_results(run.RUNS / runid, arm)
        st = (cache[(runid, arm)].get(f'{runid}|{arm}|{m}|{pid}') or {}).get('states')
        if st is None:
            continue
        ok[i] = True
        for q in D.Q:
            d0 = st[q]['dir']
            s0 = P['S_' + q].values[i]
            cols[f'dir_{tag}_{q}'][i] = d0
            cols[f'ds_{tag}_{q}'][i] = SC[gate.step(s0, d0, q)] - SC[s0]
            sz = st[q].get('size')
            cols[f'size_{tag}_{q}'][i] = sz if sz is not None else -1
    for c, v in cols.items():
        P[c] = v
    return ok


def frame(P, test=False):
    """P: populations with month, pid, w, pub_prev, S_q, facts (e.g. from eval3.person_frame on any arm set)."""
    P = P.copy()
    P['diet'] = [ext28.diet_of(p) for p in P.pid]
    if test:
        dir_arm = lambda m, d: ['FX', 'FXm', 'FXn'][d]
        size_arm = lambda m, d: ['FP', 'FPm', 'FPn'][d]
        rid = lambda m: 'T29_T'
    else:
        dir_arm = lambda m, d: (['CV', 'CVm', 'Xn'] if COVID_ERA[0] <= m <= COVID_ERA[1] else ['X', 'Xm', 'Xn'])[d]
        size_arm = lambda m, d: ['XPG', 'XPGm', 'XPGn'][d]
        rid = lambda m: 'T11_V' if m >= 202101 else 'T10_F'
    a = _fill(P, dir_arm, rid, 'F')
    b = _fill(P, size_arm, rid, 'pd')
    P = P[a & b].copy()
    P['w'] = P.w / P.groupby('month').w.transform('sum')
    return P


def H_frozen(P, name='F', g=1.5, cap=1.5, scale=0.75, norm=None):
    """Update shares. norm: training-period mean |net direction| per question (fixed at freezing)."""
    month = P.month.values
    H, norms = {}, {}
    for q in D.Q:
        sz = P[f'size_pd_{q}'].values
        h = np.where(sz == 2, 0.4, np.where(sz == 1, 0.03, 0.0)) * scale
        d = P[f'dir_{name}_{q}'].values.astype(float)
        net = np.abs(pd.Series(d).groupby(month).transform('mean').values)
        nq = norm[q] if norm is not None else float(np.mean(net[month < 202101]))
        norms[q] = nq
        H[q] = np.minimum(0.95, h * np.minimum(cap, (net / nq) ** g))
    return H, norms
