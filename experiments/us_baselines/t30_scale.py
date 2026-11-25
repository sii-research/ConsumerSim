"""T30: what does a larger simulated population buy?  Sub-sample curves from the ~1,800-per-month run (Fig. 6 analogue).

For N in NS, draw R random sub-populations per month (without replacement, weights renormalised), recompute the frozen
update shares (the consensus gain uses the sub-population's own net direction) and score:
  national   - ICS RMSE (147 months), five-question direction (>1 SE), Monte-Carlo sd of the monthly simulated change
  subgroups  - correlation of simulated vs official gap changes, six contrasts, and the simulated gap's own noise sd
  weekly     - 1-week and 4-week correlation of the interview-week series with the S&P 500
Run with CS_TEST=1:  python3 t30_scale.py 1800
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

assert os.environ.get('CS_TEST') == '1'
import data as D
import eval3 as E
import frozen as F
import monitor as MON
import t29_score as S
from cases13 import PAIRS, official_group
from score import official_target
from t30_frames import OUT as OUT_T30
from design import NORMS, FRAMES_1800, OUTDIR as OUT

E.COMP = D.components()
NS = [50, 100, 200, 400, 800, 1200, 1800]
R = 12
MONTHS = D.ALL_SCORED + D.VALID + D.TEST


def official_gaps():
    path = OUT_T30 / 'official_gaps.pkl'
    if path.exists():
        return pickle.loads(path.read_bytes())
    rows = []
    for name, (s1, s2) in PAIRS.items():
        for m in MONTHS:
            a1, e1 = official_group(m, s1)
            a0, e0 = official_group(D.ym_add(m, -1), s1)
            b1, f1 = official_group(m, s2)
            b0, f0 = official_group(D.ym_add(m, -1), s2)
            rows.append({'pair': name, 'month': m, 'off_gap_change': (a1 - b1) - (a0 - b0), 'se': np.sqrt(e1 ** 2 + e0 ** 2 + f1 ** 2 + f0 ** 2),
                         'off_g1': a1 - a0, 'off_g2': b1 - b0, 'se1': np.hypot(e1, e0), 'se2': np.hypot(f1, f0)})
    T = pd.DataFrame(rows)
    path.write_bytes(pickle.dumps(T))
    return T


def sim_gaps(P, H):
    out = {}
    for name, (s1, s2) in PAIRS.items():
        c1, c2 = S.group_change(P, H, 'ds_F', s1), S.group_change(P, H, 'ds_F', s2)
        out[name] = {m: c1.get(m, np.nan) - c2.get(m, np.nan) for m in MONTHS}
    return out


def score(P, OG, off):
    import design
    H = design.H(P)
    s, drs = MON.method_series(P, H, 'ds_F')
    ms = [m for m in MONTHS if m in s]
    t = E.items(s, drs)
    five = t[(t.item != 'ICS') & (t.z > 1) & t.month.isin(ms)]
    r = {'rmse': float(np.sqrt(np.mean([(s[m] - off[m]) ** 2 for m in ms]))), 'dir5': float(five.hit.mean()),
         'chg': {m: s[m] - off[D.ym_add(m, -1)] for m in ms}}
    sg = sim_gaps(P, H)
    for name in PAIRS:
        g = OG[OG.pair == name].set_index('month')
        x = pd.Series(sg[name]).reindex(g.index)
        ok = x.notna() & g.off_gap_change.notna()
        r[f'corr {name}'] = float(np.corrcoef(x[ok], g.off_gap_change[ok])[0, 1])
        r[f'gap {name}'] = x
    try:
        w = S.weekly(P, H, 'ds_F', '2014-07-01', '2026-09-30')
        r['week1'], r['week4'] = w['corr 1-week'], w['corr 4-week']
    except Exception:
        r['week1'] = r['week4'] = np.nan
    return r


def main(scale):
    fr = pickle.load(open(FRAMES_1800, 'rb'))
    P = fr['P']
    off = official_target()
    OG = official_gaps()
    rows, draws = [], {}
    rng = np.random.default_rng(20261117)
    for n in NS:
        reps = 1 if n >= P.groupby('month').size().min() else R
        for k in range(reps):
            if reps == 1:
                Q = P
            else:
                idx = np.concatenate([rng.choice(ix, size=min(n, len(ix)), replace=False) for ix in P.groupby('month').indices.values()])
                Q = P.iloc[np.sort(idx)].copy()
                Q['w'] = Q.w / Q.groupby('month').w.transform('sum')
            r = score(Q, OG, off)
            draws.setdefault(n, []).append(r)
            rows.append({'N': n, 'rep': k} | {a: b for a, b in r.items() if not isinstance(b, (dict, pd.Series))})
            print(n, k, {a: round(b, 3) for a, b in rows[-1].items() if isinstance(b, float)}, flush=True)
    df = pd.DataFrame(rows)
    # Monte-Carlo noise of the simulated monthly change and of each simulated gap change, across sub-samples
    noise = []
    for n, rs in draws.items():
        if len(rs) < 2:
            continue
        ch = pd.DataFrame([r['chg'] for r in rs])
        # sub-samples are drawn without replacement from ~N_month personas: divide by sqrt(1 - n/N) (finite-population correction)
        fpc = float(np.sqrt(np.mean([max(1 - n / k, 1e-9) for k in P.groupby('month').size()])))
        z = {'N': n, 'fpc': fpc, 'sd monthly change': float(ch.std().mean()) / fpc}
        for name in PAIRS:
            z[f'sd gap {name}'] = float(pd.DataFrame([r[f'gap {name}'] for r in rs]).std().mean()) / fpc
        noise.append(z)
    df.to_csv(OUT / f'scale_curve_{scale}.csv', index=False)
    pd.DataFrame(noise).to_csv(OUT / f'scale_noise_{scale}.csv', index=False)
    print(df.groupby('N').mean(numeric_only=True).round(3).to_string())
    print(pd.DataFrame(noise).round(3).to_string())


if __name__ == '__main__':
    main(int(sys.argv[1]))
