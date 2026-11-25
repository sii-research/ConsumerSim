"""T49 Table 4: mechanism ablation of the adopted design (IU: uniform exposure layer), on the 41 pilot months
(training and validation, ~600 personas/month, the same personas in every row).

Every row removes one component; rows that need different LLM answers use answers collected for them (IUa*: everyone in one
information diet; IUc*: no COVID case panel; frozen design: no identity and exposure layer; E3e: direct re-answer without
contrast elicitation). Metrics: MAE, RMSE (all 41 months, pandemic months), five-question direction, and the subgroup
fidelity that the layer is meant to deliver (party and stock-gap correlation with the Michigan microdata; mean over
contrasts of |simulated - official| monthly gap variability, a dispersion error). Run with CS_TEST=1 after t49_launch.sh.
"""
import os

import numpy as np
import pandas as pd

assert os.environ.get('CS_TEST') == '1'
import data as D
import eval3 as E
import ext28
import frozen as F
import monitor as MON
import design
from design import NORMS_IU, NORMS_FROZEN
from score import official_target
from t30_frames import base
from t43_pilot import H_of, group_change
from t47_micro_scan import CON
from t48_pilot import MONTHS, official_monthly_gaps

E.COMP = D.components()
OUT = D.ROOT / 'receipts' / 'T49'
OUT.mkdir(parents=True, exist_ok=True)
COVID = [m for m in MONTHS if 202003 <= m <= 202212]
rid = lambda m: 'T11_V' if m >= 202101 else 'T10_F'
cov = lambda m: F.COVID_ERA[0] <= m <= F.COVID_ERA[1]
IU3 = ['IU', 'IUm', 'IUn']
TRUE_SD = {'Party (Rep - Dem)': 7.85, 'Income (top - bottom third)': 2.30, 'Education (college - HS or less)': 3.63,
           'Stocks (owner - none)': 2.09, 'Home (owner - renter)': 0.0, 'Age (55+ - 18-34)': 0.0, 'Sex (men - women)': 1.72}


def frame():
    B = base(600)
    B = B[B.month.isin(MONTHS)].copy()
    B['diet'] = [ext28.diet_of(p) for p in B.pid]
    ok = F._fill(B, lambda m, d: IU3[d], rid, 'U')
    ok &= F._fill(B, lambda m, d: (['CV', 'CVm', 'Xn'] if cov(m) else ['X', 'Xm', 'Xn'])[d], rid, 'F')
    ok &= F._fill(B, lambda m, d: ['XPG', 'XPGm', 'XPGn'][d], rid, 'pd')
    for k in range(3):   # one diet for everyone: own answer where the diet matches, the IUa{k} answer otherwise
        ok &= F._fill(B, lambda m, d, k=k: IU3[k] if d == k else f'IUa{k}', rid, f'A{k}')
    ok &= F._fill(B, lambda m, d: (['IUc', 'IUcm', 'IUn'] if 202003 <= m <= 202212 else IU3)[d], rid, 'C')
    P = B[ok].copy().reset_index(drop=True)
    if os.environ.get('T56_SNAP', '0') == '1':
        import t56_frames
        P = t56_frames.override(P, 'T56_A', arms=IU3, tags=('U',))   # T56: one-sided COVID lines in 2020-02/03 re-asked
    P['w'] = P.w / P.groupby('month').w.transform('sum')
    print('personas', len(B), 'with every answer set', len(P))
    return P


def score(name, P, H, dscol, s, drs, OG, off):
    r = {'variant': name}
    e = np.array([s[m] - off[m] for m in MONTHS])
    r['MAE'] = float(np.abs(e).mean())
    r['RMSE'] = float(np.sqrt((e ** 2).mean()))
    r['RMSE pandemic'] = float(np.sqrt(np.mean([(s[m] - off[m]) ** 2 for m in COVID])))
    if drs is not None:
        t = E.items(s, drs)
        t = t[t.month.isin(MONTHS) & (t.item != 'ICS') & (t.z > 1)]
        r['five-question direction'] = f'{int(t.hit.sum())}/{len(t)}'
    if P is not None:
        disp = []
        for c, (s1, s2) in CON.items():
            c1, c2 = group_change(P, H, dscol, s1), group_change(P, H, dscol, s2)
            g = pd.Series({m: c1.get(m, np.nan) - c2.get(m, np.nan) for m in MONTHS})
            ok = g.notna() & OG[c].notna()
            if c in ('Party (Rep - Dem)', 'Stocks (owner - none)'):
                r[f'{c.split()[0]} gap corr'] = float(np.corrcoef(g[ok], OG[c][ok])[0, 1]) if g[ok].std() > 1e-9 else np.nan
            disp.append(abs(g.drop([202003, 202004, 202005]).std() - TRUE_SD[c]))
        r['dispersion error'] = float(np.mean(disp))
    return r


def main():
    off = official_target()
    OG = official_monthly_gaps()
    P = frame()
    rows = []
    H = design.H(P, 'U', 'U')
    s, drs = MON.method_series(P, H, 'ds_U')
    rows.append(score('Full ConsumerSim', P, H, 'ds_U', s, drs, OG, off))
    # consensus among similar people -> national consensus (the registered first version of this design)
    Hn = design.H(P, 'U', 'U', k=float('inf'), g=1.5)
    sn, dn = MON.method_series(P, Hn, 'ds_U')
    rows.append(score('consensus gain national instead of within groups', P, Hn, 'ds_U', sn, dn, OG, off))
    # identity and exposure layer removed = the frozen design on the same personas
    Hf = H_of(P, 'F', 'pd', NORMS_FROZEN)
    sf, df_ = MON.method_series(P, Hf, 'ds_F')
    rows.append(score('w/o identity and exposure layer', P, Hf, 'ds_F', sf, df_, OG, off))
    # persona heterogeneity removed: every persona gets the month's mean update share and score change
    R = P.copy()
    Hr = {}
    for q in D.Q:
        R[f'ds_U_{q}'] = pd.Series(P.w.values * P[f'ds_U_{q}'].values).groupby(P.month.values).transform('sum').values
        Hr[q] = pd.Series(P.w.values * H[q]).groupby(P.month.values).transform('sum').values
    sr, dr = MON.method_series(R, Hr, 'ds_U')
    rows.append(score('w/o persona heterogeneity', R, Hr, 'ds_U', sr, dr, OG, off))
    # one information diet for everyone (best of the three on MAE)
    best = None
    for k, lab in enumerate(['figures + news', 'figures only', 'news only']):
        Hk = design.H(P, f'A{k}', f'A{k}')
        sk, dk = MON.method_series(P, Hk, f'ds_A{k}')
        r = score(f'w/o information-diet mix ({lab} for everyone)', P, Hk, f'ds_A{k}', sk, dk, OG, off)
        best = r if best is None or r['MAE'] < best['MAE'] else best
    best['variant'] = 'w/o information-diet mix (best single diet: ' + best['variant'].split('(')[1]
    rows.append(best)
    # no COVID case panel
    Hc = design.H(P, 'C', 'C')
    sc, dc = MON.method_series(P, Hc, 'ds_C')
    rows.append(score('w/o COVID case panel', P, Hc, 'ds_C', sc, dc, OG, off))
    # no consensus gain
    Hg = {}
    for q in D.Q:
        sz = P[f'size_U_{q}'].values
        Hg[q] = np.minimum(0.95, np.where(sz == 2, 0.4, np.where(sz == 1, 0.03, 0.0)) * 0.75)
    sg, dg = MON.method_series(P, Hg, 'ds_U')
    rows.append(score('w/o consensus gain', P, Hg, 'ds_U', sg, dg, OG, off))
    # no graded update size: one uniform share chosen on the training months by RMSE
    tr = [m for m in MONTHS if m < 202101]
    cand = []
    P2 = P.copy()
    for q in D.Q:
        P2[f'size_U_{q}'] = 2
    G = {q: v / 0.3 for q, v in design.H(P2, 'U', 'U').items()}   # the full design's consensus-gain factor, kept when the graded size is replaced
    for u in (0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2):
        Hu = {q: np.minimum(0.95, u * G[q]) for q in D.Q}
        su, _ = MON.method_series(P, Hu, 'ds_U')
        cand.append((np.sqrt(np.mean([(su[m] - off[m]) ** 2 for m in tr])), u))
    u = min(cand)[1]
    Hu = {q: np.minimum(0.95, u * G[q]) for q in D.Q}
    su, du = MON.method_series(P, Hu, 'ds_U')
    rows.append(score(f'w/o graded update size (uniform share {u})', P, Hu, 'ds_U', su, du, OG, off))
    # no information environment: nobody changes, i.e. persistence
    rows.append(score('w/o information environment', None, None, None, {m: off[D.ym_add(m, -1)] for m in MONTHS}, None, OG, off))
    # no contrast elicitation: direct re-answer (development arm E3e, same assimilated start)
    m = pd.concat([pd.read_csv(D.ROOT / f'runs/{r}/monthly.csv') for r in ('T10_F', 'T11_V')])
    e = m[m.arm == 'E3e'].set_index('month').ICS.to_dict()
    if all(x in e for x in MONTHS):
        rows.append(score('w/o undated contrast elicitation', None, None, None, e, None, OG, off))
    T = pd.DataFrame(rows).set_index('variant')
    T.to_csv(OUT / 'ablation.csv')
    (OUT / 'table4.md').write_text('# T49 消融（Table 4，统一暴露层，41 个小测月份，每月约 600 人）\n\n' + T.round(3).to_markdown() + '\n')
    print(T.round(3).to_string())


if __name__ == '__main__':
    main()
