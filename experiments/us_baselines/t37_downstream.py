"""T37: downstream predictive value (Fig. 3 analogue), following the consumer-sentiment forecasting literature.

Step 1 (replication, official ICS, 1979-2026): Bram & Ludvigson (1998) / Carroll-Fuhrer-Wilcox (1994) design at monthly
frequency. Target y_{t+h} = (1200/h) [ln A_{t+h} - ln A_t]. Baseline: constant + lags 0..5 of own growth, real disposable
income growth, real S&P 500 growth, change in the 3-month T-bill (+ change in the 30-year mortgage rate for housing, change
in unemployment for vehicles and durables). Augmented: + ICS_t ... ICS_{t-5}. In-sample: change in adjusted R2 and the HAC
Wald test of the ICS block; out-of-sample: expanding window, origins 2000-01 onward, RMSE change and Clark-West test.

Step 2 (ConsumerSim, origins 2014-07 to 2026-09):
 (a) nowcast plug-in: the augmented model, re-estimated each origin on realised data only, is fed ICS_t from
     (i) last month's official ICS (what is known mid-month), (ii) ConsumerSim's nowcast, (iii) the official ICS (oracle).
     Share of the oracle gain recovered = (RMSE_i - RMSE_ii) / (RMSE_i - RMSE_iii).
 (b) paper-style (Huang et al. B2 vs B3): add the simulated index (or the simulated monthly change) to the augmented
     model with official ICS, 2014-07 to 2026-09 only.
Run with CS_TEST=1.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

assert os.environ.get('CS_TEST') == '1'
import data as D
from score import official_target

FD = D.ROOT / 'data/ext/fred_dl'
OUT = D.ROOT / 'receipts' / 'T37'
OUT.mkdir(parents=True, exist_ok=True)
P = 6
HS = [1, 3, 6, 12]
SPEC = os.environ.get('T37_SPEC', 'howrey')   # 'howrey': ICS_t and dICS_t (Howrey 2001, monthly); 'bl': ICS_t..ICS_t-5 (Bram-Ludvigson)
OUTCOMES = {'Housing starts': 'HOUST', 'New home sales': 'HSN1F', 'Building permits': 'PERMIT', 'Vehicle sales': 'TOTALSA',
            'Real durable goods': 'DDURRA3M086SBEA', 'Real PCE (total)': 'DPCERA3M086SBEA'}
GROUP = {'Housing starts': 'Housing', 'New home sales': 'Housing', 'Building permits': 'Housing', 'Vehicle sales': 'Car purchase',
         'Real durable goods': 'Durable goods', 'Real PCE (total)': 'Total consumption'}


def monthly(s, how='mean'):
    x = pd.read_csv(FD / f'{s}.csv', index_col=0, parse_dates=True).iloc[:, 0]
    x = pd.to_numeric(x, errors='coerce').dropna()
    x = x.resample('MS').mean() if how == 'mean' else x.resample('MS').last()
    x.index = [d.year * 100 + d.month for d in x.index]
    return x


def data():
    off = official_target()
    ics = monthly('UMCSENT')
    for m, v in off.items():
        if m >= 197801:
            ics.loc[m] = v
    ics = ics.sort_index()
    import t31_baselines as B
    spx = B.spx().resample('MS').mean()
    spx.index = [d.year * 100 + d.month for d in spx.index]
    df = pd.DataFrame({'ics': ics, 'inc': monthly('DSPIC96'), 'spx': spx, 'pcepi': monthly('PCEPI'), 'tb': monthly('TB3MS'),
                       'mort': monthly('MORTGAGE30US'), 'ur': monthly('UNRATE')})
    for k, s in OUTCOMES.items():
        df[k] = monthly(s)
    df = df[(df.index >= 197801) & (df.index <= 202609)]
    df['d_inc'] = np.log(df.inc).diff() * 1200
    df['d_spx'] = np.log(df.spx / df.pcepi).diff() * 1200
    df['d_tb'] = df.tb.diff()
    df['d_mort'] = df.mort.diff()
    if os.environ.get('T56_UR', '0') == '1':   # T56 check (audit #8): October 2025 unemployment was never collected; interpolation tested, not adopted
        df['ur'] = df.ur.interpolate(limit_area='inside')
    df['d_ur'] = df.ur.diff()
    return df


def design(df, k, h, extra_ics=True):
    a = np.log(df[k])
    y = (a.shift(-h) - a) * 1200 / h
    X = {'const': 1.0}
    ctrl = ['d_inc', 'd_spx', 'd_tb'] + (['d_mort'] if GROUP[k] == 'Housing' else ['d_ur'] if GROUP[k] in ('Car purchase', 'Durable goods') else [])
    da = a.diff() * 1200
    for j in range(P):
        X[f'own{j}'] = da.shift(j)
        for c in ctrl:
            X[f'{c}{j}'] = df[c].shift(j)
    X = pd.DataFrame(X, index=df.index)
    if SPEC == 'bl':
        I = pd.DataFrame({f'ics{j}': df.ics.shift(j) for j in range(P)}, index=df.index)
    else:
        I = pd.DataFrame({'ics0': df.ics, 'ics1': df.ics.shift(1)}, index=df.index)   # same span as ICS_t and dICS_t
    return y, X, I


def fit(y, X):
    ok = y.notna() & X.notna().all(axis=1)
    return sm.OLS(y[ok], X[ok]).fit(), ok


def clark_west(y, f0, f1):
    """one-sided test that the larger model (f1) beats the nested model (f0)."""
    e0, e1 = y - f0, y - f1
    d = e0 ** 2 - (e1 ** 2 - (f0 - f1) ** 2)
    m = sm.OLS(d.values, np.ones(len(d))).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    return float(m.tvalues[0]), float(1 - __import__('scipy').stats.norm.cdf(m.tvalues[0]))


def step1(df):
    rows = []
    for k in OUTCOMES:
        for h in HS:
            y, X, I = design(df, k, h)
            XA = pd.concat([X, I], axis=1)
            sample = (df.index >= 197901)
            b, ok0 = fit(y[sample], X[sample])
            a, ok1 = fit(y[sample], XA[sample])
            ok = ok1
            a_hac = sm.OLS(y[sample][ok], XA[sample][ok]).fit(cov_type='HAC', cov_kwds={'maxlags': max(h, 6)})
            wald = a_hac.wald_test(' , '.join(f'{c} = 0' for c in I.columns), scalar=True)
            # out-of-sample, expanding window
            f0, f1, yy = [], [], []
            origins = [m for m in df.index if 200001 <= m <= D.ym_add(202609, -h)]
            for t in origins:
                tr = (df.index >= 197901) & (df.index <= D.ym_add(t, -h))
                m0, _ = fit(y[tr], X[tr])
                m1, _ = fit(y[tr], XA[tr])
                x0, x1 = X.loc[[t]], XA.loc[[t]]
                if x1.isna().any(axis=1).iloc[0] or np.isnan(y[t]):
                    continue
                f0.append(float(m0.predict(x0).iloc[0]))
                f1.append(float(m1.predict(x1).iloc[0]))
                yy.append(float(y[t]))
            f0, f1, yy = map(np.array, (f0, f1, yy))
            r0, r1 = np.sqrt(np.mean((yy - f0) ** 2)), np.sqrt(np.mean((yy - f1) ** 2))
            cw, p = clark_west(pd.Series(yy), pd.Series(f0), pd.Series(f1))
            # excluding 2020
            o = np.array(origins[:len(yy)])
            keep = ~((o >= 202001) & (o <= 202012))
            r0x, r1x = np.sqrt(np.mean((yy[keep] - f0[keep]) ** 2)), np.sqrt(np.mean((yy[keep] - f1[keep]) ** 2))
            rows.append({'outcome': k, 'group': GROUP[k], 'h': h, 'n': int(ok.sum()), 'adjR2 base': b.rsquared_adj, 'adjR2 +ICS': a.rsquared_adj,
                         'dAdjR2': a.rsquared_adj - b.rsquared_adj, 'Wald p (ICS block)': float(wald.pvalue),
                         'OOS RMSE base': r0, 'OOS RMSE +ICS': r1, 'OOS gain %': (r0 - r1) / r0 * 100, 'Clark-West p': p,
                         'OOS gain % excl. 2020': (r0x - r1x) / r0x * 100})
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def step2(df, sim):
    rows = []
    origins_all = [m for m in D.ALL_SCORED + D.VALID + D.TEST]
    for k in OUTCOMES:
        for h in HS:
            y, X, I = design(df, k, h)
            XA = pd.concat([X, I], axis=1)
            res = {'persist': [], 'sim': [], 'oracle': [], 'y': [], 't': []}
            for t in origins_all:
                if t > D.ym_add(202609, -h) or np.isnan(y.get(t, np.nan)):
                    continue
                tr = (df.index >= 197901) & (df.index <= D.ym_add(t, -h))
                m1, _ = fit(y[tr], XA[tr])
                x = XA.loc[[t]].copy()
                if x.isna().any(axis=1).iloc[0]:
                    continue
                for lab, v in (('oracle', df.ics[t]), ('persist', df.ics[D.ym_add(t, -1)]), ('sim', sim[t])):
                    x['ics0'] = v
                    res[lab].append(float(m1.predict(x).iloc[0]))
                res['y'].append(float(y[t]))
                res['t'].append(t)
            R = pd.DataFrame(res)
            rm = {c: float(np.sqrt(((R[c] - R.y) ** 2).mean())) for c in ('persist', 'sim', 'oracle')}
            gain_or = rm['persist'] - rm['oracle']
            # (b) paper-style: add the simulated index / simulated change to the augmented model, 2014-07..2026-09 in-sample
            s = df.index.isin(origins_all)
            XB = XA.copy()
            XB['sim_gap'] = sim.reindex(df.index) - df.ics
            a, ok = fit(y[s], XA[s])
            b, _ = fit(y[s], XB[s])
            bh = sm.OLS(y[s][ok], XB[s][ok]).fit(cov_type='HAC', cov_kwds={'maxlags': max(h, 6)})
            rows.append({'outcome': k, 'group': GROUP[k], 'h': h, 'n': len(R), 'RMSE ICS_t-1 (known mid-month)': rm['persist'],
                         'RMSE ConsumerSim nowcast': rm['sim'], 'RMSE official ICS_t (oracle)': rm['oracle'],
                         'sim gain % vs t-1': (rm['persist'] - rm['sim']) / rm['persist'] * 100,
                         'oracle gain % vs t-1': gain_or / rm['persist'] * 100,
                         'share of oracle gain': (rm['persist'] - rm['sim']) / gain_or if abs(gain_or) > 1e-9 else np.nan,
                         'paper-style dAdjR2 (+ sim - official)': b.rsquared_adj - a.rsquared_adj, 'p (sim gap)': float(bh.pvalues['sim_gap'])})
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def step2b(df, sim, start=201801):
    """Out-of-sample version of the paper-style test. At origin t: fit the augmented model (official ICS) on 1979..t-h; regress
    its residuals on the simulated gap (sim - official) over 2014-07..t-h (one slope + intercept); forecast. Compare with
    the augmented model plus the same residual intercept only (so the gain is the gap's slope, not a bias correction)."""
    rows = []
    gap = sim.reindex(df.index) - df.ics
    for k in OUTCOMES:
        for h in HS:
            y, X, I = design(df, k, h)
            XA = pd.concat([X, I], axis=1)
            fa, fb, yy, tt = [], [], [], []
            for t in [m for m in D.ALL_SCORED + D.VALID + D.TEST if m >= start and m <= D.ym_add(202609, -h)]:
                tr = (df.index >= 197901) & (df.index <= D.ym_add(t, -h))
                m1, ok = fit(y[tr], XA[tr])
                rs = [m for m in df.index[tr] if m >= 201407]
                res = (y.loc[rs] - m1.predict(XA.loc[rs])).dropna()
                g = gap.loc[res.index]
                if XA.loc[[t]].isna().any(axis=1).iloc[0] or np.isnan(y[t]):
                    continue
                base = float(m1.predict(XA.loc[[t]]).iloc[0])
                b = np.polyfit(g.values, res.values, 1)
                fa.append(base + res.mean())
                fb.append(base + b[1] + b[0] * gap[t])
                yy.append(float(y[t]))
                tt.append(t)
            fa, fb, yy = map(np.array, (fa, fb, yy))
            ra, rb = np.sqrt(np.mean((yy - fa) ** 2)), np.sqrt(np.mean((yy - fb) ** 2))
            cw, p = clark_west(pd.Series(yy), pd.Series(fa), pd.Series(fb))
            tt = np.array(tt)
            keep = ~((tt >= 202001) & (tt <= 202012))
            rax, rbx = np.sqrt(np.mean((yy[keep] - fa[keep]) ** 2)), np.sqrt(np.mean((yy[keep] - fb[keep]) ** 2))
            rows.append({'outcome': k, 'group': GROUP[k], 'h': h, 'n': len(yy), 'OOS RMSE official model': ra, 'OOS RMSE + sim gap': rb,
                         'OOS gain %': (ra - rb) / ra * 100, 'Clark-West p': p, 'OOS gain % excl. 2020': (rax - rbx) / rax * 100})
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def insample(df, sim):
    """Huang et al. B2 vs B3: add the simulated index to the forecasting equation that already contains the official ICS_t and
    dICS_t; 2014-07 to 2026-09; HAC test of the added term, with and without 2020."""
    months = D.ALL_SCORED + D.VALID + D.TEST
    rows = []
    for k in OUTCOMES:
        for h in HS:
            y, X, I = design(df, k, h)
            XA = pd.concat([X, I], axis=1)
            XB = XA.copy()
            XB['sim'] = sim.reindex(df.index)
            for lab, excl in (('all', False), ('excl. 2020', True)):
                s = df.index.isin(months) & ~(excl & (df.index >= 202001) & (df.index <= 202012))
                ok = s & y.notna() & XB.notna().all(axis=1)
                a = sm.OLS(y[ok], XA[ok]).fit()
                b = sm.OLS(y[ok], XB[ok]).fit(cov_type='HAC', cov_kwds={'maxlags': max(h, 6)})
                rows.append({'outcome': k, 'h': h, 'sample': lab, 'n': int(ok.sum()), 'dAdjR2': b.rsquared_adj - a.rsquared_adj,
                             'dR2': b.rsquared - a.rsquared, 'coef sim': b.params['sim'], 'p': b.pvalues['sim']})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'insample_sim_increment.csv', index=False)
    for lab in ('all', 'excl. 2020'):
        r = R[R['sample'] == lab]
        print(lab, 'significant at 5%:', int((r.p < 0.05).sum()), '/', len(r), ' positive dR2:', int((r.dR2 > 0).sum()))
    return R


def main():
    df = data()
    f1 = OUT / f'step1_official_{SPEC}.csv'
    if f1.exists():
        S1 = pd.read_csv(f1)
    else:
        S1 = step1(df)
        S1.to_csv(f1, index=False)
    import design
    sim = pd.read_csv(design.SERIES_1800, index_col=0).sim
    S2 = step2(df, sim)
    S2.to_csv(OUT / f'step2_sim_{SPEC}.csv', index=False)
    S3 = step2b(df, sim)
    S3.to_csv(OUT / f'step2b_sim_oos_{SPEC}.csv', index=False)
    insample(df, sim)
    lines = ['# T37 下游预测力', '', '## 第一步：官方 ICS 的增量预测力（1979–2026，样本外从 2000 年起）', '', S1.round(3).to_markdown(index=False), '',
             '## 第二步：ConsumerSim（2014-07 至 2026-09）', '', S2.round(3).to_markdown(index=False), '',
             '## 第二步 b：论文式“加入模拟与官方之差”的样本外版本（起点 2018-01）', '', S3.round(3).to_markdown(index=False), '']
    (OUT / f'downstream_{SPEC}.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
