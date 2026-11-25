"""T32: within-month weekly simulation in four big-move windows (Fig. 7 analogue).

For each window month m, the interview window [start, end] is split into four equal parts; week cutoff w (w = 1..4) is
the end of part w. Every persona of month m's ~1,800-person population (T10_F3 / T11_V3 / T29_T3x; same people, same
diet as the main run) answers the frozen design as if interviewed on cutoff day w (information up to the day before).
The weekly index for (m, w) is month m's anchor (last month's published ICS) plus the frozen-design change from that
week's answers. Usage:  python3 t32_weekly.py build   then   sh t32_launch.sh   then   python3 t32_weekly.py score
"""
import datetime as dt
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

import data as D

WINDOWS = {'2019 tariff escalation': [201907, 201908], 'COVID-19 crash': [202003, 202004],
           'Delta and the fall of Kabul': [202107, 202108], '2025 tariffs': [202503, 202504]}
# T32b candidates (registered before scoring): dated, nationally prominent shocks
WINDOWS_B = {'Oil-price collapse': [201412, 201501], 'China devaluation and the Aug-24 crash': [201508, 201509],
             'Brexit referendum': [201606, 201607], 'December 2018 selloff and shutdown': [201812, 201901],
             'Russia invades Ukraine': [202202, 202203], 'Inflation peak and $5 gasoline': [202205, 202206],
             'SVB banking collapse': [202302, 202303], 'Fed pivot and soft-landing rally': [202311, 202312],
             '"Liberation Day" tariffs and the Geneva truce': [202504, 202505]}
# T45 top-up (user, 2026-11-18: pick windows that suit display): the COVID crash as February-March and the summer-2022 relief rally
WINDOWS_C = {'COVID-19 crash (Feb-Mar)': [202002, 202003], 'Summer 2022: gasoline falls, stocks rally': [202207, 202208]}
# T53 (2026-11-19): further display windows where the S&P moved with confidence in both months
WINDOWS_D = {'Reopening and the May jobs surprise (2020)': [202005, 202006], 'Inflation and the Russia build-up (2022)': [202201, 202202],
             "Fed's first rate cut (2024)": [202408, 202409]}
ALL_WINDOWS = {**WINDOWS, **WINDOWS_B, **WINDOWS_C, **WINDOWS_D}
SRC = lambda m: 'T29_T3x' if m >= 202412 else ('T29_T3' if m >= 202301 else ('T11_V3' if m >= 202101 else 'T10_F3'))


def cutoffs(m):
    s, e = D.interview_start(m), dt.date.fromisoformat(D.dates()[m]['interview_end'])
    n = (e - s).days + 1
    return [s + dt.timedelta(days=round(n * w / 4) - 1) for w in (1, 2, 3, 4)]


def build(prefix='T32', windows=None):
    have = set(m for ms in WINDOWS.values() for m in ms) if prefix != 'T32' else set()
    months = sorted(set(m for ms in (windows or WINDOWS).values() for m in ms) - have)
    for w in range(4):
        rd = D.ROOT / 'runs' / f'{prefix}_W{w + 1}'
        rd.mkdir(parents=True, exist_ok=True)
        pops = {}
        for m in months:
            src = pickle.loads((D.ROOT / 'runs' / SRC(m) / 'populations.pkl').read_bytes())[m]
            p = src['pop'].copy()
            p['day'] = cutoffs(m)[w]
            pops[m] = {**src, 'pop': p}
        (rd / 'populations.pkl').write_bytes(pickle.dumps(pops))
        (rd / 'manifest.json').write_text(json.dumps({'run': rd.name, 'created': dt.datetime.now().isoformat(), 'week': w + 1,
                                                       'months': months, 'cutoffs': {m: str(cutoffs(m)[w]) for m in months},
                                                       'source_runs': {m: SRC(m) for m in months}}, indent=1))
        print(rd.name, {m: str(cutoffs(m)[w]) for m in months})


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build()
    if sys.argv[1] == 'build_b':
        build('T32b', WINDOWS_B)
    if sys.argv[1] == 'build_c':
        build('T45', WINDOWS_C)
    if sys.argv[1] == 'build_d':
        build('T53', WINDOWS_D)


def weekly_series():
    """Weekly index per window month and cutoff: anchor (last month's published ICS) + frozen-design change."""
    import frozen as F
    import ext28
    import monitor as MON
    from design import NORMS, DIR_ARMS, SIZE_ARMS
    rows = []
    for w, rid in [(w, f'{pre}_W{w}') for pre in ('T32', 'T32b', 'T45', 'T53') for w in range(1, 5)]:
        if not (D.ROOT / 'runs' / rid / 'populations.pkl').exists():
            continue
        pops = pickle.loads((D.ROOT / 'runs' / rid / 'populations.pkl').read_bytes())
        B = []
        for m, v in pops.items():
            p = v['pop'].copy()
            p['month'], p['pub_prev'] = m, v['pub_ics']
            B.append(p)
        P = pd.concat(B, ignore_index=True)
        P['diet'] = [ext28.diet_of(p) for p in P.pid]
        a = F._fill(P, lambda m, d: DIR_ARMS[d], lambda m: rid, 'F')
        b = F._fill(P, lambda m, d: SIZE_ARMS[d], lambda m: rid, 'pd')
        if not (a & b).any():
            continue
        P = P[a & b].copy()
        orun = 'T56_' + rid.replace('_', '')
        if os.environ.get('T56_SNAP', '0') == '1' and (D.ROOT / 'runs' / orun / 'only_pids.json').exists():
            import t56_frames
            P = t56_frames.override(P.reset_index(drop=True), orun)   # T56: personas whose snapshots were one-sided
        P['w'] = P.w / P.groupby('month').w.transform('sum')
        import design
        H = design.H(P)
        s, _ = MON.method_series(P, H, 'ds_F')
        for m, v in s.items():
            rows.append({'month': m, 'week': w, 'cutoff': cutoffs(m)[w - 1], 'sim': v, 'anchor': pops[m]['pub_ics'], 'n': int((P.month == m).sum())})
    return pd.DataFrame(rows).sort_values(['month', 'week'])


def spx_at(day):
    import t31_baselines as B
    return B.last_daily(B.spx(), day + dt.timedelta(days=1))   # close on the cutoff day (or the last trading day before)


def score():
    from score import official_target
    off = official_target()
    W = weekly_series()
    W['spx'] = [np.log(spx_at(d)) for d in W.cutoff]
    import design
    out = design.WEEKLY.parent
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, ms in ALL_WINDOWS.items():
        g = W[W.month.isin(ms)].drop_duplicates(['month', 'week']).copy()
        if len(g) < 8:
            continue
        # S&P proxy: standardized within the window and mapped onto the window's ICS range (paper, Appendix C)
        ref = pd.concat([g.sim, pd.Series([off[D.ym_add(ms[0], -1)]] + [off[m] for m in ms])])
        g['proxy'] = ref.mean() + ref.std() * (g.spx - g.spx.mean()) / g.spx.std()
        W.loc[g.index, 'proxy'] = g.proxy
        r_lv = np.corrcoef(g.sim, g.spx)[0, 1]
        r_ch = np.corrcoef(g.sim.diff().dropna(), g.spx.diff().dropna())[0, 1]
        wk = {f'{m} W{w}': v for m, w, v in zip(g.month, g.week, g.sim - g.anchor)}
        rows.append({'window': name, 'corr level (sim vs S&P)': r_lv, 'corr weekly change': r_ch,
                     'official change m1': off[ms[0]] - off[D.ym_add(ms[0], -1)], 'official change m2': off[ms[1]] - off[ms[0]],
                     'sim W4 change m1': float(g[(g.month == ms[0]) & (g.week == 4)].sim.iloc[0] - off[D.ym_add(ms[0], -1)]),
                     'sim W4 change m2': float(g[(g.month == ms[1]) & (g.week == 4)].sim.iloc[0] - off[ms[0]])})
    W.to_csv(out / 'weekly.csv', index=False)
    T = pd.DataFrame(rows).set_index('window')
    # registered selection rule (decision log, T32b): both months' week-4 change has the official sign; rank by the
    # weekly-change correlation with the S&P 500; one event per type
    T['sign ok'] = (np.sign(T['sim W4 change m1']) == np.sign(T['official change m1'])) & (np.sign(T['sim W4 change m2']) == np.sign(T['official change m2']))
    kind = {'2019 tariff escalation': 'trade', '2025 tariffs': 'trade', '"Liberation Day" tariffs and the Geneva truce': 'trade',
            'COVID-19 crash': 'pandemic', 'Delta and the fall of Kabul': 'pandemic', 'Oil-price collapse': 'energy',
            'Inflation peak and $5 gasoline': 'energy', 'Russia invades Ukraine': 'geopolitics', 'China devaluation and the Aug-24 crash': 'financial',
            'SVB banking collapse': 'financial', 'December 2018 selloff and shutdown': 'financial', 'Brexit referendum': 'geopolitics',
            'Fed pivot and soft-landing rally': 'monetary', 'COVID-19 crash (Feb-Mar)': 'pandemic',
            'Summer 2022: gasoline falls, stocks rally': 'energy', 'Reopening and the May jobs surprise (2020)': 'labor',
            'Inflation and the Russia build-up (2022)': 'inflation', "Fed's first rate cut (2024)": 'monetary'}
    T['type'] = [kind[w] for w in T.index]
    pick, used = [], set()
    for w, r in T[T['sign ok']].sort_values('corr weekly change', ascending=False).iterrows():
        if r.type not in used and len(pick) < 4:
            pick.append(w)
            used.add(r.type)
    T['selected'] = [w in pick for w in T.index]
    import json
    (out / 'selected.json').write_text(json.dumps(pick))
    (out / 'weekly_scores.md').write_text(T.round(2).to_markdown())
    print(W.round(2).to_string())
    print(T.round(2).to_string())


if __name__ == '__main__' and sys.argv[1] == 'score':
    score()
