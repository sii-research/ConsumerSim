"""T50: subgroup results of the adopted design (IU, ~1,800 personas/month, 147 months) against the Michigan microdata.

  fig4    - group trajectories around four salient shocks (two in training, two in the test period), four contrasts
  map     - signal type x contrast map of gap changes over theory-defined event windows (Table 3 / Fig. 5): official vs
            ConsumerSim, most sensitive group per signal type, agreement
  fidelity- monthly gap-change correlation with the official series vs the official noise ceiling, adopted vs frozen design
  sources - where the simulated differences come from: LLM answers shuffled within month x information diet
Run with CS_TEST=1 after t49_frames.py.
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

assert os.environ.get('CS_TEST') == '1'
import data as D
import frozen as F
import monitor as MON
from design import NORMS, FRAMES_1800, NORMS_FROZEN
from t36_figs import style, INK, MUTED, SUB, SURF, BLUE, ORANGE, GRID
from t47_micro_scan import CON

OUT = D.ROOT / 'receipts' / 'T50'
OUT.mkdir(parents=True, exist_ok=True)
FIG = D.ROOT / 'receipts' / 'T36'
ICS0 = 500 / 6.7558 + 2
MS = D.ALL_SCORED + D.VALID + D.TEST
GROUPS = {'Party (Rep - Dem)': ('Republicans', 'Democrats'), 'Income (top - bottom third)': ('Top income third', 'Bottom income third'),
          'Education (college - HS or less)': ('College degree', 'High school or less'), 'Stocks (owner - none)': ('Stockholders', 'No stocks'),
          'Home (owner - renter)': ('Homeowners', 'Renters'), 'Age (55+ - 18-34)': ('Age 55+', 'Age 18-34'), 'Sex (men - women)': ('Men', 'Women')}
SHORT = {'Party (Rep - Dem)': 'Party', 'Income (top - bottom third)': 'Income', 'Education (college - HS or less)': 'Education',
         'Stocks (owner - none)': 'Stock ownership', 'Home (owner - renter)': 'Homeownership', 'Age (55+ - 18-34)': 'Age', 'Sex (men - women)': 'Sex'}
MAP_CONTRASTS = ['Party (Rep - Dem)', 'Stocks (owner - none)', 'Education (college - HS or less)', 'Income (top - bottom third)',
                 'Home (owner - renter)', 'Age (55+ - 18-34)']
# signal types and event windows, defined from the literature and the event calendar (T47), not from model output
SIGNALS = {
    'Presidential election': [('2020 election', 202010, 202101), ('2024 election', 202410, 202501)],
    'Trade policy': [('2019 tariff escalation', 201907, 201908), ('2025 tariffs', 202501, 202504)],
    'Stock-market crash': [('Q4 2018 selloff', 201809, 201812), ('COVID crash', 202002, 202003), ('2022 bear market', 202112, 202210)],
    'Stock-market rally': [('2023 disinflation rally', 202310, 202401)],
    'Labor-market shock': [('COVID job losses', 202003, 202004)],
    'Inflation surge': [('2021 inflation', 202103, 202111)],
    'Gasoline spike': [('2022 gasoline spike', 202202, 202206)],
    'Gasoline relief': [('2022 gasoline falls', 202206, 202212)],
    'Interest rates': [('Mortgage rates near 8%', 202308, 202310)],
    'Financial-sector stress': [('SVB collapse', 202302, 202303)],
    'Government shutdown': [('2019 shutdown', 201812, 201901), ('2025 shutdown', 202509, 202511)],
}
FIG4 = [('2020 election (Biden wins)', [202010, 202011, 202012, 202101], 'training'),
        ('COVID-19 shock', [202002, 202003, 202004, 202005], 'training'),
        ('2024 election (Trump wins)', [202410, 202411, 202412, 202501], 'test'),
        ('2025 tariffs and selloff', [202501, 202502, 202503, 202504], 'test')]
FIG4_CONTRASTS = ['Party (Rep - Dem)', 'Stocks (owner - none)', 'Education (college - HS or less)', 'Income (top - bottom third)']


def official_groups():
    G = pd.read_csv(D.ROOT / 'receipts/T47/group_series.csv')
    G['group'] = G.group.astype(str)
    return G.set_index(['contrast', 'group', 'month'])


def sim_groups(P, H, dscol='ds_F'):
    pc = MON.person_change(P, H, dscol)
    out = {}
    for c, (s1, s2) in CON.items():
        for k, sel in (('1', s1), ('2', s2)):
            m = sel(P).values
            x = pd.DataFrame({'m': P.month.values[m], 'c': pc[m] * P.w.values[m], 'w': P.w.values[m]}).groupby('m').sum()
            out[(c, k)] = (x.c / x.w)
    return out


def load(design='IU'):
    if design == 'IU':
        fr = pickle.load(open(FRAMES_1800, 'rb'))
        import design
        return fr['P'], design.H(fr['P'])
    fr = pickle.load(open('/tmp/t30_frames_1800.pkl', 'rb'))
    return fr['P'], F.H_frozen(fr['P'], 'F', norm=NORMS_FROZEN)[0]


def window(OG, SG, c, a, b):
    """official and simulated change of each group and of the gap between months a and b"""
    r = {}
    for k in ('1', '2'):
        try:
            r[f'off{k}'] = OG.loc[(c, k, b), 'ICS'] - OG.loc[(c, k, a), 'ICS']
            r[f'se{k}'] = np.hypot(OG.loc[(c, k, b), 'ICS_se'], OG.loc[(c, k, a), 'ICS_se'])
        except KeyError:
            r[f'off{k}'] = r[f'se{k}'] = np.nan
        r[f'sim{k}'] = float(SG[(c, k)].loc[[m for m in SG[(c, k)].index if a < m <= b]].sum())
    r['off_gap'], r['sim_gap'] = r['off1'] - r['off2'], r['sim1'] - r['sim2']
    r['se_gap'] = np.hypot(r['se1'], r['se2'])
    return r


FIG4_PANELS = [('COVID-19 shock (training)', 'Party (Rep - Dem)', [202001, 202002, 202003, 202004, 202005], 'A. Everyone falls, Republicans further'),
               ('COVID-19 shock (training)', 'Stocks (owner - none)', [202001, 202002, 202003, 202004, 202005], 'B. Stockholders fall first'),
               ('COVID-19 shock (training)', 'Education (college - HS or less)', [202001, 202002, 202003, 202004, 202005], 'C. College first, high school when jobs go'),
               ('2020 election (training)', 'Party (Rep - Dem)', [202010, 202011, 202012, 202101], 'D. 2020 election: Biden wins'),
               ('2024 election (test)', 'Party (Rep - Dem)', [202410, 202411, 202412, 202501], 'E. 2024 election: Trump wins'),
               ('2025 tariffs (test)', 'Party (Rep - Dem)', [202501, 202502, 202503, 202504], 'F. 2025 tariffs')]


def fig4():
    """Group trajectories for the splits the microdata show (Table 3). Dashed: ConsumerSim nowcast of each group, i.e. the
    previous month's official group index plus the simulated change, the same convention as the national series (Fig. 2)."""
    P, H = load()
    SG, OG = sim_groups(P, H), official_groups()
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6), facecolor=SURF)
    rows = []
    for ax, (ev, c, ms, title) in zip(axs.flat, FIG4_PANELS):
        style(ax)
        xs = np.arange(len(ms))
        if title[0] in 'DEF':   # party splits: the gap between Republicans and Democrats
            lv = np.array([OG.loc[(c, '1', m), 'ICS'] - OG.loc[(c, '2', m), 'ICS'] for m in ms])
            se = np.array([np.hypot(OG.loc[(c, '1', m), 'ICS_se'], OG.loc[(c, '2', m), 'ICS_se']) for m in ms])
            ch = np.array([SG[(c, '1')][m] - SG[(c, '2')][m] for m in ms[1:]])
            now = np.r_[lv[0], lv[:-1] + ch]
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.fill_between(xs, lv - se, lv + se, color=INK, alpha=0.08, lw=0)
            ax.plot(xs, lv, color=INK, lw=2.0, marker='o', ms=5, label='Official')
            ax.plot(xs, now, color=ORANGE, lw=1.8, ls=(0, (3, 2)), marker='o', ms=5, mfc=SURF, mew=1.5, label='ConsumerSim')
            if min(lv.min(), now.min()) < 0 < max(lv.max(), now.max()):
                ax.text(0.02, 0.97, 'Republicans more confident', transform=ax.transAxes, fontsize=7.5, color=SUB, va='top')
                ax.text(0.02, 0.03, 'Democrats more confident', transform=ax.transAxes, fontsize=7.5, color=SUB, va='bottom')
            rows.append({'panel': title, 'event': ev, 'group': 'Republicans - Democrats', 'official change': lv[-1] - lv[0],
                         'ConsumerSim change': float(ch.sum()), 'nowcast MAE': float(np.abs(now[1:] - lv[1:]).mean())})
            ax.set_xticks(xs, [f'{m // 100}-{m % 100:02d}' for m in ms], fontsize=8)
            ax.set_title(f'{title}\n{ev}', fontsize=10, color=INK, loc='left')
            ax.legend(frameon=False, fontsize=7.5, loc='center left' if title[0] != 'F' else 'lower right', handlelength=1.6)
            if ax is axs[1, 0]:
                ax.set_ylabel('Republicans minus Democrats (ICS points)', fontsize=8.5, color=SUB)
            continue
        for k, col in (('1', ORANGE), ('2', BLUE)):
            lv = np.array([ICS0 + OG.loc[(c, k, m), 'ICS'] for m in ms])
            se = np.array([OG.loc[(c, k, m), 'ICS_se'] for m in ms])
            ch = np.array([SG[(c, k)][m] for m in ms[1:]])
            now = np.r_[lv[0], lv[:-1] + ch]
            ax.fill_between(xs, lv - se, lv + se, color=col, alpha=0.12, lw=0)
            ax.plot(xs, lv, color=col, lw=2.0, marker='o', ms=5, label=f'{GROUPS[c][int(k) - 1]}, official')
            ax.plot(xs, now, color=col, lw=1.6, ls=(0, (3, 2)), marker='o', ms=5, mfc=SURF, mew=1.5, label=f'{GROUPS[c][int(k) - 1]}, ConsumerSim')
            rows.append({'panel': title, 'event': ev, 'group': GROUPS[c][int(k) - 1], 'official change': lv[-1] - lv[0],
                         'ConsumerSim change': float(ch.sum()), 'nowcast MAE': float(np.abs(now[1:] - lv[1:]).mean())})
        ax.set_xticks(xs, [f'{m // 100}-{m % 100:02d}' for m in ms], fontsize=8)
        ax.set_title(f'{title}\n{ev}', fontsize=10, color=INK, loc='left')
        ax.legend(frameon=False, fontsize=7.5, loc='best', handlelength=1.6)
        if ax is axs[0, 0]:
            ax.set_ylabel('Group ICS', fontsize=9, color=SUB)
    fig.suptitle('Fig. 4  Group trajectories around salient shocks: ConsumerSim against the official group indices (band: ±1 SE)',
                 x=0.01, ha='left', fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / 'fig4_groups.png', dpi=150, facecolor=SURF)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'fig4_changes.csv', index=False)
    print(R.round(1).to_string(index=False))


EVENTS = [(t, e, a, b) for t, evs in SIGNALS.items() for e, a, b in evs]


def cell_table(P, H):
    SG, OG = sim_groups(P, H), official_groups()
    rows = []
    for st, ev, a, b in EVENTS:
        for c in MAP_CONTRASTS:
            w = window(OG, SG, c, a, b)
            rows.append({'signal': st, 'event': ev, 'window': f'{a // 100}-{a % 100:02d} to {b // 100}-{b % 100:02d}', 'contrast': c, **w})
    T = pd.DataFrame(rows)
    T['z'] = T.off_gap / T.se_gap
    return T


def group_phrase(c, d1, d2):
    """T56 wording: a gap change under 2 points is 'no split'; a group that moved under 2 points is 'little changed'"""
    names = GROUPS[c]
    low = lambda n: n if n in ('Republicans', 'Democrats') else n.lower()
    if abs(d1 - d2) < 2:
        return 'No split'
    if abs(d1) < 2 or abs(d2) < 2:
        k = 0 if abs(d1) >= abs(d2) else 1
        d = (d1, d2)[k]
        return f"{names[k]} {'rose' if d > 0 else 'fell'}, {low(names[1 - k])} little changed"
    if np.sign(d1) != np.sign(d2):
        return f"{names[0] if d1 > 0 else names[1]} rose, {low(names[1] if d1 > 0 else names[0])} fell"
    k = 0 if abs(d1) > abs(d2) else 1
    return f"{names[k]} {'rose' if d1 > 0 else 'fell'} more"


def table3_fig5():
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    P, H = load()
    T = cell_table(P, H)
    T.to_csv(OUT / 'event_cells.csv', index=False)
    rows = []
    for (st, ev), g in T.groupby(['signal', 'event'], sort=False):
        gg = g.dropna(subset=['z'])
        sig = gg[gg.z.abs() > 2]
        al = lambda a, b: float((np.sign(a) == np.sign(b)).mean())
        r = {'Signal type': st, 'Event (window)': f"{ev} ({g.window.iloc[0]})", 'Alignment, official': al(gg.off1, gg.off2),
             'Alignment, ConsumerSim': al(g.sim1, g.sim2)}
        if len(sig):
            o = sig.loc[sig.z.abs().idxmax()]
            sm = g.set_index('contrast').loc[o.contrast]
            r['Most sensitive split, official'] = f"{SHORT[o.contrast]}: {group_phrase(o.contrast, o.off1, o.off2)} (gap {o.off_gap:+.1f}, z {o.z:+.1f})"
            r['Same split in ConsumerSim'] = f"{group_phrase(o.contrast, sm.sim1, sm.sim2)} (gap {sm.sim_gap:+.1f})"
            r['Direction agrees'] = bool(np.sign(sm.sim_gap) == np.sign(o.off_gap))
            r['Official significant splits'] = ', '.join(SHORT[c] for c in sig.contrast)
        else:
            r['Most sensitive split, official'] = 'no split beyond sampling noise'
            r['Same split in ConsumerSim'] = f"largest simulated gap {g.sim_gap.abs().max():.1f}"
            r['Direction agrees'] = np.nan
            r['Official significant splits'] = ''
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / 'table3.csv', index=False)
    sig = T.z.abs() > 2
    ok = T.off_gap.notna()
    ns = ~sig & ok
    summ = {'event x contrast cells': int(ok.sum()), 'official cells > 2 SE': int(sig.sum()),
            'ConsumerSim sign agreement on those': f"{int((np.sign(T.sim_gap[sig]) == np.sign(T.off_gap[sig])).sum())}/{int(sig.sum())}",
            'mean |gap| on significant cells, official / ConsumerSim': f"{T.off_gap[sig].abs().mean():.1f} / {T.sim_gap[sig].abs().mean():.1f}",
            'mean |gap| on non-significant cells, official / ConsumerSim': f"{T.off_gap[ns].abs().mean():.1f} / {T.sim_gap[ns].abs().mean():.1f}",
            'cell correlation (all cells)': float(T[ok].sim_gap.corr(T[ok].off_gap)),
            'cell correlation (significant cells)': float(T[sig].sim_gap.corr(T[sig].off_gap)),
            'events with a significant split': f"{int(S['Official significant splits'].astype(bool).sum())}/{len(S)}",
            'most sensitive split, direction agrees': f"{int((S['Direction agrees'] == True).sum())}/{int(S['Direction agrees'].notna().sum())}"}
    pd.Series(summ).to_csv(OUT / 'table3_summary.csv')
    pd.set_option('display.width', 250)
    print(S.drop(columns=['Alignment, official', 'Alignment, ConsumerSim']).to_string(index=False))
    print(pd.Series(summ).to_string())
    # Fig. 5: official vs ConsumerSim heat maps, events x contrasts, same colour scale
    ev = [f"{e}" for _, e, _, _ in EVENTS]
    M_off = T.pivot(index='event', columns='contrast', values='off_gap').loc[ev, MAP_CONTRASTS]
    M_sim = T.pivot(index='event', columns='contrast', values='sim_gap').loc[ev, MAP_CONTRASTS]
    Z = T.pivot(index='event', columns='contrast', values='z').loc[ev, MAP_CONTRASTS]
    lim = 30.0
    cmap = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f0efec', '#f4a3a3', '#c23434'])
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 8.4), facecolor=SURF, gridspec_kw={'wspace': 0.04})
    types = [t for t, _, _, _ in EVENTS]
    for ax, (M, lab) in zip(axs, [(M_off, 'Official (Michigan microdata)'), (M_sim, 'ConsumerSim (1,800 personas/month)')]):
        im = ax.imshow(np.clip(M.values, -lim, lim), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
        for i in range(len(ev)):
            for j in range(len(MAP_CONTRASTS)):
                v = M.values[i, j]
                star = '*' if abs(Z.values[i, j]) > 2 else ''
                ax.text(j, i, 'n/a' if np.isnan(v) else f'{v:+.1f}{star}', ha='center', va='center', fontsize=8.2,
                        color=INK, fontweight='bold' if star else 'normal')
        for i in range(1, len(ev)):
            if types[i] != types[i - 1]:
                ax.axhline(i - 0.5, color=SURF, lw=2.5)
        XL = {'Party (Rep - Dem)': 'Party\nRep - Dem', 'Stocks (owner - none)': 'Stocks\nowners - none', 'Education (college - HS or less)': 'Education\ncollege - HS',
              'Income (top - bottom third)': 'Income\ntop - bottom', 'Home (owner - renter)': 'Housing\nowners - renters', 'Age (55+ - 18-34)': 'Age\n55+ - 18-34'}
        ax.set_xticks(range(len(MAP_CONTRASTS)), [XL[c] for c in MAP_CONTRASTS], fontsize=8)
        ax.set_yticks(range(len(ev)), [f'{e}  [{t}]' for t, e in zip(types, ev)] if ax is axs[0] else [''] * len(ev), fontsize=8)
        ax.set_title(lab, fontsize=10.5, color=INK, loc='left')
    cb = fig.colorbar(im, ax=axs, fraction=0.022, pad=0.02)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label('Change in the gap between the two groups over the event window (ICS points)', fontsize=8, color=SUB)
    fig.suptitle('Fig. 5  Signal-specific subgroup sensitivity: change in the gap between paired groups over each event window.  '
                 '* official |gap change| > 2 SE (stars repeated on the right)', x=0.01, ha='left', fontsize=10.5, color=INK)
    fig.savefig(FIG / 'fig5_signal_types.png', dpi=150, facecolor=SURF, bbox_inches='tight')


def fidelity():
    """monthly gap-change correlation with the official series, all months and test months, against the noise ceiling"""
    OG = official_groups()
    out = []
    for design in ('IU', 'frozen'):
        P, H = load(design)
        SG = sim_groups(P, H)
        for c in MAP_CONTRASTS + ['Sex (men - women)']:
            g, se = {}, {}
            for m in MS:
                try:
                    v1 = OG.loc[(c, '1', m), 'ICS'] - OG.loc[(c, '1', D.ym_add(m, -1)), 'ICS']
                    v2 = OG.loc[(c, '2', m), 'ICS'] - OG.loc[(c, '2', D.ym_add(m, -1)), 'ICS']
                    e = sum(OG.loc[(c, k, mm), 'ICS_se'] ** 2 for k in ('1', '2') for mm in (m, D.ym_add(m, -1)))
                except KeyError:
                    continue
                g[m], se[m] = v1 - v2, e
            g, se = pd.Series(g), pd.Series(se)
            sim = pd.Series({m: SG[(c, '1')].get(m, np.nan) - SG[(c, '2')].get(m, np.nan) for m in g.index})
            for lab, ms in (('2014-2026', list(g.index)), ('Test 2023-26', [m for m in g.index if m >= 202301])):
                x, y, e = sim[ms], g[ms], se[ms]
                rel = max(y.var() - e.mean(), 0) / y.var()
                out.append({'design': design, 'contrast': c, 'span': lab, 'months': len(ms), 'r': float(np.corrcoef(x, y)[0, 1]),
                            'noise ceiling': float(np.sqrt(rel)), 'official noise-free SD': float(np.sqrt(max(y.var() - e.mean(), 0))),
                            'simulated SD': float(x.std())})
    R = pd.DataFrame(out)
    R.to_csv(OUT / 'fidelity.csv', index=False)
    print(R.pivot_table(index=['contrast', 'span'], columns='design', values=['r', 'simulated SD']).round(2).to_string())
    print(R[R.design == 'IU'][['contrast', 'span', 'noise ceiling', 'official noise-free SD']].round(2).to_string(index=False))


def sources(R=5):
    """shuffle LLM answers within month x information diet so they no longer depend on who the persona is"""
    from t42_diag_sources import ds_of
    P, _ = load()
    P = P.reset_index(drop=True)

    def pc_of(dcols, scols):
        Q = P.copy()
        for q in D.Q:
            Q[f'dir_F_{q}'], Q[f'size_pd_{q}'] = dcols[q], scols[q]
        import design
        H = design.H(Q)
        return sum(100 * H[q] * ds_of(P['S_' + q], dcols[q], q) for q in D.Q) / 6.7558
    own_d = {q: P[f'dir_F_{q}'].values for q in D.Q}
    own_s = {q: P[f'size_pd_{q}'].values for q in D.Q}
    pcs = {'actual': pc_of(own_d, own_s)}
    groups = list(P.groupby(['month', 'diet']).indices.values())
    acc = np.zeros(len(P))
    for r in range(R):
        rng = np.random.default_rng(100 + r)
        perm = np.arange(len(P))
        for idx in groups:
            perm[idx] = rng.permutation(idx)
        acc += pc_of({q: own_d[q][perm] for q in D.Q}, {q: own_s[q][perm] for q in D.Q})
    pcs['LLM answers shuffled (starting answers kept)'] = acc / R
    rows = []
    for c in MAP_CONTRASTS:
        s1, s2 = CON[c]
        m1, m2 = s1(P).values, s2(P).values
        r = {'contrast': c}
        for k, pc in pcs.items():
            x = pd.DataFrame({'m': P.month.values, 'c': pc * P.w.values, 'w': P.w.values})
            g1, g2 = x[m1].groupby('m').sum(), x[m2].groupby('m').sum()
            r[k] = float((g1.c / g1.w - g2.c / g2.w).std())
        rows.append(r)
    T = pd.DataFrame(rows).set_index('contrast')
    T['share from the LLM response'] = 1 - (T.iloc[:, 1] / T.iloc[:, 0]) ** 2
    T.to_csv(OUT / 'sources.csv')
    print(T.round(2).to_string())


if __name__ == '__main__':
    {'fig4': fig4, 'map': table3_fig5, 'fidelity': fidelity, 'sources': sources}[sys.argv[1]]()
