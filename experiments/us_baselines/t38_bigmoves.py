"""T38: reaction fidelity in the largest official movements (data-driven Table 2 / Fig. 4).

The 20 months with the largest |official ICS change| over 2014-07 to 2026-09 are taken first; events are identified
afterwards from that month's front pages (receipts/T38/headlines.md). Every method is scored on the same 20 months.
Run with CS_TEST=1 after t33_paper.py.
"""
import datetime as dt
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

assert os.environ.get('CS_TEST') == '1'
import data as D
import env
from score import paired
from t33_paper import PAPER, SIM, STRONG
from t36_figs import style, INK, MUTED, SUB, SURF, BLUE, ORANGE, GRID

OUT = D.ROOT / 'receipts' / 'T38'
OUT.mkdir(parents=True, exist_ok=True)
N = 10   # user decision 2026-11-19: the ten largest moves
EVENT = {   # identified from the month's front pages (headlines.md) and the Michigan Final Results text
    202004: ('COVID-19 lockdowns and job losses', '疫情封锁与失业潮'),
    202003: ('COVID-19 outbreak and market crash', '疫情暴发与股市崩盘'),
    202108: ('Delta wave and the fall of Kabul', 'Delta 疫情反弹与喀布尔陷落'),
    202401: ('Disinflation and record stock highs', '通胀回落、股市创新高'),
    201908: ('U.S.-China tariff escalation', '中美关税升级'),
    202506: ('Tariff truce and market recovery', '关税休战、股市回升'),
    202312: ('Fed pivot and soft-landing hopes', '美联储转向、软着陆预期'),
    202206: ('$5 gasoline and 40-year-high inflation', '5 美元汽油与 40 年最高通胀'),
    202405: ('Higher-for-longer rates and sticky inflation', '高利率更久与通胀反复'),
    202103: ('$1,400 relief checks and vaccine rollout', '1,400 美元纾困支票与疫苗加速'),
    202503: ('Steel and aluminum tariffs and federal layoffs', '钢铝关税生效与联邦裁员'),
    202307: ('Inflation falls to 3%', '通胀降到 3%'),
    201901: ('Government shutdown', '政府停摆'),
    202205: ('Inflation stays hot, stocks near bear market', '通胀居高、股市逼近熊市'),
    202208: ('Gasoline falls below $4, jobs fully recovered', '汽油跌破 4 美元、就业恢复到疫情前'),
    201611: ('Presidential election (Trump)', '特朗普当选'),
    202502: ('Tariff threats and soaring egg prices', '关税威胁与蛋价飙升'),
    202009: ('Reopening and jobs recovery', '经济重启、就业恢复'),
    202204: ('Gasoline retreats from its March peak', '汽油从 3 月高点回落'),
    202006: ('Reopening and surprise jobs rebound', '经济重启、5 月就业意外大增'),
}


def main():
    df = pd.read_csv(D.ROOT / 'receipts/T33/all_series.csv', index_col=0)
    methods = [SIM, 'Persistence'] + PAPER + STRONG
    d = df.Official - df.Persistence
    top = list(d.abs().sort_values(ascending=False).head(N).index)
    # headlines used to identify the events
    hl = ['# T38 二十个最大变动月份的头版（访问窗口最后一天之前 30 天，按报道量 × 突发度排序）', '']
    for m in top:
        e = dt.date.fromisoformat(D.dates()[m]['interview_end'])
        hl += [f'## {m}  官方变化 {d[m]:+.1f}', ''] + [f'- {a["title"]}' for a in env.news(e, n=8)] + [''] + \
              ['经济类：'] + [f'- {a["title"]}' for a in env.news(e, n=6, econ=True)] + ['']
    (OUT / 'headlines.md').write_text('\n'.join(hl))
    rows = []
    for m in top:
        s = df.loc[m, SIM] - df.loc[m, 'Persistence']
        rows.append({'event': EVENT[m][0], '事件': EVENT[m][1], 'month': f'{m // 100}-{m % 100:02d}', 'official change': d[m],
                     'ConsumerSim change': s, 'SARIMAX change': df.loc[m, 'SARIMAX Macro'] - df.loc[m, 'Persistence'],
                     'direction': bool(np.sign(s) == np.sign(d[m])), 'ConsumerSim error': abs(df.loc[m, SIM] - df.loc[m, 'Official']),
                     'persistence error': abs(d[m])})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'table2_top20.csv', index=False)
    board = []
    for k in methods:
        dk = df.loc[top, k] - df.loc[top, 'Persistence']
        board.append({'method': k, 'direction': int((np.sign(dk) == np.sign(d[top])).sum()) if k != 'Persistence' else 0,
                      'MAE': float((df.loc[top, k] - df.loc[top, 'Official']).abs().mean()),
                      'RMSE': float(np.sqrt(((df.loc[top, k] - df.loc[top, 'Official']) ** 2).mean())),
                      'share of move captured': float((dk * d[top]).sum() / (d[top] ** 2).sum())})
    B = pd.DataFrame(board).set_index('method')
    sig = {k: paired(df[SIM].to_dict(), df[k].to_dict(), sorted(top), block=2) for k in methods[1:]}
    B['paired RMSE diff (ConsumerSim - method)'] = [np.nan] + [f"{sig[k]['diff']:+.2f} [{sig[k]['ci_low']:+.2f}, {sig[k]['ci_high']:+.2f}]" +
                                                              (' *' if sig[k]['significant_better'] else '') for k in methods[1:]]
    B.to_csv(OUT / 'scoreboard_top20.csv')
    lines = ['# T38 二十个最大变动月份', '', T.round(2).to_markdown(index=False), '', f'## 各方法在这 {N} 个月上', '', B.round(2).to_markdown(), '']
    (OUT / 'table2_top20.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
    # Fig. 4
    fig = plt.figure(figsize=(15, 8.2), facecolor=SURF)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1], wspace=0.42, left=0.2, right=0.97, top=0.9, bottom=0.08)
    ax = fig.add_subplot(gs[0])
    style(ax)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, lw=0.8)
    TT = T.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(TT))
    h = 0.27
    ax.barh(y + h, TT['official change'], height=h, color=INK, label='Official', zorder=3)
    ax.barh(y, TT['ConsumerSim change'], height=h, color=ORANGE, label='ConsumerSim', zorder=3)
    ax.barh(y - h, TT['SARIMAX change'], height=h, color=BLUE, label='SARIMAX Macro (best baseline overall)', zorder=3)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, [f"{r.month}  {r.event}" for r in TT.itertuples()], fontsize=8)
    ax.set_xlim(-19, 12)
    ax.set_xlabel('Change from last month (ICS points)', fontsize=8.5, color=SUB)
    ax.legend(frameon=False, fontsize=8.5, loc='lower left')
    ax.set_title(f'A. The {N} largest monthly moves, 2014-2026', fontsize=10, color=INK, loc='left')
    ax2 = fig.add_subplot(gs[1])
    style(ax2)
    ax2.grid(axis='y', visible=False)
    ax2.grid(axis='x', color=GRID, lw=0.8)
    full = {k: float((df[k] - df.Official).abs().mean()) for k in [SIM, 'Persistence'] + PAPER}
    show = [k for k in full if full[k] <= full['Persistence'] + 1e-9]   # forecasters that beat no-change over the full sample
    show = sorted(show, key=lambda k: B.loc[k, 'MAE'])
    Bs = B.loc[show].iloc[::-1]
    yy = np.arange(len(Bs))
    ax2.barh(yy, Bs.MAE, color=[ORANGE if k == SIM else '#b9b8b3' for k in Bs.index], height=0.6, zorder=3)
    for yi, (k, r) in zip(yy, Bs.iterrows()):
        ax2.text(r.MAE + 0.1, yi, f"{r.MAE:.2f}", va='center', fontsize=8, color=INK)
    ax2.set_yticks(yy, [k.replace(' (1,800/month)', '').replace(' (6 figures)', '').replace(' (changes)', '') for k in Bs.index], fontsize=8)
    ax2.set_xlim(0, 11.5)
    ax2.set_xlabel(f'Mean absolute error in the {N} months (ICS points)', fontsize=8.5, color=SUB)
    ax2.set_title('B. The 8 forecasters that beat no-change', fontsize=10, color=INK, loc='left')
    fig.suptitle('Fig. 2b  Reaction fidelity in the ten largest monthly moves of consumer sentiment', x=0.01, ha='left', fontsize=11.5, color=INK)
    fig.savefig(D.ROOT / 'receipts/T36/fig4_top20.png', dpi=160, facecolor=SURF)


def paper_table():
    """Paper-format Table 2 (user decision 2026-11-19): the ten largest monthly moves of the official ICS; forecasters are ranked
    by their error in the month of the move, among those whose full-sample error is below persistence."""
    df = pd.read_csv(D.ROOT / 'receipts/T33/all_series.csv', index_col=0)
    every = [SIM, 'Persistence'] + PAPER
    full = {k: float((df[k] - df.Official).abs().mean()) for k in every}
    adm = [k for k in every if full[k] <= full['Persistence'] + 1e-9]
    d = df.Official - df.Persistence
    top = list(d.abs().sort_values(ascending=False).head(N).index)
    err = lambda k, ms: (df.loc[ms, k] - df.loc[ms, 'Official'])
    rows, wins = [], {k: 0.0 for k in adm}
    for m in top:
        e = pd.Series({k: float(abs(err(k, [m]).iloc[0])) for k in adm})
        r = e.rank(method='min')
        for k in r[r == 1].index:
            wins[k] += 1 / (r == 1).sum()
        rows.append({'Regime': EVENT[m][0], '事件': EVENT[m][1], 'Event month': f'{m // 100}-{m % 100:02d}', 'Official change': d[m],
                     'ConsumerSim change': df.loc[m, SIM] - df.loc[m, 'Persistence'], 'ConsumerSim error': e[SIM],
                     'Best other error': float(e.drop(SIM).min()), 'Rank': int(r[SIM])})
    pm = pd.Series({k: float(err(k, top).abs().mean()) for k in adm}).rank(method='min')
    pr = pd.Series({k: float(np.sqrt((err(k, top) ** 2).mean())) for k in adm}).rank(method='min')
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'table2_paper.csv', index=False)
    meta = {'methods': adm, 'n_methods': len(adm), 'N': N, 'pooled MAE rank': int(pm[SIM]), 'pooled RMSE rank': int(pr[SIM]),
            '#1': int((T.Rank == 1).sum()), 'top-3': int((T.Rank <= 3).sum()), 'worst rank': int(T.Rank.max()),
            'wins by method': {k: round(v, 1) for k, v in sorted(wins.items(), key=lambda x: -x[1])}}
    (OUT / 'table2_paper_meta.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(T.round(2).to_string(index=False))
    print(json.dumps(meta, indent=1, ensure_ascii=False))
    return T, meta


if __name__ == '__main__':
    import sys
    paper_table() if sys.argv[1:] == ['paper'] else main()
