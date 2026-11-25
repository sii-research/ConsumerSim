"""Analyze the date-identification probe against ConsumerSim and baseline errors."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
r = pd.DataFrame([j for j in map(json.loads, open(HERE / 'probe_raw.jsonl')) if j.get('ok')])
r = r.drop_duplicates(['month', 'task', 'sample'])
S = pd.read_csv(ROOT / 'rebuttal/03_strong_baselines/all_methods_series.csv', index_col=0)
S.index = S.index.astype(int)

d = r[r.task == 'date'].copy()
d['gidx'] = d.guess_year * 12 + d.guess_month - 1
d['tidx'] = (d.month // 100) * 12 + d.month % 100 - 1
d['err_m'] = (d.gidx - d.tidx).abs()
lv = r[r.task == 'level'].copy()

M = pd.DataFrame(index=sorted(r.month.unique()))
M['date_exact'] = d.groupby('month').apply(lambda x: (x.err_m == 0).mean())
M['date_err_med'] = d.groupby('month').err_m.median()
M['date_conf'] = d.groupby('month').guess_confidence.mean()
M['ics_guess'] = lv.groupby('month').guess_ics.median()
M['official'] = S.Official
M['recall_err'] = (M.ics_guess - M.official).abs()
M['prev_official'] = S.Persistence
for c, k in [('ConsumerSim', 'cs'), ('Persistence', 'pers'), ('Undated Direct-LLM (gpt-6-luna)', 'dllm'),
             ('SARIMAX Macro', 'sarimax')]:
    M[f'e_{k}'] = (S[c] - S.Official).abs()
M['gain_pers'] = M.e_pers - M.e_cs
# "recall oracle": the undated level guess used as a forecast
M['e_oracle'] = M.recall_err
M.to_csv(HERE / 'probe_months.csv')

rm = lambda e: float(np.sqrt(np.mean(e ** 2)))
out = {}
out['n_months'] = len(M)
out['date_exact_share'] = float(M.date_exact.mean())
out['date_within1_share'] = float((M.date_err_med <= 1).mean())
out['mean_conf'] = float(M.date_conf.mean())
out['undated_recall_rmse'] = rm(M.recall_err)
out['undated_recall_mae'] = float(M.recall_err.mean())
out['cs_rmse'] = rm(M.e_cs); out['pers_rmse'] = rm(M.e_pers); out['dllm_rmse'] = rm(M.e_dllm)
by_year = M.groupby(M.index // 100).agg(date_exact=('date_exact', 'mean'), recall_mae=('recall_err', 'mean'),
                                         cs_mae=('e_cs', 'mean'), pers_mae=('e_pers', 'mean'))
# split by recall quality
known = M.recall_err <= 1.0
for lab, m in [('recall_within_1pt', known), ('recall_off_more_than_1pt', ~known)]:
    sub = M[m]
    out[lab] = dict(n=int(m.sum()), cs_rmse=rm(sub.e_cs), pers_rmse=rm(sub.e_pers), dllm_rmse=rm(sub.e_dllm),
                    cs_gain_vs_pers_rmse=rm(sub.e_pers) - rm(sub.e_cs), mean_recall_err=float(sub.recall_err.mean()))
idf = M.date_exact >= 2 / 3
for lab, m in [('date_identified', idf), ('date_not_identified', ~idf)]:
    sub = M[m]
    out[lab] = dict(n=int(m.sum()), cs_rmse=rm(sub.e_cs), pers_rmse=rm(sub.e_pers), dllm_rmse=rm(sub.e_dllm),
                    cs_gain_vs_pers_rmse=rm(sub.e_pers) - rm(sub.e_cs))
out['corr_recall_err_vs_cs_gain'] = float(np.corrcoef(M.recall_err, M.gain_pers)[0, 1])
out['corr_recall_err_vs_cs_err'] = float(np.corrcoef(M.recall_err, M.e_cs)[0, 1])
# does ConsumerSim's predicted change follow the recalled change?
rc = M.ics_guess - M.prev_official
ac = M.official - M.prev_official
cc = S.ConsumerSim.reindex(M.index) - M.prev_official
out['corr_cs_change_actual'] = float(np.corrcoef(cc, ac)[0, 1])
out['corr_recalled_change_actual'] = float(np.corrcoef(rc, ac)[0, 1])
# partial: does CS change track recalled change beyond actual change?
import statsmodels.api as sm
X = sm.add_constant(np.column_stack([ac, rc]))
f = sm.OLS(cc.values, X).fit(cov_type='HC3')
out['cs_change_on_actual_and_recalled'] = dict(b_actual=float(f.params[1]), p_actual=float(f.pvalues[1]),
                                               b_recalled=float(f.params[2]), p_recalled=float(f.pvalues[2]))
(HERE / 'probe_summary.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print(by_year.round(2).to_string())

# ---------- confound checks ----------
M['abs_move'] = (M.official - M.prev_official).abs()
M['cs_signed'] = S.ConsumerSim.reindex(M.index) - M.official
M['dllm_signed'] = S['Undated Direct-LLM (gpt-6-luna)'].reindex(M.index) - M.official
M['recall_signed'] = M.ics_guess - M.official
fp = {}
for k in ['cs_signed', 'dllm_signed']:
    fp[k] = float(np.corrcoef(M[k], M.recall_signed)[0, 1])
# fingerprint regression controlling the actual move
X = sm.add_constant(np.column_stack([ac.values, M.recall_signed.values]))
f2 = sm.OLS(cc.values, X).fit(cov_type='HC3')
fp['cs_change_on_recall_error'] = dict(b=float(f2.params[2]), se=float(f2.bse[2]), p=float(f2.pvalues[2]))
fd = sm.OLS((S['Undated Direct-LLM (gpt-6-luna)'].reindex(M.index) - M.prev_official).values, X).fit(cov_type='HC3')
fp['dllm_change_on_recall_error'] = dict(b=float(fd.params[2]), se=float(fd.bse[2]), p=float(fd.pvalues[2]))
# gain vs recall error controlling move size
X3 = sm.add_constant(np.column_stack([M.abs_move.values, M.recall_err.values]))
f3 = sm.OLS(M.gain_pers.values, X3).fit(cov_type='HC3')
fp['gain_on_recall_err_ctrl_move'] = dict(b=float(f3.params[2]), p=float(f3.pvalues[2]), b_move=float(f3.params[1]))
M['move_t'] = pd.qcut(M.abs_move, 3, labels=['small', 'medium', 'large'])
strata = []
for t, g in M.groupby('move_t'):
    for lab, m in [('recalled', g.recall_err <= 1.0), ('not_recalled', g.recall_err > 1.0)]:
        sub = g[m]
        if len(sub):
            strata.append(dict(move=t, group=lab, n=len(sub), cs_rmse=rm(sub.e_cs), pers_rmse=rm(sub.e_pers),
                               gain=rm(sub.e_pers) - rm(sub.e_cs)))
fp['strata'] = strata
fp['undated_identified_months_cs_vs_pers'] = None
print(json.dumps(fp, indent=2, default=str))
out['fingerprint'] = fp
(HERE / 'probe_summary.json').write_text(json.dumps(out, indent=2, default=str))
M.to_csv(HERE / 'probe_months.csv')
