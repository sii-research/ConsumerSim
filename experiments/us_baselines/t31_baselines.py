"""T31: the paper's ICS baselines (Huang et al., Appendix A), real-time, expanding window, all scored months 2014-07 to 2026-09.

Every forecast for month t uses only (i) official ICS up to t-1 (the Final for t-1 is out before month t ends) and
(ii) covariates as of the nowcast day d_t = middle of month t's interview window (day 10 before 2013), with approximate
publication lags for monthly series. Covariates are final-vintage FRED data (this favours the baselines a little).
Hyperparameters (ridge penalties, ARIMA orders) are chosen on the training window only.
Run with CS_TEST=1.  Output: receipts/T31/baselines.csv (month x method).
"""
import datetime as dt
import os
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

assert os.environ.get('CS_TEST') == '1'
import data as D
import env
from score import official_target

warnings.filterwarnings('ignore')
FD = D.ROOT / 'data/ext/fred_dl'
OUT = D.ROOT / 'receipts/T31' if os.environ.get('T56_CPI') != '1' else D.ROOT / 'receipts/T56/baselines_cpifix'
OUT.mkdir(parents=True, exist_ok=True)
EVAL = D.ALL_SCORED + D.VALID + D.TEST
START = 199101
TOPICS = ['government', 'covid', 'jobs', 'prices', 'energy', 'rates', 'stocks', 'trade', 'business']


@lru_cache(maxsize=None)
def fred(s):
    x = pd.read_csv(FD / f'{s}.csv', index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(x, errors='coerce').dropna()


@lru_cache(maxsize=None)
def spx():
    s = pd.read_csv(D.ROOT / 'data/ext/spx/gspc_daily.csv', index_col=0, parse_dates=True).close
    t = env._daily('SP500').copy()
    t.index = pd.to_datetime(t.index)
    return pd.concat([s[s.index < t.index[0]], t]).sort_index()


def ym_iter(a, b):
    m = a
    while m <= b:
        yield m
        m = D.ym_add(m, 1)


@lru_cache(maxsize=None)
def nowcast_day(m):
    ds = D.dates()
    if m in ds:
        s, e = D.interview_start(m), dt.date.fromisoformat(ds[m]['interview_end'])
        return s + (e - s) / 2
    return dt.date(m // 100, m % 100, 10)


def last_daily(x, day, back=0):
    d = pd.Timestamp(day - dt.timedelta(days=back))
    y = x[x.index < d]
    return float(y.iloc[-1]) if len(y) else np.nan


def last_monthly(x, day, lag_days, k=0):
    """value of the k-th latest monthly observation published before `day` (published lag_days after month end)."""
    pub = x.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag_days)
    y = x[pub < pd.Timestamp(day)]
    return float(y.iloc[-1 - k]) if len(y) > k else np.nan


@lru_cache(maxsize=None)
def _pubs():
    arts = env._all_front_pages()
    return [pd.Timestamp(a['pub']).tz_convert(None) for a in arts], [a['id'] for a in arts]


@lru_cache(maxsize=None)
def news_shares(day):
    tags = env._tags()
    lo, hi = pd.Timestamp(day - dt.timedelta(days=30)), pd.Timestamp(day)
    pub, ids = _pubs()
    arts = [{'id': i} for t, i in zip(pub, ids) if lo <= t < hi]
    cnt = {t: 0 for t in TOPICS}
    n = 0
    for a in arts:
        tg = tags.get(a['id'])
        if not tg:
            continue
        n += 1
        for t in tg:
            if t in cnt:
                cnt[t] += 1
    return {f'news_{t}': cnt[t] / max(n, 1) for t in TOPICS} | {'news_n': np.log1p(n)}


@lru_cache(maxsize=None)
def feats(m):
    d = nowcast_day(m)
    cpi = fred('CPIAUCSL')
    if os.environ.get('T56_CPI') == '1':   # T56 check (audit #9): year-on-year by date; the positional shift spans 13 months after the missing 2025-10 CPI
        infl = (cpi / cpi.shift(12, freq='MS') - 1).dropna() * 100
    else:
        infl = (cpi / cpi.shift(12) - 1) * 100
    sp = spx()
    f = {
        'unrate': last_monthly(fred('UNRATE'), d, 8), 'd_unrate': last_monthly(fred('UNRATE'), d, 8) - last_monthly(fred('UNRATE'), d, 8, 1),
        'infl': last_monthly(infl.dropna(), d, 15), 'd_infl': last_monthly(infl.dropna(), d, 15) - last_monthly(infl.dropna(), d, 15, 1),
        'payroll': last_monthly(fred('PAYEMS').diff().dropna(), d, 8),
        'claims': np.log(last_daily(fred('ICSA').rolling(4).mean().dropna().shift(5, freq='D'), d)),
        'gas': np.log(last_daily(fred('GASREGW'), d)), 'd_gas': np.log(last_daily(fred('GASREGW'), d) / last_daily(fred('GASREGW'), d, 30)),
        'mortgage': last_daily(fred('MORTGAGE30US'), d), 'd_mortgage': last_daily(fred('MORTGAGE30US'), d) - last_daily(fred('MORTGAGE30US'), d, 30),
        'fed': last_daily(fred('DFF'), d), 'd_fed': last_daily(fred('DFF'), d) - last_daily(fred('DFF'), d, 30),
        'spx': np.log(last_daily(sp, d)), 'r_spx': np.log(last_daily(sp, d) / last_daily(sp, d, 30)),
        'vix': last_daily(fred('VIXCLS'), d), 'y10': last_daily(fred('DGS10'), d), 'd_y10': last_daily(fred('DGS10'), d) - last_daily(fred('DGS10'), d, 30),
        'term': last_daily(fred('T10Y2Y'), d),
        'mich': last_monthly(fred('MICH'), d, 0), 'd_mich': last_monthly(fred('MICH'), d, 0) - last_monthly(fred('MICH'), d, 0, 1),
        'cle': last_monthly(fred('EXPINF1YR'), d, -10), 'd_cle': last_monthly(fred('EXPINF1YR'), d, -10) - last_monthly(fred('EXPINF1YR'), d, -10, 1),
    }
    if m >= 201208:
        f.update(news_shares(d))
    return f


MACRO = ['unrate', 'd_unrate', 'infl', 'd_infl', 'payroll', 'claims', 'gas', 'd_gas', 'mortgage', 'd_mortgage', 'fed', 'd_fed']
MARKET = ['spx', 'r_spx', 'vix', 'y10', 'd_y10', 'term']
NEWS = [f'news_{t}' for t in TOPICS] + ['news_n']
EXPECT = ['mich', 'd_mich', 'cle', 'd_cle']
STABLE = ['unrate', 'infl', 'fed', 'mortgage']
SARIMAX_X = ['d_unrate', 'infl', 'r_spx', 'd_gas', 'mortgage', 'fed']


def ics_hist():
    o = official_target()
    u = fred('UMCSENT')
    out = {int(f'{d.year}{d.month:02d}'): float(v) for d, v in u.items() if d.year >= 1978}
    out.update({m: v for m, v in o.items() if m >= 197801})
    return out


def ridge_fit_predict(X, y, xt):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    r = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(X), y)
    return float(r.predict(sc.transform(xt[None, :]))[0])


def main():
    ics = ics_hist()
    hist = [m for m in ym_iter(START, EVAL[-1])]
    F = pd.DataFrame({m: feats(m) for m in hist}).T
    F.to_csv(OUT / 'features.csv')
    y = pd.Series({m: ics[m] for m in hist})
    lag = lambda k: pd.Series({m: ics[D.ym_add(m, -k)] for m in hist})
    L1, L2, L3 = lag(1), lag(2), lag(3)
    res = {k: {} for k in ['AR Ridge', 'Expectations Reg.', 'News Semantic FAAR', '3-Month Mean', 'SARIMAX Macro', 'ETS Damped Trend',
                           'Theta Model', 'Auto ARIMA', 'Prophet Macro', 'Prophet Univ.', 'Macro-Market-News Ridge', 'Macro Stable Ridge',
                           'Persistence + MMN Ridge (changes)']}
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.forecasting.theta import ThetaModel
    orders = {}
    for t in EVAL:
        tr = [m for m in hist if m < t]
        trn = [m for m in tr if m >= 201309]           # news features need a full 30-day window of tagged front pages
        Xa = np.c_[L1[tr], L2[tr], L3[tr]]
        res['AR Ridge'][t] = ridge_fit_predict(Xa, y[tr].values, np.array([L1[t], L2[t], L3[t]]))
        Xe = np.c_[L1[tr], F.loc[tr, EXPECT].values]
        res['Expectations Reg.'][t] = ridge_fit_predict(Xe, y[tr].values, np.r_[L1[t], F.loc[t, EXPECT].values.astype(float)])
        # FAAR: AR(3) + 3 principal components of the news topic shares (PCA fitted on the training window)
        from sklearn.decomposition import PCA
        Nn = F.loc[trn, NEWS].values.astype(float)
        mu, sd = Nn.mean(0), Nn.std(0) + 1e-9
        pca = PCA(3).fit((Nn - mu) / sd)
        fz = pca.transform((Nn - mu) / sd)
        ft = pca.transform(((F.loc[t, NEWS].values.astype(float) - mu) / sd)[None, :])[0]
        res['News Semantic FAAR'][t] = ridge_fit_predict(np.c_[L1[trn], L2[trn], L3[trn], fz], y[trn].values, np.r_[L1[t], L2[t], L3[t], ft])
        res['3-Month Mean'][t] = float(np.mean([L1[t], L2[t], L3[t]]))
        yy = y[tr].values
        ex = F.loc[tr, SARIMAX_X].values.astype(float)
        # order grids re-selected by AIC every January (and at the first month), parameters refit every month
        if t % 100 == 1 or 'sx' not in orders:
            best = {}
            for kind in ('sx', 'ar'):
                cands = []
                for p in (0, 1, 2):
                    for dd in (0, 1):
                        for q in (0, 1, 2):
                            if p + q == 0:
                                continue
                            try:
                                r = SARIMAX(yy, exog=ex if kind == 'sx' else None, order=(p, dd, q), trend='c' if dd == 0 else 'n').fit(disp=False)
                                cands.append((r.aic, (p, dd, q)))
                            except Exception:
                                pass
                best[kind] = min(cands)[1]
            orders.update(best)
        o = orders['sx']
        r = SARIMAX(yy, exog=ex, order=o, trend='c' if o[1] == 0 else 'n').fit(disp=False)
        res['SARIMAX Macro'][t] = float(r.forecast(1, exog=F.loc[[t], SARIMAX_X].values.astype(float))[0])
        o = orders['ar']
        r = SARIMAX(yy, order=o, trend='c' if o[1] == 0 else 'n').fit(disp=False)
        res['Auto ARIMA'][t] = float(r.forecast(1)[0])
        res['ETS Damped Trend'][t] = float(ExponentialSmoothing(yy, trend='add', damped_trend=True).fit().forecast(1)[0])
        res['Theta Model'][t] = float(ThetaModel(yy, period=12, deseasonalize=False).fit().forecast(1).iloc[0])
        Xm = F.loc[trn, MACRO + MARKET + NEWS].values.astype(float)
        res['Macro-Market-News Ridge'][t] = ridge_fit_predict(Xm, y[trn].values, F.loc[t, MACRO + MARKET + NEWS].values.astype(float))
        res['Macro Stable Ridge'][t] = ridge_fit_predict(F.loc[tr, STABLE].values.astype(float), yy, F.loc[t, STABLE].values.astype(float))
        # stronger variant (ours): persistence + ridge on month-over-month changes
        ch = [c for c in MACRO + MARKET if c.startswith(('d_', 'r_')) or c in ('payroll',)] + NEWS
        dy = (y[trn] - L1[trn]).values
        res['Persistence + MMN Ridge (changes)'][t] = L1[t] + ridge_fit_predict(F.loc[trn, ch].values.astype(float), dy, F.loc[t, ch].values.astype(float))
        print(t, {k: round(v[t], 1) for k, v in res.items() if t in v}, flush=True)
    # Prophet (slow to import; fitted monthly)
    from prophet import Prophet
    import logging
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    logging.getLogger('prophet').setLevel(logging.ERROR)
    ds = pd.Series({m: pd.Timestamp(f'{m // 100}-{m % 100:02d}-01') for m in hist})
    for t in EVAL:
        tr = [m for m in hist if m < t]
        df = pd.DataFrame({'ds': ds[tr].values, 'y': y[tr].values})
        p = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False).fit(df)
        res['Prophet Univ.'][t] = float(p.predict(pd.DataFrame({'ds': [ds[t]]})).yhat.iloc[0])
        df2 = df.copy()
        for c in SARIMAX_X:
            df2[c] = F.loc[tr, c].values.astype(float)
        p = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        for c in SARIMAX_X:
            p.add_regressor(c)
        p.fit(df2)
        nt = pd.DataFrame({'ds': [ds[t]]} | {c: [float(F.loc[t, c])] for c in SARIMAX_X})
        res['Prophet Macro'][t] = float(p.predict(nt).yhat.iloc[0])
    out = pd.DataFrame(res)
    out.index.name = 'month'
    out.to_csv(OUT / 'baselines.csv')
    print(out.tail())


if __name__ == '__main__':
    main()
