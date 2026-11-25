import os
"""Fetch official EU27 (Eurostat) and Japan (Cabinet Office ESRI / FRED) CCI target series
plus real-time regional macro, market, and Guardian/NYT economic news data (2018-01 to 2026-10)
strictly inside experiments/01_eu_jp/raw_data/."""
import json
import re
import time
from pathlib import Path

import httpx
import pandas as pd

OUT = Path(__file__).resolve().parent / 'raw_data'
OUT.mkdir(parents=True, exist_ok=True)
ENV_FILE = Path(os.environ.get('CONSUMERSIM_ENV_FILE', str(Path(__file__).resolve().parents[2] / '.env')))


def load_env():
    cfg = {}
    for line in ENV_FILE.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k, v = line.removeprefix('export ').split('=', 1)
            cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


def fetch_eurostat_eu27(cli):
    """Fetch Eurostat official EU27_2020 seasonally adjusted Consumer Confidence Indicator (BS-CSMCI)
    and its 4 sub-components (BS-FS-LY, BS-FS-NY, BS-GES-NY, BS-MP-PR)."""
    p = OUT / 'eurostat_eu27_cci.csv'
    if p.exists():
        print('[EU27] cached:', p)
        return
    url = 'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ei_bsco_m'
    params = [
        ('format', 'JSON'),
        ('lang', 'EN'),
        ('geo', 'EU27_2020'),
        ('s_adj', 'SA'),
        ('sinceTimePeriod', '2010-01'),
    ]
    r = cli.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    dims = data['id']
    sizes = data['size']
    dim_info = data['dimension']
    # Build index -> label maps
    idx_maps = []
    for d in dims:
        cat = dim_info[d]['category']['index']
        inv = {v: k for k, v in cat.items()}
        idx_maps.append(inv)
    vals = data['value']
    rows = []
    # Strides
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    for k_str, v in vals.items():
        k = int(k_str)
        coords = []
        rem = k
        for st in strides:
            coords.append(rem // st)
            rem %= st
        rec = {dims[i]: idx_maps[i][coords[i]] for i in range(len(dims))}
        rec['value'] = float(v)
        rows.append(rec)
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index='time', columns='indic', values='value').sort_index()
    piv.to_csv(p)
    print(f'[EU27] Eurostat fetched: shape={piv.shape}, cols={list(piv.columns)}, range={piv.index[0]}..{piv.index[-1]}')
    print(piv.tail(5))


def fetch_japan_esri(cli, fred_key):
    """Fetch official Japan Cabinet Office / ESRI Consumer Confidence Index (seasonally adjusted, two-or-more-person households)
    directly from ESRI or fallback to FRED OECD series."""
    p = OUT / 'japan_esri_cci.csv'
    if p.exists():
        print('[JP] cached:', p)
        return
    # Try ESRI English page to discover the latest CSV/XLS link
    try:
        r = cli.get('https://www.esri.cao.go.jp/en/stat/shouhi/shouhi-e.html', follow_redirects=True)
        r.raise_for_status()
        # Look for csv or xls links
        links = re.findall(r'href=["\']([^"\']+\.(?:csv|xlsx|xls))["\']', r.text, re.I)
        print('[JP] ESRI links found:', links[:15])
        (OUT / 'esri_links.json').write_text(json.dumps(links, indent=2))
    except Exception as e:
        print('[JP] ESRI page fetch failed:', e)

    # Also check FRED for OECD Japan Consumer Confidence series
    for sid in ['CSCICP03JPM665S', 'JPNCCICP03STSAM', 'CSCICP02JPM460S']:
        try:
            r = cli.get('https://api.stlouisfed.org/fred/series/observations', params={
                'series_id': sid, 'api_key': fred_key, 'file_type': 'json', 'observation_start': '2010-01-01'
            })
            if r.status_code == 200:
                obs = r.json().get('observations', [])
                vals = [(o['date'][:7], float(o['value'])) for o in obs if o['value'] not in ('.', '')]
                if vals:
                    print(f'[JP FRED] {sid}: {len(vals)} obs, {vals[0][0]}..{vals[-1][0]}, last={vals[-1][1]}')
                    pd.DataFrame(vals, columns=['time', sid]).set_index('time').to_csv(OUT / f'jp_fred_{sid}.csv')
        except Exception as e:
            print(f'[JP FRED] {sid} err:', e)


def fetch_macro_market(cli, fred_key):
    """Fetch EU27 and Japan macro/market series from FRED & Yahoo Finance."""
    fred_ids = [
        # EU / Euro Area
        'CP0000EZ19M086NEST',  # Euro area HICP
        'LRHUTTTTEZM156S',     # Euro area unemployment rate
        'ECBDFR',              # ECB Deposit Facility Rate
        'DEXUSEU',             # USD/EUR
        'DCOILBRENTEU',        # Brent Crude Oil
        # Japan
        'JPNCPIALLMINMEI',     # Japan CPI
        'LRUNTTTTJPM156S',     # Japan unemployment rate
        'IRSTCI01JPM156N',     # Japan Immediate / Call Rate
        'DEXJPUS',             # JPY/USD
        # Global financial stress
        'VIXCLS',
    ]
    fdir = OUT / 'fred'
    fdir.mkdir(exist_ok=True)
    for sid in fred_ids:
        p = fdir / f'{sid}.csv'
        if p.exists():
            continue
        r = cli.get('https://api.stlouisfed.org/fred/series/observations', params={
            'series_id': sid, 'api_key': fred_key, 'file_type': 'json', 'observation_start': '2010-01-01'
        })
        if r.status_code == 200:
            obs = r.json().get('observations', [])
            rows = [(o['date'], float(o['value'])) for o in obs if o['value'] not in ('.', '')]
            df = pd.DataFrame(rows, columns=['date', sid]).set_index('date')
            df.to_csv(p)
            print(f'[FRED] {sid}: {len(df)} obs, last={df.index[-1] if len(df) else None}')
        else:
            print(f'[FRED] {sid} FAILED {r.status_code}')
        time.sleep(0.15)

    # Yahoo Finance daily equity indices: Euro Stoxx 50 (^STOXX50E) and Nikkei 225 (^N225)
    mdir = OUT / 'market'
    mdir.mkdir(exist_ok=True)
    headers = {'User-Agent': 'Mozilla/5.0'}
    for sym, name in [('^STOXX50E', 'stoxx50e'), ('^N225', 'nikkei225')]:
        p = mdir / f'{name}.csv'
        if p.exists():
            continue
        r = cli.get(
            f'https://query1.finance.yahoo.com/v8/finance/chart/{sym}',
            params={'period1': '1262304000', 'period2': '1795500000', 'interval': '1d'},
            headers=headers,
        )
        if r.status_code == 200:
            res = r.json()['chart']['result'][0]
            ts = res['timestamp']
            closes = res['indicators']['quote'][0]['close']
            rows = [(pd.to_datetime(t, unit='s').strftime('%Y-%m-%d'), c) for t, c in zip(ts, closes) if c is not None]
            df = pd.DataFrame(rows, columns=['date', name]).set_index('date')
            df.to_csv(p)
            print(f'[Yahoo] {name}: {len(df)} obs, {df.index[0]}..{df.index[-1]}')
        else:
            print(f'[Yahoo] {name} FAILED {r.status_code}')


def main():
    cfg = load_env()
    with httpx.Client(timeout=45) as cli:
        fetch_eurostat_eu27(cli)
        fetch_japan_esri(cli, cfg['FRED_API_KEY'])
        fetch_macro_market(cli, cfg['FRED_API_KEY'])


if __name__ == '__main__':
    main()
