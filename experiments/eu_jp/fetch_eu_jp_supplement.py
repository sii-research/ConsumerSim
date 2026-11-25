import os
"""Fetch remaining EU27 (Eurostat unemployment & HICP YoY) + Japan (CPI YoY & ESRI xlsx)
+ Guardian regional economic news (2019-11-01..2026-10-31) into experiments/01_eu_jp/raw_data/."""
import json
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


def parse_eurostat(data):
    dims, sizes, dim_info = data['id'], data['size'], data['dimension']
    idx_maps = [{v: k for k, v in dim_info[d]['category']['index'].items()} for d in dims]
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    rows = []
    for k_str, v in data['value'].items():
        k = int(k_str)
        coords = []
        rem = k
        for st in strides:
            coords.append(rem // st)
            rem %= st
        rec = {dims[i]: idx_maps[i][coords[i]] for i in range(len(dims))}
        rec['value'] = float(v)
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    cfg = load_env()
    fred_key = cfg['FRED_API_KEY']
    g_key = cfg['GUARDIAN_API_KEY']

    with httpx.Client(timeout=45) as cli:
        # 1. Eurostat EU27 unemployment (SA, total, % active population)
        p_u = OUT / 'eurostat_eu27_unemp.csv'
        if not p_u.exists():
            r = cli.get('https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/une_rt_m', params=[
                ('format', 'JSON'), ('lang', 'EN'), ('geo', 'EU27_2020'),
                ('s_adj', 'SA'), ('age', 'TOTAL'), ('sex', 'T'), ('unit', 'PC_ACT'),
                ('sinceTimePeriod', '2010-01'),
            ])
            r.raise_for_status()
            df = parse_eurostat(r.json())[['time', 'value']].sort_values('time').set_index('time')
            df.columns = ['eu27_unemp']
            df.to_csv(p_u)
            print(f'[EU27 Unemp] {len(df)} obs, {df.index[0]}..{df.index[-1]}, last={df.iloc[-1,0]}')

        # 2. Eurostat EU27 HICP annual rate of change (YoY %, all-items + food + energy)
        p_h = OUT / 'eurostat_eu27_hicp_yoy.csv'
        if not p_h.exists():
            r = cli.get('https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hicp_manr', params=[
                ('format', 'JSON'), ('lang', 'EN'), ('geo', 'EU27_2020'),
                ('coicop', 'CP00'), ('coicop', 'FOOD'), ('coicop', 'NRG'),
                ('sinceTimePeriod', '2010-01'),
            ])
            r.raise_for_status()
            df = parse_eurostat(r.json()).pivot_table(index='time', columns='coicop', values='value').sort_index()
            df.to_csv(p_h)
            print(f'[EU27 HICP YoY] shape={df.shape}, {df.index[0]}..{df.index[-1]}')

        # 3. Japan CPI YoY from FRED
        fdir = OUT / 'fred'
        for sid in ['CPALTT01JPM659N', 'JPNCPALTT01GYM', 'JPNCPIALLMINMEI', 'FPCPITOTLZGJPN']:
            p = fdir / f'{sid}.csv'
            if p.exists() and sid == 'JPNCPIALLMINMEI':
                continue
            r = cli.get('https://api.stlouisfed.org/fred/series/observations', params={
                'series_id': sid, 'api_key': fred_key, 'file_type': 'json', 'observation_start': '2010-01-01'
            })
            if r.status_code == 200:
                obs = r.json().get('observations', [])
                rows = [(o['date'], float(o['value'])) for o in obs if o['value'] not in ('.', '')]
                if rows:
                    df = pd.DataFrame(rows, columns=['date', sid]).set_index('date')
                    df.to_csv(p)
                    print(f'[FRED JP] {sid}: {len(df)} obs, last={df.index[-1]}, val={df.iloc[-1,0]}')
            time.sleep(0.15)

        # 4. Japan ESRI shouhi1.xlsx (for exact official sub-components)
        p_es = OUT / 'shouhi1.xlsx'
        if not p_es.exists():
            r = cli.get('https://www.esri.cao.go.jp/en/stat/shouhi/shouhi1.xlsx', follow_redirects=True)
            if r.status_code == 200:
                p_es.write_bytes(r.content)
                print(f'[JP ESRI] downloaded shouhi1.xlsx ({len(r.content)} bytes)')

        # 5. Guardian regional economic news for EU27 and Japan (quarterly queries 2019-11..2026-10)
        ndir = OUT / 'guardian_news'
        ndir.mkdir(exist_ok=True)
        queries = {
            'eu': ('business', 'europe OR eurozone OR ECB OR "European Union" OR inflation OR energy'),
            'jp': ('business|world', 'Japan AND (economy OR inflation OR yen OR "Bank of Japan" OR wages OR prices OR trade)'),
        }
        # Fetch half-year chunks so every month 2019-11..2026-10 has dense top headlines
        chunks = [
            ('2019-11-01', '2020-06-30'), ('2020-07-01', '2020-12-31'),
            ('2021-01-01', '2021-06-30'), ('2021-07-01', '2021-12-31'),
            ('2022-01-01', '2022-06-30'), ('2022-07-01', '2022-12-31'),
            ('2023-01-01', '2023-06-30'), ('2023-07-01', '2023-12-31'),
            ('2024-01-01', '2024-06-30'), ('2024-07-01', '2024-12-31'),
            ('2025-01-01', '2025-06-30'), ('2025-07-01', '2025-12-31'),
            ('2026-01-01', '2026-06-30'), ('2026-07-01', '2026-10-31'),
        ]
        for reg, (sec, q) in queries.items():
            p = ndir / f'guardian_{reg}_2019_2026.json'
            if p.exists():
                continue
            items = []
            for d0, d1 in chunks:
                for page in (1, 2):
                    r = cli.get('https://content.guardianapis.com/search', params={
                        'section': sec,
                        'q': q,
                        'from-date': d0,
                        'to-date': d1,
                        'order-by': 'relevance',
                        'page-size': 200,
                        'page': page,
                        'show-fields': 'trailText,headline',
                        'api-key': g_key,
                    })
                    if r.status_code != 200:
                        print(f'[Guardian {reg} {d0}] status={r.status_code}')
                        break
                    resp = r.json()['response']
                    items.extend(resp['results'])
                    if page >= resp['pages']:
                        break
                    time.sleep(0.2)
            # Deduplicate by id
            dedup = {x['id']: x for x in items}
            p.write_text(json.dumps(list(dedup.values())))
            print(f'[Guardian {reg}] saved {len(dedup)} unique articles')


if __name__ == '__main__':
    main()
