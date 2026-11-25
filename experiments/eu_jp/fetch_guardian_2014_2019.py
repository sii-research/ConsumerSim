import os
"""Extend Guardian EU and Japan regional economic news back to 2014-05-01 (covering full 2014-07..2026-09 U.S. window)."""
import json
import time
from pathlib import Path
import httpx

RAW = Path(__file__).resolve().parent / 'raw_data'
NDIR = RAW / 'guardian_news'
ENV_FILE = Path(os.environ.get('CONSUMERSIM_ENV_FILE', str(Path(__file__).resolve().parents[2] / '.env')))


def load_key():
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith('GUARDIAN_API_KEY='):
            return line.split('=', 1)[1].strip().strip('"').strip("'")
    raise RuntimeError('GUARDIAN_API_KEY not found')


def main():
    g_key = load_key()
    queries = {
        'eu': ('business', 'europe OR eurozone OR ECB OR "European Union" OR inflation OR energy'),
        'jp': ('business|world', 'Japan AND (economy OR inflation OR yen OR "Bank of Japan" OR wages OR prices OR trade)'),
    }
    chunks = [
        ('2014-05-01', '2014-12-31'),
        ('2015-01-01', '2015-06-30'), ('2015-07-01', '2015-12-31'),
        ('2016-01-01', '2016-06-30'), ('2016-07-01', '2016-12-31'),
        ('2017-01-01', '2017-06-30'), ('2017-07-01', '2017-12-31'),
        ('2018-01-01', '2018-06-30'), ('2018-07-01', '2018-12-31'),
        ('2019-01-01', '2019-06-30'), ('2019-07-01', '2019-10-31'),
    ]
    with httpx.Client(timeout=45) as cli:
        for reg, (sec, q) in queries.items():
            out_p = NDIR / f'guardian_{reg}_2014_2026.json'
            if out_p.exists():
                print(f'[Guardian {reg}] already merged:', out_p)
                continue
            existing = json.loads((NDIR / f'guardian_{reg}_2019_2026.json').read_text())
            items = list(existing)
            for d0, d1 in chunks:
                for page in (1, 2):
                    r = cli.get('https://content.guardianapis.com/search', params={
                        'section': sec, 'q': q, 'from-date': d0, 'to-date': d1,
                        'order-by': 'relevance', 'page-size': 200, 'page': page,
                        'show-fields': 'trailText,headline', 'api-key': g_key,
                    })
                    if r.status_code != 200:
                        print(f'[Guardian {reg} {d0}] status={r.status_code}')
                        break
                    resp = r.json()['response']
                    items.extend(resp['results'])
                    if page >= resp['pages']:
                        break
                    time.sleep(0.2)
            dedup = {x['id']: x for x in items}
            out_p.write_text(json.dumps(list(dedup.values())))
            print(f'[Guardian {reg}] merged 2014-2026: {len(dedup)} unique articles')


if __name__ == '__main__':
    main()
