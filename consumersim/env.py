"""Environment information available to a respondent interviewed on a given day.

All values are as known on the day before the interview (asof = day - 1):
- vintage series (UNRATE, CPIAUCSL) use the FRED/ALFRED realtime window;
- daily/weekly market series (S&P 500, gasoline, mortgage rate, Fed target) use observation dates;
- news = NYT front-page (print section A, page 1) stories published before the interview day.
"""
import os
import datetime as dt
import json
import re
from collections import Counter
from functools import lru_cache

import pandas as pd

from data import V3, MAX_MONTH as D_MAX

RAW = V3 / 'raw'
NEW_KEY = {'Subject': 'subject', 'Organization': 'organizations', 'Location': 'glocations', 'Person': 'persons'}
MICHIGAN_RE = re.compile(r'University of Michigan|Michigan survey|consumer sentiment', re.I)
# Economic scope frozen in T9 before any T9/T10 results (economic_scope_frozen.json), applied verbatim
_SCOPE = json.loads((V3 / 'clean/T9_20261115/E/economic_scope_frozen.json').read_text())['scope']
ECON_SUBJECT = re.compile(_SCOPE['subject_pattern'], re.I)
ECON_SECTIONS = set(_SCOPE['sections'])
ECON_DESKS = set(_SCOPE['desks'])


def _d(s):
    return dt.date.fromisoformat(s[:10])


@lru_cache(maxsize=None)
def _vintage(series):
    obs = json.loads((RAW / f'fred/{series}.vintages.all.p0.json').read_text())['observations']
    rows = [(_d(o['realtime_start']), _d(o['realtime_end']) if o['realtime_end'] != '9999-12-31' else dt.date(9999, 12, 31),
             _d(o['date']), float(o['value'])) for o in obs if o['value'] not in ('.', '')]
    return pd.DataFrame(rows, columns=['rs', 're', 'date', 'value'])


def vintage_view(series, asof):
    """Series as published on `asof`: {observation_date: value}, plus the date that release appeared."""
    v = _vintage(series)
    cur = v[(v.rs <= asof) & (v.re >= asof)]
    s = cur.set_index('date').value.sort_index()
    return s, cur.rs.max()


@lru_cache(maxsize=None)
def _daily(series):
    if series == 'SP500':
        y = json.loads((RAW / 'market/sp500/yahoo/yahoo_gspc_20130701_20170131.json').read_text())['chart']['result'][0]
        closes = y['indicators']['quote'][0]['close']
        ys = {dt.datetime.utcfromtimestamp(ts).date(): c for ts, c in zip(y['timestamp'], closes) if c is not None}
        obs = json.loads((RAW / 'fred/SP500.current.all.p0.json').read_text())['observations']
        fs = {_d(o['date']): float(o['value']) for o in obs if o['value'] not in ('.', '')}
        merged = {**ys, **fs}  # FRED where it exists (from 2016-11-14)
        return pd.Series(merged).sort_index()
    obs = json.loads((RAW / f'fred/{series}.current.all.p0.json').read_text())['observations']
    return pd.Series({_d(o['date']): float(o['value']) for o in obs if o['value'] not in ('.', '')}).sort_index()


def last_obs(series, asof):
    s = _daily(series)
    s = s[s.index <= asof]
    return s.index[-1], s.iloc[-1]


def macro(day):
    """Macro facts for an interview on `day`, compared with about one month earlier."""
    asof = day - dt.timedelta(days=1)
    back = asof - dt.timedelta(days=30)
    out = {}
    for key in ['GASREGW', 'SP500', 'MORTGAGE30US', 'DFEDTARU']:
        d1, v1 = last_obs(key, asof)
        d0, v0 = last_obs(key, back)
        out[key] = {'now': v1, 'before': v0, 'obs_now': d1, 'obs_before': d0}
    u, urel = vintage_view('UNRATE', asof)
    out['UNRATE'] = {'now': u.iloc[-1], 'before': u.iloc[-2], 'obs_now': u.index[-1], 'released': urel}
    c, crel = vintage_view('CPIAUCSL', asof)
    last = c.index[-1]
    prev = c.index[-2]
    yoy = lambda d: 100 * (c[d] / c[d.replace(year=d.year - 1)] - 1)
    out['CPI'] = {'now': yoy(last), 'before': yoy(prev), 'obs_now': last, 'released': crel}
    return out


def macro_text(m):
    g, s, mo, f, u, c = (m[k] for k in ['GASREGW', 'SP500', 'MORTGAGE30US', 'DFEDTARU', 'UNRATE', 'CPI'])
    sp = 100 * (s['now'] / s['before'] - 1)
    fed = ('unchanged' if abs(f['now'] - f['before']) < 1e-9 else
           ('raised' if f['now'] > f['before'] else 'lowered'))
    fed_line = (f"Federal Reserve's target interest rate: {fed}, now {max(f['now'] - 0.25, 0):.2f}% to {f['now']:.2f}%"
                if fed == 'unchanged' else
                f"Federal Reserve's target interest rate: {fed} from {max(f['before'] - 0.25, 0):.2f}%-{f['before']:.2f}% to {max(f['now'] - 0.25, 0):.2f}%-{f['now']:.2f}%")
    lines = [
        f"Regular gasoline: ${g['now']:.2f} a gallon (a month ago: ${g['before']:.2f})",
        f"S&P 500 stock index: {s['now']:,.0f} ({'up' if sp >= 0 else 'down'} {abs(sp):.1f}% from a month ago)",
        f"Unemployment rate in the latest monthly jobs report: {u['now']:.1f}% (previous report: {u['before']:.1f}%)",
        f"Consumer prices in the latest report: {'up' if c['now'] >= 0 else 'down'} {abs(c['now']):.1f}% from a year earlier (previous report: {'up' if c['before'] >= 0 else 'down'} {abs(c['before']):.1f}%)",
        f"30-year fixed mortgage rate: {mo['now']:.2f}% (a month ago: {mo['before']:.2f}%)",
        fed_line,
    ]
    return 'Economic figures in the news (latest compared with about a month ago):\n' + '\n'.join('- ' + x for x in lines)


PRICE_SERIES = [('CUUR0000SAF11', 'Groceries (food at home)'), ('CUUR0000SEHA', 'Rent'),
                ('CUUR0000SETA01', 'New cars and trucks'), ('CUUR0000SETA02', 'Used cars and trucks'),
                ('CUSR0000SAD', 'Durable goods (furniture, appliances, electronics, vehicles)')]


def prices_text(day):
    """Year-over-year price changes of everyday categories in the latest CPI report known the day before `day`."""
    asof = day - dt.timedelta(days=1)
    lines = []
    for sid, label in PRICE_SERIES:
        v, _ = vintage_view(sid, asof)
        last = v.index[-1]
        yoy = 100 * (v[last] / v[last.replace(year=last.year - 1)] - 1)
        lines.append(f"{label}: {'up' if yoy >= 0 else 'down'} {abs(yoy):.1f}% from a year earlier")
    return 'Prices of everyday items in the latest consumer price report:\n' + '\n'.join('- ' + x for x in lines)


def macro_levels_text(m):
    """Levels only (no comparison), for one snapshot."""
    g, s, mo, f, u, c = (m[k] for k in ['GASREGW', 'SP500', 'MORTGAGE30US', 'DFEDTARU', 'UNRATE', 'CPI'])
    lines = [
        f"Regular gasoline: ${g['now']:.2f} a gallon",
        f"S&P 500 stock index: {s['now']:,.0f}",
        f"Unemployment rate in the latest monthly jobs report: {u['now']:.1f}%",
        f"Consumer prices in the latest report: {'up' if c['now'] >= 0 else 'down'} {abs(c['now']):.1f}% from a year earlier",
        f"30-year fixed mortgage rate: {mo['now']:.2f}%",
        f"Federal Reserve's target interest rate: {max(f['now'] - 0.25, 0):.2f}% to {f['now']:.2f}%",
    ]
    return 'Economic figures in the news:\n' + '\n'.join('- ' + x for x in lines)


@lru_cache(maxsize=None)
def front_pages(ym):
    """Front-page (print section A, page 1) NYT news articles for a calendar month file."""
    p = RAW / f'nyt/{ym // 100:04d}-{ym % 100:02d}.json'
    if not p.exists():
        return []
    docs = json.loads(p.read_text())['response']['docs']
    out = []
    for a in docs:
        if str(a.get('print_page')) != '1' or a.get('print_section') != 'A':
            continue
        if (a.get('type_of_material') or 'News') not in ('News', 'News Analysis'):
            continue
        title = (a.get('headline') or {}).get('main') or ''
        abstract = a.get('abstract') or a.get('snippet') or ''
        if not title:
            continue
        byl = a.get('byline') or {}
        byline = ' '.join(p.get('lastname') or '' for p in (byl.get('person') or []))
        kws = a.get('keywords') or []
        # T30 fix: from 2025-01 the archive capitalises keyword types (Subject / Organization / Location / Person) and
        # drops the byline person list; map them to the old names so 2025-2026 stories keep their keys. Unchanged before 2025.
        if any(k.get('name') in NEW_KEY for k in kws):
            kws = [{**k, 'name': NEW_KEY.get(k.get('name'), k.get('name'))} for k in kws]
            byline = byline or (byl.get('original') or '')
        keys = []
        for k in sorted(kws, key=lambda k: k.get('rank', 99)):
            if k['name'] not in ('subject', 'organizations', 'glocations', 'persons'):
                continue
            if k['name'] == 'persons' and byline and k['value'].split(',')[0] in byline:
                continue
            keys.append(k['value'])
        pub = re.sub(r'([+-]\d{2})(\d{2})$', r'\1:\2', a['pub_date'].replace('Z', '+00:00'))
        subjects = [k['value'] for k in kws if k['name'] == 'subject']
        econ = (a.get('section_name') in ECON_SECTIONS or a.get('news_desk') in ECON_DESKS
                or any(ECON_SUBJECT.search(v) for v in subjects))
        out.append({'id': a.get('uri') or a.get('_id'), 'pub': dt.datetime.fromisoformat(pub), 'econ': econ,
                    'title': title, 'abstract': abstract, 'keys': keys[:5], 'words': int(a.get('word_count') or 0),
                    'michigan': bool(MICHIGAN_RE.search(title + ' ' + abstract))})
    return out


def _first_sentence(s, n=240):
    s = re.sub(r'\s+', ' ', s).strip()
    # sentence end: ". " followed by a capital, not after a title or a single capital letter (Mr., U.S.)
    for m in re.finditer(r'[.!?](?=\s+[A-Z“"‘])', s):
        before = s[:m.start()].split(' ')[-1]
        if before in ('Mr', 'Mrs', 'Ms', 'Dr', 'St', 'Gov', 'Sen', 'Rep', 'Gen', 'Lt', 'Jr') or re.fullmatch(r'(?:[A-Z]\.)*[A-Z]', before):
            continue
        s = s[:m.start() + 1]
        break
    return s if len(s) <= n else s[:n].rsplit(' ', 1)[0] + '…'


@lru_cache(maxsize=1)
def _all_front_pages():
    arts = []
    m = 201207
    while m <= D_MAX:  # NYT files after the validation set are read only in test mode
        arts.extend(front_pages(m))
        m = m + 1 if m % 100 < 12 else (m // 100 + 1) * 100 + 1
    return sorted(arts, key=lambda a: a['pub'])


@lru_cache(maxsize=None)
def _key_counts(lo, hi, econ=False):
    """Topic-key article counts for stories published in [lo, hi)."""
    c = Counter()
    for a in _all_front_pages():
        if lo <= a['pub'] < hi and not a['michigan'] and (a['econ'] or not econ):
            c.update(set(a['keys']))
    return c


def _bounds(day, window):
    hi = dt.datetime.combine(day, dt.time(0), dt.timezone(dt.timedelta(hours=-5)))  # start of interview day, US East
    return hi - dt.timedelta(days=window), hi


@lru_cache(maxsize=None)
def news(day, n=15, window=30, baseline_days=360, econ=False):
    """Front-page topics of the `window` days before `day`, ranked by coverage x burst.

    burst = coverage in the window relative to the average 30-day coverage over the prior `baseline_days`,
    capped at 3, so always-present tags (e.g. 'United States Politics and Government') do not crowd out
    new events. Big topics get up to 3 headlines, taken from their peak coverage days."""
    lo, hi = _bounds(day, window)
    arts = [a for a in _all_front_pages() if lo <= a['pub'] < hi and not a['michigan'] and (a['econ'] or not econ)]
    base = _key_counts(lo - dt.timedelta(days=baseline_days), lo, econ)
    return _rank(arts, base, window / baseline_days, n)


def _rank(arts, base, scale, n):
    """Coverage x burst ranking shared by news() and news_topic(); big topics get up to 3 headlines."""
    picked, covered = [], set()
    while len(picked) < n:
        cnt = Counter(k for a in arts if a['id'] not in covered for k in set(a['keys']))
        cnt = {k: c for k, c in cnt.items() if c >= 2}
        if not cnt:
            break
        score = {k: c * min(3.0, c / (base.get(k, 0) * scale + 1.0)) for k, c in cnt.items()}
        key = max(score, key=score.get)
        on = [a for a in arts if a['id'] not in covered and key in a['keys']]
        slots = 1 + (len(on) >= 15) + (len(on) >= 40)
        by_day = Counter(a['pub'].astimezone(dt.timezone(dt.timedelta(hours=-5))).date() for a in on)
        for d0, _ in sorted(by_day.items(), key=lambda x: (-x[1], -x[0].toordinal()))[:slots]:
            same = [a for a in on if a['pub'].astimezone(dt.timezone(dt.timedelta(hours=-5))).date() == d0]
            rep = max(same, key=lambda a: a['words'])
            picked.append({**rep, 'topic': key, 'count': len(on), 'score': score[key]})
            if len(picked) >= n:
                break
        covered |= {a['id'] for a in on}
    return tuple(picked)


@lru_cache(maxsize=1)
def _tags():
    from topics import load_tags
    return load_tags()


@lru_cache(maxsize=None)
def news_topic(day, topic, n, window=30, baseline_days=360, burst=False):
    """Front-page stories of the `window` days before `day` tagged with `topic` (T24). No survey-article filter:
    under the input rule only the time stamp gates what may be shown."""
    lo, hi = _bounds(day, window)
    tags = _tags()
    arts = [a for a in _all_front_pages() if lo <= a['pub'] < hi and topic in tags.get(a['id'], ())]
    base = _key_counts(lo - dt.timedelta(days=baseline_days), lo, False) if burst else {}
    return _rank(arts, base, window / baseline_days, n)  # without burst: rank by coverage within the topic


def news_text(items, econ=False):
    def title(t):
        return t if t[-1:] in '.?!' else t + '.'
    lines = [f"- {title(a['title'])} {_first_sentence(a['abstract'])}".rstrip() for a in items]
    head = ('Front-page news stories about the economy over the past month (most-covered topics first):' if econ
            else 'Front-page news stories over the past month (most-covered topics first):')
    return head + '\n' + '\n'.join(lines) if items else head + '\n- (none)'


def _yoy(series, asof):
    v, _ = vintage_view(series, asof)
    last = v.index[-1]
    return 100 * (v[last] / v[last.replace(year=last.year - 1)] - 1)


def personal_text(p, day):
    """Figures and stories most relevant to this persona's own household, as known the day before `day`."""
    asof = day - dt.timedelta(days=1)
    ud = lambda x: f"{'up' if x >= 0 else 'down'} {abs(x):.1f}%"
    lines = [f"Average hourly pay of workers: {ud(_yoy('CES0500000003', asof))} from a year earlier"]
    if p.get('YTL3') == 1 or p.get('HOMEOWN') == 2:
        lines.append(f"Grocery prices: {ud(_yoy('CUUR0000SAF11', asof))} from a year earlier")
        lines.append(f"Rents: {ud(_yoy('CUUR0000SEHA', asof))} from a year earlier")
    if p.get('HOMEOWN') == 1 and not vintage_view('CSUSHPINSA', asof - dt.timedelta(days=31))[0].empty:
        # only when the series was already published at both snapshot dates (real-time vintages start 2014-11-25)
        lines.append(f"Home prices (Case-Shiller): {ud(_yoy('CSUSHPINSA', asof))} from a year earlier")
    if p.get('INVEST') == 1 and _daily('SP500').index[0] <= asof - dt.timedelta(days=365 + 31):
        sp = _daily('SP500')
        s1 = sp[sp.index <= asof].iloc[-1]
        s0 = sp[sp.index <= asof - dt.timedelta(days=365)].iloc[-1]
        lines.append(f"Stock market (S&P 500): {ud(100 * (s1 / s0 - 1))} over the past 12 months")
    out = 'Things that matter most for your own household:\n' + '\n'.join('- ' + x for x in lines)
    if p.get('POLAFF') in (1, 2):
        econ_ids = {a['id'] for a in news(day, econ=True)}
        pol = [a for a in news(day) if a['id'] not in econ_ids][:4]
        if pol:
            out += '\n\nOther front-page stories (politics and public affairs):\n' + '\n'.join(
                f"- {a['title'] if a['title'][-1:] in '.?!' else a['title'] + '.'}" for a in pol)
    return out


def political_text(day):
    """Who holds the presidency as of the day before `day` (real-time: race calls 2016-11-09 and 2020-11-07)."""
    d = day - dt.timedelta(days=1)
    if d < dt.date(2016, 11, 9):
        s = 'President: Barack Obama (Democrat).'
    elif d < dt.date(2017, 1, 20):
        s = 'President: Barack Obama (Democrat). President-elect: Donald Trump (Republican), who won the November 2016 election.'
    elif d < dt.date(2020, 11, 7):
        s = 'President: Donald Trump (Republican).'
    elif d < dt.date(2021, 1, 20):
        s = 'President: Donald Trump (Republican). President-elect: Joe Biden (Democrat), who won the November 2020 election.'
    elif d < dt.date(2024, 11, 6):   # T43 fix: 2024 election (race called 2024-11-06) and Trump's second term
        s = 'President: Joe Biden (Democrat).'
    elif d < dt.date(2025, 1, 20):
        s = 'President: Joe Biden (Democrat). President-elect: Donald Trump (Republican), who won the November 2024 election.'
    else:
        s = 'President: Donald Trump (Republican).'
    return 'Political situation:\n- ' + s


def job_exposed(p):
    """T48 job-loss exposure: working age (18-64) and either no four-year degree or under 35 (Hoynes, Miller & Schaller 2012)."""
    age, educ = p.get('AGE'), p.get('EDUC')
    if age != age or age is None or not 18 <= age < 65:
        return False
    return (educ == educ and educ is not None and educ <= 4) or age < 35


REGION_CPI = {3: ('CUUR0100SA0', 'the Northeast'), 2: ('CUUR0200SA0', 'the Midwest'), 4: ('CUUR0300SA0', 'the South'),
              1: ('CUUR0400SA0', 'the West')}


def household_uniform(p, day, other=None):
    """T48 uniform exposure layer, one snapshot, as known the day before `day`: every attribute that maps to an economic
    channel brings its own real-time indicator (wealth: stocks; work: claims or the rate savings earn; budget: groceries and
    gasoline; housing: home prices and mortgage rate, or rents; place: regional prices). Sex and family bring none."""
    asof = day - dt.timedelta(days=1)
    ud = lambda x: f"{'up' if x >= 0 else 'down'} {abs(x):.1f}%"
    lines = []
    if p.get('INVEST') == 1:
        _, s1 = last_obs('SP500', asof)
        _, s0 = last_obs('SP500', asof - dt.timedelta(days=30))
        ch = 100 * (s1 / s0 - 1)
        lines.append(f"Your family's stock investments: {'up' if ch >= 0 else 'down'} about {abs(ch):.0f}% over the past month")
    age = p.get('AGE')
    if age == age and age is not None and age >= 65:
        _, f = last_obs('DFEDTARU', asof)
        lines.append(f"What savings earn (Federal Reserve's target rate): {max(f - 0.25, 0):.2f}% to {f:.2f}%")
    elif job_exposed(p):
        c, _ = vintage_view('ICSA', asof)
        v = float(c.iloc[-1])
        lines.append('New claims for unemployment benefits in the latest week: ' + (f'{v / 1e6:.1f} million' if v >= 1e6 else f'{round(v, -3):,.0f}'))
    if p.get('YTL3') in (1, 2):
        lines.append(f"Grocery prices: {ud(_yoy('CUUR0000SAF11', asof))} from a year earlier")
        _, g = last_obs('GASREGW', asof)
        lines.append(f'Regular gasoline: ${g:.2f} a gallon')
    if p.get('HOMEOWN') == 1:
        # T56 fix: shown only when available in both snapshots (`other` = the other snapshot's day); one-sided in 2014-12
        have = lambda d0: not vintage_view('CSUSHPINSA', d0 - dt.timedelta(days=1))[0].empty
        if have(day) and (other is None or os.environ.get('T56_SNAPFIX') != '1' or have(other)):   # T56 test, not adopted
            lines.append(f"Home prices (Case-Shiller): {ud(_yoy('CSUSHPINSA', asof))} from a year earlier")
        _, mo = last_obs('MORTGAGE30US', asof)
        lines.append(f'30-year fixed mortgage rate: {mo:.2f}%')
    elif p.get('HOMEOWN') == 2:
        lines.append(f"Rents: {ud(_yoy('CUUR0000SEHA', asof))} from a year earlier")
    if p.get('REGION') in REGION_CPI:
        sid, name = REGION_CPI[p['REGION']]
        lines.append(f'Consumer prices in {name}: {ud(_yoy(sid, asof))} from a year earlier')
    return ('For your household:\n' + '\n'.join('- ' + x for x in lines)) if lines else ''
