"""Front-page topic tags (T24) and the mapping from Michigan NEWS codes to the same topics.

NEWS micro codes are unlabelled in our codebook. Favorable codes are 10-49, unfavorable 50-89. Topics follow the
published category list (time-series codebook: Government/elections, Employment, Consumer demand, Prices, Credit,
Energy, Stock market, Trade) and each code's peak months (receipts/T23).
"""
import json
import re

import numpy as np
import pandas as pd

import data as D

TOPICS = {
    'government': 'the U.S. federal government and national politics: the president and the administration\'s decisions (including foreign policy and military actions), Congress, elections, taxes, government spending or shutdowns',
    'covid': 'the COVID-19 pandemic or other public-health emergencies',
    'jobs': 'jobs, unemployment, layoffs, hiring or wages',
    'prices': 'inflation, prices, the cost of living or shortages of goods',
    'energy': 'gasoline, oil or energy prices',
    'rates': 'interest rates, the Federal Reserve, mortgages or credit',
    'stocks': 'the stock market, Wall Street or investments',
    'trade': 'trade, tariffs or the world economy',
    'business': 'business conditions in general: companies, consumer spending, housing, recession or growth',
}
NEWS_CODES = {
    'government': [10, 50, 14, 54, 16, 19, 59, 22, 62],
    'covid': [20, 60, 21, 61, 47, 87],
    'jobs': [23, 63],
    'prices': [32, 72, 28, 68],
    'energy': [31, 71],
    'rates': [33, 73],
    'stocks': [35, 75, 36, 76],
    'trade': [38, 78, 56],
}
TAGS_PATH = D.ROOT / 'data' / 'topic_tags.json'


def tag_messages(a):
    lines = '\n'.join(f'- {k}: {v}' for k, v in TOPICS.items())
    return [{'role': 'system', 'content': 'You classify newspaper stories by topic.'},
            {'role': 'user', 'content': f"Headline: {' '.join(a['title'].split())}\nSummary: {' '.join(a['abstract'].split())}\n\n"
                                        f"Which of these topics is the story mainly about? Choose at most two, or none.\n{lines}\n\n"
                                        'Return only a JSON object {"topics": [...]} using the topic names above '
                                        '(an empty list if none applies).'}]


def parse_topics(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return None, 'no_json'
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None, 'bad_json'
    t = obj.get('topics')
    if not isinstance(t, list):
        return None, 'no_topics'
    t = [x for x in t if x in TOPICS][:2]
    return {'topics': t}, None


def load_tags():
    return json.loads(TAGS_PATH.read_text())


def news_codes_by_topic():
    known = {c for v in NEWS_CODES.values() for c in v}
    other = [c for c in range(10, 90) if c not in known]
    return {**NEWS_CODES, 'business': other}


def heard(groupby=None, lo=201301, hi=202212):
    """Weighted share of respondents mentioning each topic, per month (and group)."""
    cols = ['YYYYMM', 'WT', 'NEWS1', 'NEWS2', 'POLAFF', 'YTL3']
    d = pd.read_csv(D.MICRO, usecols=cols, dtype=str)
    for c in cols:
        d[c] = pd.to_numeric(d[c].str.strip(), errors='coerce')
    d = d[d.YYYYMM.between(lo, hi)].copy()
    for tp, codes in news_codes_by_topic().items():
        d[tp] = (d.NEWS1.isin(codes) | d.NEWS2.isin(codes)).astype(float) * d.WT
    keys = ['YYYYMM'] + ([groupby] if groupby else [])
    g = d.groupby(keys)
    out = g[list(TOPICS)].sum().div(g.WT.sum(), axis=0)
    out['n'] = g.size()
    return out
