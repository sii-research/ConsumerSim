"""Date-identification probe on ConsumerSim's undated inputs (knowledge-leakage diagnostic).

For each of the 147 evaluation months, gpt-6-luna receives the exact undated shared snapshot used
by ConsumerSim and the Direct LLM baseline (political context, macroeconomic figures, NYT economic
front-page news) and is asked, in separate calls,
  (A) which calendar year and month the snapshot describes, and
  (B) the level of the Michigan Index of Consumer Sentiment at that time.
Three independent samples per month and task. Outputs: probe_raw.jsonl, probe_months.csv.
Usage: python run_date_probe.py [--pilot]
"""
import asyncio
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'consumersim'))
sys.path.insert(0, str(ROOT / 'consumersim'))
os.environ['CS_TEST'] = '1'
import data as D  # noqa: E402
import env  # noqa: E402
import engine  # noqa: E402
from llm_client import make_payload  # noqa: E402

OUT = ROOT / 'consumersim' / '06_date_probe'
RAW = OUT / 'probe_raw.jsonl'
LEDGER = OUT / 'cost_ledger.jsonl'
EVAL = D.ALL_SCORED + D.VALID + D.TEST
MODEL = 'gpt-6-luna'
N_SAMPLES = 3


def nowcast_day(m):
    ds = D.dates()
    if m in ds:
        s, e = D.interview_start(m), dt.date.fromisoformat(ds[m]['interview_end'])
        return s + (e - s) / 2
    return dt.date(m // 100, m % 100, 10)


def make_snap(day):
    parts = [env.political_text(day), env.macro_levels_text(env.macro(day)),
             env.news_text(env.news(day, econ=True), econ=True)]
    return '\n\n'.join(parts)


PROMPTS = {
    'date': (
        "Below is a snapshot of U.S. economic and political conditions. All dates have been removed.\n\n"
        "=== SNAPSHOT ===\n{snap}\n\n"
        "Which calendar year and month does this snapshot most likely describe? "
        "Give your single best guess even if you are unsure.\n"
        'Respond ONLY with valid JSON: {{"year": <int>, "month": <int 1-12>, "confidence": <int 0-100>}}'
    ),
    'level': (
        "Below is a snapshot of U.S. economic and political conditions. All dates have been removed.\n\n"
        "=== SNAPSHOT ===\n{snap}\n\n"
        "What was the level of the University of Michigan Index of Consumer Sentiment (ICS) at the time of "
        "this snapshot? Give your single best estimate.\n"
        'Respond ONLY with valid JSON: {{"ics": <float>}}'
    ),
}


def done_keys():
    if not RAW.exists():
        return set()
    return {(j['month'], j['task'], j['sample']) for j in map(json.loads, open(RAW)) if j.get('ok')}


async def main(pilot=False):
    months = [201407, 201811, 202004, 202206, 202501] if pilot else EVAL
    snaps = {m: make_snap(nowcast_day(m)) for m in months}
    have = done_keys()
    jobs = [(m, t, s) for m in months for t in PROMPTS for s in range(N_SAMPLES) if (m, t, s) not in have]
    print(f'{len(jobs)} calls to run')
    key = engine._key()
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    url = f'{engine.ROUTE}/chat/completions'
    sem = asyncio.Semaphore(16)

    async def one(cli, m, task, s):
        prompt = PROMPTS[task].format(snap=snaps[m])
        async with sem:
            for attempt in range(4):
                try:
                    r = await cli.post(url, headers=headers, json=make_payload(
                        MODEL, [{'role': 'user', 'content': prompt}], max_tokens=80, seed=1000 + s))
                    r.raise_for_status()
                    data = r.json()
                    txt = data['choices'][0]['message']['content']
                    u = data.get('usage', {})
                    with open(LEDGER, 'a') as f:
                        f.write(json.dumps({'task': f'date_probe_{task}', 'month': m, 'model': MODEL,
                                            'in': u.get('prompt_tokens', 0), 'out': u.get('completion_tokens', 0)}) + '\n')
                    j = json.loads(re.search(r'\{.*\}', txt, re.S).group(0))
                    rec = dict(month=m, task=task, sample=s, ok=True, **{f'guess_{k}': v for k, v in j.items()})
                    with open(RAW, 'a') as f:
                        f.write(json.dumps(rec) + '\n')
                    return
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        print(f'failed {m} {task} {s}: {e}')
                    await asyncio.sleep(1.5 * (attempt + 1))

    async with httpx.AsyncClient(timeout=90) as cli:
        await asyncio.gather(*(one(cli, *j) for j in jobs))

    recs = [j for j in map(json.loads, open(RAW)) if j.get('ok')]
    df = pd.DataFrame(recs)
    print(df.tail(12).to_string())


if __name__ == '__main__':
    asyncio.run(main(pilot='--pilot' in sys.argv))
