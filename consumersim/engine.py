"""Minimal async LLM client with resumable JSONL ledger and a hard budget cap."""
import asyncio
import json
import os
import random
import time
from pathlib import Path

import httpx

from prompts import parse, parse_contrast

MODEL = 'gpt-6-luna'
ROUTE = os.environ.get('OPENAI_BASE_URL', os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')).rstrip('/')
PRICE_IN, PRICE_OUT = 0.12 / 1e6, 0.6 / 1e6
# reference prices (USD per token) from T2/C6 public ratios x1.2 (same convention as the T9 ledger for GPT)
# gateway ratios (api/pricing): input $/token = model_ratio * 2e-6, output = input * completion_ratio; x1.2 as in T9 ledger
_RATIOS = {'gpt-6-luna': (0.05, 5), 'deepseek-v4.1-flash': (0.5, 4), 'glm-5.3-flash': (0.2, 3.5),
           'grok-4-1-fast-non-reasoning': (0.1, 2.5), 'gpt-oss-120b': (0.05, 5), 'gemini-3.1-flash-lite': (0.125, 6),
           'gemini-3.5-flash-lite': (0.15, 8.333), 'qwen3.5-flash': (0.1, 10), 'doubao-seed-1-6-flash-250615': (0.075, 10),
           'gpt-5.4-nano': (0.1, 6.25), 'deepseek-v4-flash': (0.5, 2),
           'gpt-6.1-terra': (0.5, 5), 'gpt-6-sol': (1, 5), 'gpt-6-astra': (5, 5)}
PRICES = {m: (r * 2e-6 * 1.2, r * 2e-6 * 1.2 * c) for m, (r, c) in _RATIOS.items()}
EXTRA = {'deepseek-v4.1-flash': {'thinking': {'type': 'disabled'}}, 'deepseek-v4-flash': {'thinking': {'type': 'disabled'}},
         'qwen3.5-flash': {'enable_thinking': False}, 'doubao-seed-1-6-flash-250615': {},
         'gemini-3.1-flash-lite': {}, 'gemini-3.5-flash-lite': {},
         'gpt-oss-120b': {'reasoning_effort': 'low'}, 'gpt-5.4-nano': {'reasoning_effort': 'none'}}
NO_REASONING_PARAM = {'grok-4-1-fast-non-reasoning', 'qwen3.5-flash', 'doubao-seed-1-6-flash-250615', 'deepseek-v4-flash',
                      'gemini-3.1-flash-lite', 'gemini-3.5-flash-lite',
                      'deepseek-v4.1-flash', 'glm-5.3-flash'}
ENV = Path(os.environ.get('CONSUMERSIM_ENV_FILE', str(Path(__file__).resolve().parents[1] / '.env')))
LEDGER = Path(__file__).resolve().parents[1] / 'runs' / 'cost_ledger.jsonl'


def _key():
    for env_k in ('OPENAI_API_KEY', 'LLM_API_KEY'):
        if os.environ.get(env_k):
            return os.environ[env_k].strip()
    if ENV.exists():
        cfg = {}
        for line in ENV.read_text(encoding='utf-8').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k, v = line.removeprefix('export ').split('=', 1)
                cfg[k.strip().lower()] = v.strip().strip('"').strip("'")
        for k in ('openai_api_key', 'llm_api_key'):
            if cfg.get(k):
                return cfg[k]
    raise RuntimeError('Set OPENAI_API_KEY (or LLM_API_KEY) in environment or .env')


def spent_total():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)['cost'] for l in LEDGER.open())


class BudgetExceeded(Exception):
    pass


async def run_requests(requests, out_path, cap_usd, concurrency=96, max_attempts=3):
    """requests: list of {'rid', 'messages', ...meta}. Appends one result line per rid to out_path.
    Already-finished rids in out_path are skipped (resume)."""
    for k in ('ALL_PROXY', 'all_proxy'):
        os.environ.pop(k, None)
    out_path = Path(out_path)
    done = set()
    if out_path.exists():
        for l in out_path.open():
            try:
                done.add(json.loads(l)['rid'])
            except json.JSONDecodeError:
                pass
    todo = [r for r in requests if r['rid'] not in done]
    if not todo:
        return
    key = _key()
    start_spent = spent_total()
    state = {'cost': 0.0, 'n': 0, 'fail': 0, 't0': time.time()}
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    fout = out_path.open('a')
    fledger = LEDGER.open('a')
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=ROUTE, timeout=90, limits=limits,
                                 headers={'Authorization': f'Bearer {key}'}) as client:

        async def one(r):
            async with sem:
                result = {'rid': r['rid'], 'states': None, 'error': None, 'attempts': 0, 'raw': [], 'cost': 0.0,
                          'in_tok': 0, 'out_tok': 0}
                for att in range(1, max_attempts + 1):
                    if start_spent + state['cost'] > cap_usd:
                        raise BudgetExceeded(f'cap {cap_usd} reached')
                    result['attempts'] = att
                    model = r.get('model') or MODEL
                    payload = {'model': model, 'messages': r['messages'], 'temperature': 1, 'top_p': 1,
                               'max_tokens': r.get('max_tokens') or 200}
                    if model not in NO_REASONING_PARAM:
                        payload['reasoning_effort'] = r.get('reasoning') or 'none'
                    payload.update(EXTRA.get(model, {}))
                    try:
                        resp = await client.post('/chat/completions', json=payload)
                        obj = resp.json() if resp.status_code == 200 else None
                    except Exception as e:  # network / decode
                        obj, err = None, f'transport:{type(e).__name__}'
                    else:
                        err = None if obj else f'http_{resp.status_code}'
                    if obj and obj.get('choices'):
                        u = obj.get('usage') or {}
                        pi, po = u.get('prompt_tokens') or 0, u.get('completion_tokens') or 0
                        pin, pout = PRICES.get(r.get('model') or MODEL, (PRICE_IN, PRICE_OUT))
                        c = pi * pin + po * pout
                        result['cost'] += c
                        result['in_tok'] += pi
                        result['out_tok'] += po
                        async with lock:
                            state['cost'] += c
                            fledger.write(json.dumps({'rid': r['rid'], 'run': r.get('run'), 'cost': c, 'ts': time.time()}) + '\n')
                        text = obj['choices'][0]['message'].get('content') or ''
                        result['raw'].append(text)
                        if r.get('parser') == 'contrast':
                            states, perr = parse_contrast(text, r['now_label'])
                        elif r.get('parser') == 'rewrite':
                            from t28_rewrite import parse as parse_rw
                            states, perr = parse_rw(text)
                        elif r.get('parser') == 'topics':
                            from topics import parse_topics
                            states, perr = parse_topics(text)
                        elif r.get('parser') == 'number':   # T31 LLM-analyst baseline
                            from t31_llm import parse_number
                            states, perr = parse_number(text)
                        else:
                            states, perr = parse(text.replace('’', "'"), r.get('prev'))
                        if states:
                            result['states'], result['error'] = states, None
                            break
                        result['error'] = perr
                    else:
                        result['error'] = err or 'no_choices'
                        await asyncio.sleep(1.5 * att + random.random())
                async with lock:
                    state['n'] += 1
                    state['fail'] += result['states'] is None
                    fout.write(json.dumps(result) + '\n')
                    fout.flush()  # one complete line per write, safe when several streams append to one file
                    if state['n'] % 2000 == 0:
                        fout.flush()
                        fledger.flush()
                        el = time.time() - state['t0']
                        print(f"  {state['n']}/{len(todo)} done, fail {state['fail']}, ${state['cost']:.2f}, {state['n'] / el:.1f}/s", flush=True)

        try:
            await asyncio.gather(*(one(r) for r in todo))
        finally:
            fout.close()
            fledger.close()
    print(f"finished {state['n']} requests, failed {state['fail']}, cost ${state['cost']:.2f}", flush=True)
