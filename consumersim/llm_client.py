"""Shared OpenAI-compatible LLM payload helper matching consumersim/engine.py."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'consumersim') not in sys.path:
    sys.path.insert(0, str(ROOT / 'consumersim'))
import engine  # noqa: E402


def make_payload(model, messages, max_tokens=240, temperature=None, seed=None):
    """Build chat-completions payload matching consumersim/engine.py."""
    payload = {
        'model': model,
        'messages': messages,
        'temperature': 1 if temperature is None else temperature,
        'top_p': 1,
        'max_tokens': max_tokens,
    }
    if model not in engine.NO_REASONING_PARAM:
        payload['reasoning_effort'] = 'none'
    payload.update(engine.EXTRA.get(model, {}))
    if seed is not None:
        payload['seed'] = int(seed)
    return payload
