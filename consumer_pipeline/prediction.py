from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from typing import Protocol

from .models import CATEGORIES, QUESTIONS, Consumer, InformationEnvironment, RegionProfile


class CorePredictor(Protocol):
    def predict(self, consumers: list[Consumer], environment: InformationEnvironment) -> None: ...


class ProbabilisticCorePredictor:
    """Reproducible local predictor used when no external model is configured."""

    def __init__(self, profile: RegionProfile, seed: int, signal_strength: float) -> None:
        self.profile = profile
        self.rng = random.Random(seed)
        self.signal_strength = signal_strength

    def predict(self, consumers: list[Consumer], environment: InformationEnvironment) -> None:
        for consumer in consumers:
            response: dict[str, str] = {}
            demographic_shift = self._demographic_shift(consumer)
            for question in QUESTIONS:
                prior = self.profile.question_priors[question]
                shift = self.signal_strength * environment.combined_score + demographic_shift
                probabilities = self._shift_distribution(prior, shift)
                response[question] = self.rng.choices(CATEGORIES, weights=probabilities, k=1)[0]
            consumer.response = response

    @staticmethod
    def _demographic_shift(consumer: Consumer) -> float:
        income = {"low": -0.08, "middle": 0.0, "high": 0.06}.get(consumer.income_group, 0.0)
        age = {"18-34": 0.02, "35-54": 0.0, "55+": -0.01}.get(consumer.age_group, 0.0)
        return income + age

    @staticmethod
    def _shift_distribution(prior: dict[str, float], shift: float) -> list[float]:
        positive = max(0.01, prior["positive"] + shift)
        negative = max(0.01, prior["negative"] - shift)
        neutral = max(0.01, prior["neutral"])
        total = positive + neutral + negative
        return [positive / total, neutral / total, negative / total]


class OpenAICompatibleCorePredictor:
    """Optional batch predictor; the API key is accepted at runtime, never stored in config."""

    def __init__(self, profile: RegionProfile, endpoint: str, model: str, api_key: str, timeout: int = 60) -> None:
        if not api_key:
            raise ValueError("Configured credential environment variable is empty")
        self.profile = profile
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def predict(self, consumers: list[Consumer], environment: InformationEnvironment) -> None:
        for consumer in consumers:
            consumer.response = self._request(consumer, environment)

    def _request(self, consumer: Consumer, environment: InformationEnvironment) -> dict[str, str]:
        prompt = {
            "task": "Return one of positive, neutral, negative for every survey question.",
            "region": self.profile.region.value,
            "consumer": {
                "age_group": consumer.age_group,
                "income_group": consumer.income_group,
                "education_group": consumer.education_group,
                "location_group": consumer.location_group,
            },
            "information_environment": {
                "news_score": environment.news_score,
                "indicator_score": environment.indicator_score,
                "headlines": [item.title for item in environment.news_items[:8]],
            },
            "questions": list(QUESTIONS),
        }
        body = json.dumps({
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
        }).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            result = json.loads(content)
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Prediction request failed: {exc}") from exc
        invalid = {question: value for question, value in result.items() if value not in CATEGORIES}
        if set(result) != set(QUESTIONS) or invalid:
            raise ValueError("Model response must contain exactly the five questions with valid categories")
        return {question: str(result[question]) for question in QUESTIONS}
