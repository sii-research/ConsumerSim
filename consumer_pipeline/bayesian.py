from __future__ import annotations

import random
from collections import defaultdict

from .models import CATEGORIES, QUESTIONS, Consumer, RegionProfile


class BayesianAggregator:
    """Dirichlet-multinomial update by demographic group, followed by population expansion."""

    def __init__(self, profile: RegionProfile, prior_strength: float, seed: int) -> None:
        if prior_strength <= 0:
            raise ValueError("bayesian.prior_strength must be positive")
        self.profile = profile
        self.prior_strength = prior_strength
        self.rng = random.Random(seed)

    def update_and_expand(
        self,
        population: list[Consumer],
        core: list[Consumer],
        group_fields: tuple[str, ...],
    ) -> dict[str, dict[str, dict[str, float]]]:
        observations: dict[tuple[str, ...], list[Consumer]] = defaultdict(list)
        for consumer in core:
            if consumer.response is None:
                raise ValueError(f"Core consumer {consumer.consumer_id} has no prediction")
            observations[consumer.group_key(group_fields)].append(consumer)

        population_groups = {consumer.group_key(group_fields) for consumer in population}
        global_posterior = self._posterior(core)
        posteriors: dict[tuple[str, ...], dict[str, dict[str, float]]] = {}
        for key in population_groups:
            group_observations = observations.get(key, [])
            posteriors[key] = self._posterior(group_observations) if group_observations else global_posterior

        for consumer in population:
            if consumer.response is not None:
                continue
            posterior = posteriors[consumer.group_key(group_fields)]
            consumer.response = {
                question: self.rng.choices(
                    CATEGORIES,
                    weights=[posterior[question][category] for category in CATEGORIES],
                    k=1,
                )[0]
                for question in QUESTIONS
            }
        return {"|".join(key): value for key, value in sorted(posteriors.items())}

    def aggregate(self, population: list[Consumer]) -> tuple[float, dict[str, float]]:
        total_weight = sum(max(consumer.weight, 0.0) for consumer in population)
        if total_weight <= 0:
            raise ValueError("Population weights must sum to a positive value")
        question_scores: dict[str, float] = {}
        for question in QUESTIONS:
            positive = sum(
                consumer.weight for consumer in population if consumer.response and consumer.response[question] == "positive"
            )
            negative = sum(
                consumer.weight for consumer in population if consumer.response and consumer.response[question] == "negative"
            )
            question_scores[question] = round(100.0 + 100.0 * (positive - negative) / total_weight, 4)
        overall = round(sum(question_scores.values()) / len(question_scores), 4)
        return overall, question_scores

    def _posterior(self, observations: list[Consumer]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for question in QUESTIONS:
            alpha = {
                category: self.profile.question_priors[question][category] * self.prior_strength
                for category in CATEGORIES
            }
            for consumer in observations:
                assert consumer.response is not None
                category = consumer.response[question]
                if category not in CATEGORIES:
                    raise ValueError(f"Invalid response category {category!r}")
                alpha[category] += max(consumer.weight, 0.0)
            total = sum(alpha.values())
            result[question] = {category: round(alpha[category] / total, 8) for category in CATEGORIES}
        return result
