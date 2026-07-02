from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from .bayesian import BayesianAggregator
from .config import read_secret_env, resolve_path
from .debias import PreviousMonthCorrector
from .information import InformationEnvironmentBuilder
from .models import PipelineResult
from .output import write_outputs
from .population import PopulationSampler
from .prediction import OpenAICompatibleCorePredictor, ProbabilisticCorePredictor
from .regions import get_region_profile


class ConsumerPipeline:
    """Population -> information -> core prediction -> Bayesian aggregation -> t-1 correction."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.profile = get_region_profile(str(config["region"]))
        self.seed = int(config.get("seed", 42))

    def run(self, target_month: str, as_of: str) -> tuple[PipelineResult, Path]:
        self._validate_run_dates(target_month, as_of)
        population_config = self.config["population"]
        sampler = PopulationSampler(self.profile, self.seed)
        population = sampler.sample(
            int(population_config["size"]),
            resolve_path(self.config, population_config.get("source_csv")),
        )
        group_fields = tuple(population_config.get(
            "group_fields", ["age_group", "income_group", "education_group", "location_group"]
        ))
        core = sampler.select_core(population, float(population_config["core_ratio"]), group_fields)

        information_config = self.config["information"]
        environment = InformationEnvironmentBuilder().build(
            as_of=as_of,
            news_path=resolve_path(self.config, information_config.get("news_jsonl")),
            indicators_path=resolve_path(self.config, information_config.get("indicators_csv")),
            news_weight=float(information_config.get("news_weight", 0.5)),
            indicator_weight=float(information_config.get("indicator_weight", 0.5)),
        )

        predictor = self._build_predictor()
        predictor.predict(core, environment)

        bayesian_config = self.config["bayesian"]
        aggregator = BayesianAggregator(
            self.profile,
            prior_strength=float(bayesian_config.get("prior_strength", 20.0)),
            seed=self.seed + 1,
        )
        posteriors = aggregator.update_and_expand(population, core, group_fields)
        raw_score, question_scores = aggregator.aggregate(population)

        correction_config = self.config["correction"]
        corrected_score, correction = PreviousMonthCorrector(
            weight=float(correction_config.get("weight", 0.5)),
            max_absolute_adjustment=float(correction_config.get("max_absolute_adjustment", 10.0)),
        ).apply(
            raw_score,
            target_month,
            resolve_path(self.config, correction_config.get("history_csv")),
        )

        result = PipelineResult(
            region=self.profile.region.value,
            target_month=target_month,
            as_of=as_of,
            population_size=len(population),
            core_size=len(core),
            raw_score=raw_score,
            corrected_score=corrected_score,
            question_scores=question_scores,
            environment={
                "news_score": environment.news_score,
                "indicator_score": environment.indicator_score,
                "combined_score": environment.combined_score,
                "news_count": len(environment.news_items),
                "indicator_count": len(environment.indicators),
                "news_items": [asdict(item) for item in environment.news_items[-8:]],
                "indicators": [asdict(item) for item in environment.indicators[-8:]],
            },
            correction=asdict(correction),
            group_posteriors=posteriors,
        )
        output_path = resolve_path(self.config, self.config["output"]["directory"])
        assert output_path is not None
        return result, write_outputs(output_path, result, population)

    def _build_predictor(self):
        prediction = self.config["prediction"]
        provider = str(prediction.get("provider", "local")).lower()
        if provider == "local":
            return ProbabilisticCorePredictor(
                self.profile,
                self.seed,
                signal_strength=float(prediction.get("signal_strength", 0.08)),
            )
        if provider == "openai_compatible":
            api_key = read_secret_env(self.config, "model_api_key")
            return OpenAICompatibleCorePredictor(
                self.profile,
                endpoint=str(prediction["endpoint"]),
                model=str(prediction["model"]),
                api_key=api_key or "",
                timeout=int(prediction.get("timeout_seconds", 60)),
            )
        raise ValueError("prediction.provider must be local or openai_compatible")

    @staticmethod
    def _validate_run_dates(target_month: str, as_of: str) -> None:
        try:
            month_start = date.fromisoformat(f"{target_month}-01")
            cutoff = date.fromisoformat(as_of)
        except ValueError as exc:
            raise ValueError("target_month must be YYYY-MM and as_of must be YYYY-MM-DD") from exc
        next_month = date(month_start.year + (month_start.month == 12), month_start.month % 12 + 1, 1)
        if cutoff >= next_month:
            raise ValueError("as_of must not be later than the target month")
