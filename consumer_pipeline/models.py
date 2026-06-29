from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


QUESTIONS = ("current_finance", "durable_buying", "future_finance", "business_12m", "business_5y")
CATEGORIES = ("positive", "neutral", "negative")


class Region(str, Enum):
    JP = "JP"
    EU = "EU"
    US = "US"


@dataclass(frozen=True)
class RegionProfile:
    region: Region
    language: str
    locations: tuple[str, ...]
    age_distribution: dict[str, float]
    income_distribution: dict[str, float]
    education_distribution: dict[str, float]
    location_distribution: dict[str, float]
    question_priors: dict[str, dict[str, float]]


@dataclass
class Consumer:
    consumer_id: str
    region: Region
    age_group: str
    income_group: str
    education_group: str
    location_group: str
    weight: float = 1.0
    response: dict[str, str] | None = None
    is_core: bool = False

    def group_key(self, fields: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(str(getattr(self, field)) for field in fields)


@dataclass(frozen=True)
class NewsItem:
    published_at: str
    title: str
    sentiment: float
    relevance: float = 1.0


@dataclass(frozen=True)
class Indicator:
    observed_at: str
    name: str
    z_score: float
    weight: float = 1.0


@dataclass(frozen=True)
class InformationEnvironment:
    as_of: str
    news_items: tuple[NewsItem, ...]
    indicators: tuple[Indicator, ...]
    news_score: float
    indicator_score: float
    combined_score: float


@dataclass(frozen=True)
class Correction:
    previous_month: str | None
    previous_prediction: float | None
    previous_actual: float | None
    residual: float
    applied_adjustment: float


@dataclass
class PipelineResult:
    region: str
    target_month: str
    as_of: str
    population_size: int
    core_size: int
    raw_score: float
    corrected_score: float
    question_scores: dict[str, float]
    environment: dict[str, Any]
    correction: dict[str, Any]
    group_posteriors: dict[str, Any] = field(default_factory=dict)
