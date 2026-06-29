from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from .models import Indicator, InformationEnvironment, NewsItem


class InformationEnvironmentBuilder:
    """Builds a point-in-time-safe information snapshot from normalized local inputs."""

    def build(
        self,
        as_of: str,
        news_path: Path | None,
        indicators_path: Path | None,
        news_weight: float,
        indicator_weight: float,
    ) -> InformationEnvironment:
        cutoff = date.fromisoformat(as_of)
        news = tuple(item for item in self._read_news(news_path) if date.fromisoformat(item.published_at[:10]) <= cutoff)
        indicators = tuple(
            item for item in self._read_indicators(indicators_path) if date.fromisoformat(item.observed_at[:10]) <= cutoff
        )
        news_score = self._weighted_mean([(item.sentiment, item.relevance) for item in news])
        indicator_score = self._weighted_mean([(item.z_score, item.weight) for item in indicators])
        combined = self._weighted_mean([(news_score, news_weight), (indicator_score, indicator_weight)])
        return InformationEnvironment(
            as_of=as_of,
            news_items=news,
            indicators=indicators,
            news_score=round(news_score, 6),
            indicator_score=round(indicator_score, 6),
            combined_score=round(max(-3.0, min(3.0, combined)), 6),
        )

    def _read_news(self, path: Path | None) -> list[NewsItem]:
        if path is None:
            return []
        if not path.exists():
            raise FileNotFoundError(f"News input not found: {path}")
        items: list[NewsItem] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    items.append(NewsItem(
                        published_at=str(row["published_at"]),
                        title=str(row["title"]),
                        sentiment=self._bounded(float(row["sentiment"])),
                        relevance=max(0.0, float(row.get("relevance", 1.0))),
                    ))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid news row {line_number} in {path}: {exc}") from exc
        return items

    def _read_indicators(self, path: Path | None) -> list[Indicator]:
        if path is None:
            return []
        if not path.exists():
            raise FileNotFoundError(f"Indicator input not found: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"observed_at", "name", "z_score", "weight"}
        if rows and not required.issubset(rows[0]):
            raise ValueError(f"Indicator CSV missing columns: {', '.join(sorted(required - set(rows[0])))}")
        return [
            Indicator(
                observed_at=row["observed_at"],
                name=row["name"],
                z_score=max(-3.0, min(3.0, float(row["z_score"]))),
                weight=max(0.0, float(row.get("weight") or 1.0)),
            )
            for row in rows
        ]

    @staticmethod
    def _weighted_mean(values: list[tuple[float, float]]) -> float:
        total_weight = sum(weight for _, weight in values if weight > 0)
        if total_weight == 0:
            return 0.0
        return sum(value * weight for value, weight in values if weight > 0) / total_weight

    @staticmethod
    def _bounded(value: float) -> float:
        return max(-1.0, min(1.0, value))
