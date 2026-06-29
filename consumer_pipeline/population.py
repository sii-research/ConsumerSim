from __future__ import annotations

import csv
import random
from pathlib import Path

from .models import Consumer, RegionProfile


POPULATION_COLUMNS = (
    "consumer_id", "age_group", "income_group", "education_group", "location_group", "weight"
)


class PopulationSampler:
    """Loads a harmonized population file or creates a reproducible synthetic sample."""

    def __init__(self, profile: RegionProfile, seed: int) -> None:
        self.profile = profile
        self.rng = random.Random(seed)

    def sample(self, size: int, source_path: Path | None) -> list[Consumer]:
        if size <= 0:
            raise ValueError("population.size must be positive")
        population = self._load_csv(source_path) if source_path else self._synthetic(size)
        if len(population) < size:
            raise ValueError(f"Population source has {len(population)} rows, fewer than requested {size}")
        if len(population) == size:
            return population
        return self._weighted_without_replacement(population, size)

    def select_core(self, population: list[Consumer], ratio: float, group_fields: tuple[str, ...]) -> list[Consumer]:
        target = max(1, round(len(population) * ratio))
        groups: dict[tuple[str, ...], list[Consumer]] = {}
        for consumer in population:
            groups.setdefault(consumer.group_key(group_fields), []).append(consumer)

        selected: list[Consumer] = []
        for key in sorted(groups):
            members = groups[key]
            self.rng.shuffle(members)
            group_count = max(1, round(len(members) * ratio))
            selected.extend(members[:group_count])

        if len(selected) > target:
            selected = self.rng.sample(selected, target)
        elif len(selected) < target:
            chosen = {consumer.consumer_id for consumer in selected}
            remaining = [consumer for consumer in population if consumer.consumer_id not in chosen]
            selected.extend(self.rng.sample(remaining, target - len(selected)))

        for consumer in selected:
            consumer.is_core = True
        return selected

    def _load_csv(self, path: Path) -> list[Consumer]:
        if not path.exists():
            raise FileNotFoundError(f"Population source not found: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = set(POPULATION_COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Population CSV missing columns: {', '.join(sorted(missing))}")
            rows = [self._from_row(row) for row in reader]
        return rows

    def _from_row(self, row: dict[str, str]) -> Consumer:
        weight = float(row.get("weight") or 1.0)
        if weight < 0:
            raise ValueError(f"Population weight must be non-negative for {row['consumer_id']!r}")
        allowed = {
            "age_group": self.profile.age_distribution,
            "income_group": self.profile.income_distribution,
            "education_group": self.profile.education_distribution,
            "location_group": self.profile.location_distribution,
        }
        for field, choices in allowed.items():
            if row[field] not in choices:
                raise ValueError(f"Invalid {field} {row[field]!r} for region {self.profile.region.value}")
        return Consumer(
            consumer_id=row["consumer_id"],
            region=self.profile.region,
            age_group=row["age_group"],
            income_group=row["income_group"],
            education_group=row["education_group"],
            location_group=row["location_group"],
            weight=weight,
        )

    def _synthetic(self, size: int) -> list[Consumer]:
        return [
            Consumer(
                consumer_id=f"{self.profile.region.value.lower()}-{index + 1:06d}",
                region=self.profile.region,
                age_group=self._choice(self.profile.age_distribution),
                income_group=self._choice(self.profile.income_distribution),
                education_group=self._choice(self.profile.education_distribution),
                location_group=self._choice(self.profile.location_distribution),
            )
            for index in range(size)
        ]

    def _choice(self, distribution: dict[str, float]) -> str:
        labels = list(distribution)
        return self.rng.choices(labels, weights=[distribution[label] for label in labels], k=1)[0]

    def _weighted_without_replacement(self, population: list[Consumer], size: int) -> list[Consumer]:
        pool = list(population)
        selected: list[Consumer] = []
        for _ in range(size):
            weights = [max(consumer.weight, 0.0) for consumer in pool]
            if sum(weights) == 0:
                weights = [1.0] * len(pool)
            chosen = self.rng.choices(range(len(pool)), weights=weights, k=1)[0]
            selected.append(pool.pop(chosen))
        return selected
