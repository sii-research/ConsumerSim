from __future__ import annotations

from .models import QUESTIONS, Region, RegionProfile


def _priors(positive: float, neutral: float) -> dict[str, dict[str, float]]:
    return {
        question: {
            "positive": positive,
            "neutral": neutral,
            "negative": 1.0 - positive - neutral,
        }
        for question in QUESTIONS
    }


REGION_PROFILES: dict[Region, RegionProfile] = {
    Region.JP: RegionProfile(
        region=Region.JP,
        language="ja",
        locations=("Hokkaido-Tohoku", "Kanto", "Chubu", "Kansai", "Chugoku-Shikoku", "Kyushu-Okinawa"),
        age_distribution={"18-34": 0.22, "35-54": 0.35, "55+": 0.43},
        income_distribution={"low": 0.34, "middle": 0.33, "high": 0.33},
        education_distribution={"secondary": 0.42, "tertiary": 0.58},
        location_distribution={
            "Hokkaido-Tohoku": 0.11, "Kanto": 0.35, "Chubu": 0.17,
            "Kansai": 0.17, "Chugoku-Shikoku": 0.09, "Kyushu-Okinawa": 0.11,
        },
        question_priors=_priors(0.25, 0.47),
    ),
    Region.EU: RegionProfile(
        region=Region.EU,
        language="en",
        locations=("Northern", "Western", "Southern", "Central-Eastern"),
        age_distribution={"18-34": 0.25, "35-54": 0.36, "55+": 0.39},
        income_distribution={"low": 0.34, "middle": 0.33, "high": 0.33},
        education_distribution={"secondary": 0.52, "tertiary": 0.48},
        location_distribution={"Northern": 0.13, "Western": 0.35, "Southern": 0.25, "Central-Eastern": 0.27},
        question_priors=_priors(0.27, 0.43),
    ),
    Region.US: RegionProfile(
        region=Region.US,
        language="en",
        locations=("Northeast", "Midwest", "South", "West"),
        age_distribution={"18-34": 0.29, "35-54": 0.34, "55+": 0.37},
        income_distribution={"low": 0.34, "middle": 0.33, "high": 0.33},
        education_distribution={"secondary": 0.56, "tertiary": 0.44},
        location_distribution={"Northeast": 0.17, "Midwest": 0.21, "South": 0.38, "West": 0.24},
        question_priors=_priors(0.29, 0.42),
    ),
}


def get_region_profile(value: str) -> RegionProfile:
    try:
        return REGION_PROFILES[Region(value.upper())]
    except ValueError as exc:
        raise ValueError(f"Unsupported region {value!r}; expected JP, EU, or US") from exc
