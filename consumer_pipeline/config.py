from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_dir"] = str(config_path.parent)
    _validate(config)
    return config


def resolve_path(config: dict[str, Any], value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return path.resolve()


def read_secret_env(config: dict[str, Any], key: str) -> str | None:
    variable = config.get("credentials", {}).get(key)
    return os.getenv(variable) if variable else None


def _validate(config: dict[str, Any]) -> None:
    required = ("region", "population", "information", "prediction", "bayesian", "correction", "output")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")
    if str(config["region"]).upper() not in {"JP", "EU", "US"}:
        raise ValueError("region must be JP, EU, or US")
    ratio = float(config["population"].get("core_ratio", 0.1))
    if not 0.0 < ratio <= 1.0:
        raise ValueError("population.core_ratio must be in (0, 1]")
    for name, variable in config.get("credentials", {}).items():
        if not isinstance(variable, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", variable):
            raise ValueError(
                f"credentials.{name} must be an environment variable name, never a literal credential"
            )
