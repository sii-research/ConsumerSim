from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendSettings:
    base_url: str
    api_key: str | None = None
    api_key_header: str = "Authorization"
    api_key_scheme: str = "Bearer"
    forecast_path: str = "/forecast"
    times_path: str = "/forecast/times"
    site_data_path: str = "/site-data"
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "BackendSettings":
        base_url = os.environ.get("CONSUMERSIM_API_BASE_URL", "").strip()
        if not base_url:
            raise RuntimeError("CONSUMERSIM_API_BASE_URL is required")
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=_blank_to_none(os.environ.get("CONSUMERSIM_API_KEY")),
            api_key_header=_blank_to_none(os.environ.get("CONSUMERSIM_API_KEY_HEADER")) or "Authorization",
            api_key_scheme=_blank_to_none(os.environ.get("CONSUMERSIM_API_KEY_SCHEME")) or "Bearer",
            forecast_path=_path(_blank_to_none(os.environ.get("CONSUMERSIM_FORECAST_PATH")) or "/forecast"),
            times_path=_path(_blank_to_none(os.environ.get("CONSUMERSIM_TIMES_PATH")) or "/forecast/times"),
            site_data_path=_path(_blank_to_none(os.environ.get("CONSUMERSIM_SITE_DATA_PATH")) or "/site-data"),
            timeout_seconds=float(_blank_to_none(os.environ.get("CONSUMERSIM_TIMEOUT_SECONDS")) or "30"),
        )

    def url_for(self, path: str) -> str:
        return f"{self.base_url}{_path(path)}"

    def auth_headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        if self.api_key_header.lower() == "authorization" and self.api_key_scheme:
            return {self.api_key_header: f"{self.api_key_scheme} {self.api_key}"}
        return {self.api_key_header: self.api_key}


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _path(value: str) -> str:
    stripped = value.strip()
    return stripped if stripped.startswith("/") else f"/{stripped}"
