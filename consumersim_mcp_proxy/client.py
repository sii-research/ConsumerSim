from __future__ import annotations

from typing import Any

import httpx

from .settings import BackendSettings


class ConsumerSimBackend:
    def __init__(self, settings: BackendSettings | None = None) -> None:
        self.settings = settings or BackendSettings.from_env()

    async def forecast_lookup(self, region: str, month: str, week: str | int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"region": region, "month": month}
        if week is not None:
            payload["week"] = week
        return await self._post_json(self.settings.forecast_path, payload)

    async def forecast_times(self, region: str | None = None) -> dict[str, Any]:
        params = {"region": region} if region else None
        return await self._get_json(self.settings.times_path, params=params)

    async def site_data_csv(self) -> str:
        async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
            response = await client.get(
                self.settings.url_for(self.settings.site_data_path),
                headers=self.settings.auth_headers(),
            )
            response.raise_for_status()
            return response.text

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
            response = await client.post(
                self.settings.url_for(path),
                json=payload,
                headers=self.settings.auth_headers(),
            )
            response.raise_for_status()
            return response.json()

    async def _get_json(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
            response = await client.get(
                self.settings.url_for(path),
                params=params,
                headers=self.settings.auth_headers(),
            )
            response.raise_for_status()
            return response.json()
