from __future__ import annotations

from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from httpx import HTTPError
from pydantic import BaseModel

from .client import ConsumerSimBackend


ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / "site"


class ForecastRequest(BaseModel):
    region: str
    month: str
    week: str | int | None = None


app = FastAPI(title="ConsumerSim Public Web Bridge")


@app.post("/api/forecast")
async def api_forecast_lookup(request: ForecastRequest) -> dict[str, Any]:
    try:
        return await ConsumerSimBackend().forecast_lookup(request.region, request.month, request.week)
    except (HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/forecast/times")
async def api_forecast_times(region: str | None = None) -> dict[str, Any]:
    try:
        return await ConsumerSimBackend().forecast_times(region)
    except (HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/site-data")
async def api_site_data() -> PlainTextResponse:
    try:
        csv_text = await ConsumerSimBackend().site_data_csv()
    except (HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return PlainTextResponse(csv_text, media_type="text/csv; charset=utf-8")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(SITE_DIR / "index.html")


app.mount("/", StaticFiles(directory=SITE_DIR), name="site")


def main() -> None:
    uvicorn.run("consumersim_mcp_proxy.web:app", host="127.0.0.1", port=4173, reload=False)


if __name__ == "__main__":
    main()
