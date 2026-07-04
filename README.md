# ConsumerSim MCP Proxy

This repository is the public ConsumerSim interface package. It does not contain
the ConsumerSim forecasting pipeline, model code, training logic, data refresh
scripts, or private datasets.

It contains:

- A thin MCP server that exposes forecast tools.
- A GitHub Actions refresh job that writes static forecast data for GitHub Pages.
- An optional local web bridge for previewing the website against a private backend.
- Static website assets copied from the public forecast site.

All forecast refreshes are pulled from a private HTTPS backend configured through
environment variables. GitHub Pages serves only static files.

## Tools

`forecast_lookup`

```json
{
  "region": "US",
  "month": "2026-07",
  "week": 1
}
```

If `week` is omitted, the backend should return the monthly forecast.

`forecast_times`

```json
{
  "region": "EU27"
}
```

## Backend Contract

By default the proxy calls:

- `POST {CONSUMERSIM_API_BASE_URL}/forecast`
- `GET {CONSUMERSIM_API_BASE_URL}/forecast/times`
- `GET {CONSUMERSIM_API_BASE_URL}/site-data`

Expected `POST /forecast` request:

```json
{
  "region": "EU27",
  "month": "2026-06",
  "week": 4
}
```

Expected response:

```json
{
  "region": "EU27",
  "cadence": "weekly",
  "requested_month": "2026-06",
  "requested_week": "Jun W4",
  "target_month": "2026-07",
  "target_period": "Jul-26",
  "week_label": "Jun W4",
  "as_of": "2026-07-04",
  "forecast": -13.32,
  "interval_low": -14.71,
  "interval_high": -12.39,
  "signal": "Softening signal",
  "interpretation": "Weekly nowcast through 2026-06-27 from the ConsumerSim pipeline."
}
```

Expected `GET /site-data` response, used by GitHub Actions:

```text
as_of,record_type,region,...
2026-07-04,monthly_prediction,us,...
```

GitHub Actions writes this CSV to:

```text
site/data/consumersim_site_data.csv
```

The website reads that static file from GitHub Pages. The private backend can
refresh forecasts without publishing internal logic.

## Configuration

Copy `.env.example` into your deployment environment and set:

```powershell
$env:CONSUMERSIM_API_BASE_URL = "https://your-private-consumersim.example.com"
$env:CONSUMERSIM_API_KEY = "<optional token>"
```

Optional overrides:

- `CONSUMERSIM_FORECAST_PATH`
- `CONSUMERSIM_TIMES_PATH`
- `CONSUMERSIM_SITE_DATA_PATH`
- `CONSUMERSIM_API_KEY_HEADER`
- `CONSUMERSIM_API_KEY_SCHEME`
- `CONSUMERSIM_TIMEOUT_SECONDS`

## Run The MCP Server

```powershell
python -m pip install -e .
consumersim-mcp
```

## Run The Website Bridge

```powershell
python -m pip install -e .
consumersim-web
```

Then open:

```text
http://127.0.0.1:4173
```

The web bridge serves `site/` and proxies website data requests to the private
backend. A browser cannot call a stdio MCP server directly, so the website uses
the HTTP bridge, which calls the same backend proxy implementation as the MCP
tools.

## GitHub Pages Deployment

The recommended public website deployment is GitHub Pages with the included
workflow:

```text
.github/workflows/refresh-site.yml
```

The workflow:

1. installs this package,
2. downloads the latest CSV from the private backend,
3. validates and writes `site/data/consumersim_site_data.csv`,
4. commits that data file back to the repo when it changed,
5. deploys `site/` to GitHub Pages.

Configure repository settings:

```text
Secret or variable: CONSUMERSIM_API_BASE_URL
Optional secret:    CONSUMERSIM_API_KEY
Optional variable:  CONSUMERSIM_SITE_DATA_PATH
```

In GitHub, set Pages source to "GitHub Actions".

For pure static hosting, the browser cannot hide API keys and cannot call stdio
MCP directly. This is why the workflow refreshes a static CSV ahead of time.
If you ever need a browser to call a public read-only endpoint directly, configure:

```javascript
window.CONSUMERSIM_SITE_DATA_URL = "https://your-private-consumersim.example.com/site-data";
```

Use `site/site-config.example.js` as the template for that direct-browser mode.

## What Not To Commit Here

Do not add:

- `consumer_pipeline/`
- model or prompt logic
- private forecast-generation scripts
- private configs, examples, or outputs
- generated CSV data that should remain server-side

Keep this repo as an interface shell only.
