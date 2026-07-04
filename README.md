# ConsumerSim Forecast Interface

ConsumerSim provides consumer confidence forecasts for three markets:

- `US`
- `EU27`
- `JP`

Users can access the forecasts in two ways:

- View the public forecast website hosted from this repository.
- Connect to the MCP server and ask for forecasts by region, month, and optional week.

This repository is an interface package only. It does not publish the private
forecasting pipeline, model prompts, data refresh logic, source API keys, or
private datasets.

## What You Can Ask For

Use `forecast_lookup` when you need a forecast value.

Monthly forecast:

```json
{
  "region": "US",
  "month": "2026-07"
}
```

Weekly forecast:

```json
{
  "region": "EU27",
  "month": "2026-07",
  "week": 1
}
```

Supported inputs:

- `region`: `US`, `EU27`, or `JP`
- `month`: target month in `YYYY-MM` format
- `week`: optional week number within that month, such as `1`, `2`, `3`, or `4`

Use `forecast_times` to see which forecast periods are available.

```json
{
  "region": "JP"
}
```

## Typical Response

`forecast_lookup` returns a forecast snapshot like this:

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

The exact fields may vary by backend version, but the response is designed to
include the requested region and time, the forecast value, a confidence band,
and a short interpretation.

## Run As An MCP Server

Install the package:

```powershell
python -m pip install -e .
```

Configure the private backend endpoint:

```powershell
$env:CONSUMERSIM_API_BASE_URL = "https://your-consumersim-backend.example.com"
$env:CONSUMERSIM_API_KEY = "<your access token>"
```

Start the MCP server:

```powershell
consumersim-mcp
```

The MCP server exposes:

- `forecast_lookup`
- `forecast_times`

It forwards requests to the configured ConsumerSim backend and returns the
backend result to the MCP client.

## Run The Website Locally

The website is a static forecast dashboard under `site/`.

For a local preview backed by the private API:

```powershell
python -m pip install -e .
$env:CONSUMERSIM_API_BASE_URL = "https://your-consumersim-backend.example.com"
$env:CONSUMERSIM_API_KEY = "<your access token>"
consumersim-web
```

Open:

```text
http://127.0.0.1:4173
```

The local web bridge serves the static site and proxies `/api/site-data` to the
private backend. This keeps backend credentials out of browser JavaScript.

## Public Website Deployment

The recommended public deployment is GitHub Pages.

The included workflow:

```text
.github/workflows/refresh-site.yml
```

does the following:

1. Calls the private backend for the latest forecast CSV.
2. Validates the CSV structure.
3. Writes the result to `site/data/consumersim_site_data.csv`.
4. Commits the updated CSV when it changes.
5. Deploys the `site/` directory to GitHub Pages.

Repository setup:

- Set Pages source to `GitHub Actions`.
- Allow GitHub Actions `Read and write permissions`.
- Add `CONSUMERSIM_API_BASE_URL` as a repository variable.
- Add `CONSUMERSIM_API_KEY` as a repository secret if the backend requires a token.

Do not put source data API keys or model API keys in GitHub. Those belong only
on the private backend server.

## Backend Settings

Required:

- `CONSUMERSIM_API_BASE_URL`

Usually required:

- `CONSUMERSIM_API_KEY`

Optional:

- `CONSUMERSIM_FORECAST_PATH`, default `/forecast`
- `CONSUMERSIM_TIMES_PATH`, default `/forecast/times`
- `CONSUMERSIM_SITE_DATA_PATH`, default `/site-data`
- `CONSUMERSIM_API_KEY_HEADER`, default `Authorization`
- `CONSUMERSIM_API_KEY_SCHEME`, default `Bearer`
- `CONSUMERSIM_TIMEOUT_SECONDS`, default `30`

## Backend Contract

The proxy calls these backend routes:

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

Expected `GET /site-data` response:

```text
as_of,record_type,region,...
2026-07-04,monthly_prediction,us,...
```

## Repository Boundary

This public repository should contain only:

- MCP proxy code
- public website assets
- GitHub Pages refresh workflow
- tests for the public interface
- examples and documentation for users

Do not commit:

- private forecasting pipeline code
- model prompts or internal simulation logic
- private data refresh scripts
- source API keys
- LLM API keys
- private datasets

The private backend can update forecasts without exposing the internal
ConsumerSim implementation.
