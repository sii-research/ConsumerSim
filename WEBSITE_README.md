# ConsumerSim Confidence Forecasts Website

Static website for weekly-updated consumer confidence forecasts across:

- US: North America representative market
- EU27: Europe representative market
- Japan: Asia representative market

## Preview

Open `index.html` directly in a browser, or serve this folder with any static server.

```powershell
python -m http.server 4173
```

Then visit `http://localhost:4173`.

## Weekly Update

The website reads all displayed page data from one structured data file:

```text
data/consumersim_site_data.csv
```

For each weekly refresh:

1. Run the latest ConsumerSim forecasting jobs.
2. Append new rows to `data/consumersim_site_data.csv` with a new `as_of` value.
3. Update the `meta` rows for `generatedAt`, `updateCadence`, and `nextUpdate`.
4. Redeploy the static folder.

The frontend automatically selects the latest `as_of` value. To preview an older snapshot, open:

```text
http://localhost:4173/?as_of=2026-06
```

Current source inputs are the local ConsumerSim result files under `results_us`, `results_eu`, and `results_jp`, plus forecast rows added directly to the site data file when a current month has not yet been released.
