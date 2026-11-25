# Live Monthly Inference & GitHub Pages Sync (`sociox` → GitHub Pages)

## Architecture Overview

Instead of exposing inbound SSH ports on `sociox` for hourly GitHub Actions polling,
`sociox` runs the scheduled monthly inference locally (where FRED, Eurostat, Cabinet
Office/ESRI, NYT/Guardian APIs, and LLM endpoints are reachable) and **pushes only
the validated static CSV** to a dedicated `live-data` branch on GitHub:

```text
[sociox cron / systemd timer (monthly)]
  1. Fetch latest official US / EU27 / JP releases + real-time macro & news
  2. Run frozen v4 ConsumerSim cohort inference (US / EU27 / JP)
  3. Rebuild site/data/consumersim_site_data.csv (keeping legacy + v4 history)
  4. Validate CSV via scripts/refresh_site_data.py
  5. Run live_pipeline/sync_to_github.sh -> pushes ONLY CSV to origin/live-data
                         │
                         ▼ (automatic GitHub Actions trigger, ~30s)
[.github/workflows/refresh-site.yml]
  - Checks out `main` (site HTML/CSS/JS)
  - Overlays latest `site/data/consumersim_site_data.csv` from `origin/live-data`
  - Runs schema validation + pytest
  - Deploys static bundle to https://sii-research.github.io/ConsumerSim/
```

## Example `sociox` Cron Entry

```cron
# Run on the 1st of every month at 08:00 UTC (after monthly macro/survey cutoffs)
0 8 1 * * cd /path/to/ConsumerSim && /usr/bin/python3 scripts/build_site_data_v4.py && bash live_pipeline/sync_to_github.sh >> /var/log/consumersim_live.log 2>&1
```

## Authentication on `sociox`

- Reuse an already-authenticated `gh` CLI session (`gh auth setup-git`) or configure a
  fine-grained GitHub PAT scoped **only** to `contents: write` on `sii-research/ConsumerSim`.
- No API keys, raw microdata, or private server IPs are ever committed to GitHub.
