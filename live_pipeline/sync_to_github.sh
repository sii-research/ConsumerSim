#!/usr/bin/env bash
# Live push-sync helper to run on sociox after US / EU27 / JP extrapolation finishes.
# Pushes ONLY site/data/consumersim_site_data.csv to the dedicated `live-data` branch
# so `main` stays clean code while GitHub Actions immediately redeploys GitHub Pages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_URL="${CONSUMERSIM_GITHUB_REPO:-https://github.com/sii-research/ConsumerSim.git}"
CSV_PATH="${1:-$REPO_ROOT/site/data/consumersim_site_data.csv}"
WORK_DIR="${TMPDIR:-/tmp}/consumersim_live_data_sync"
AS_OF_TAG="$(date -u +%Y-%m-%d)"

if [ ! -f "$CSV_PATH" ]; then
  echo "ERROR: CSV not found at $CSV_PATH" >&2
  exit 1
fi

# 1. Validate schema locally before pushing
python3 "$REPO_ROOT/scripts/refresh_site_data.py" --input "$CSV_PATH" --output "$CSV_PATH"

# 2. Push ONLY the validated CSV to the lightweight `live-data` branch
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
git -C "$WORK_DIR" init -q
git -C "$WORK_DIR" config user.name "${CONSUMERSIM_GIT_USER:-ConsumerSim Live Bot}"
git -C "$WORK_DIR" config user.email "${CONSUMERSIM_GIT_EMAIL:-consumersim-bot@users.noreply.github.com}"
git -C "$WORK_DIR" remote add origin "$REPO_URL"

if git -C "$WORK_DIR" ls-remote --exit-code --heads origin live-data >/dev/null 2>&1; then
  git -C "$WORK_DIR" fetch --depth 1 origin live-data
  git -C "$WORK_DIR" checkout -q -b live-data origin/live-data
else
  git -C "$WORK_DIR" checkout -q --orphan live-data
fi

mkdir -p "$WORK_DIR/site/data"
cp "$CSV_PATH" "$WORK_DIR/site/data/consumersim_site_data.csv"

git -C "$WORK_DIR" add site/data/consumersim_site_data.csv
if git -C "$WORK_DIR" diff --cached --quiet; then
  echo "No change in consumersim_site_data.csv; skipping push."
  exit 0
fi

git -C "$WORK_DIR" commit -q -m "Update live ConsumerSim forecast data (${AS_OF_TAG})"
git -C "$WORK_DIR" push -q origin live-data
echo "Pushed updated site/data/consumersim_site_data.csv to origin/live-data."

# 3. Trigger GitHub Pages deploy on main (overlays origin/live-data CSV while satisfying github-pages main-only branch policy)
REPO_SLUG="$(echo "$REPO_URL" | sed -E 's#.*github\.com[:/]([^/]+/[^/.]+)(\.git)?$#\1#')"
if command -v gh >/dev/null 2>&1; then
  gh workflow run refresh-site.yml -R "$REPO_SLUG" --ref main
  echo "Triggered refresh-site.yml on ${REPO_SLUG} (main) to redeploy GitHub Pages."
else
  echo "NOTE: gh CLI not found; GitHub Pages will redeploy on next scheduled workflow run or manual dispatch."
fi
