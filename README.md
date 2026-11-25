# ConsumerSim: Salience-Driven Generative Social Simulation for Consumer Confidence

Official reproduction codebase, frozen evaluation outputs, and public forecast interface for **ConsumerSim** across the **United States (Michigan ICS)**, **EU27 (Eurostat BS-CSMCI)**, and **Japan (Cabinet Office ESRI CCI)** from **July 2014 to September 2026** (147 aligned months, zero aggregate $\lambda$ blending).

> **Note on prospective longitudinal panel data:** Primary three-wave U.S. human panel records are withheld pending publication and are not included in this release; de-identified replication tables will be deposited upon publication.

---

## Repository Layout

```text
.
├── consumersim/               # Core framework (Persona Substrate, Uniform Exposure Layer,
│                              # Situational Signal Field, Contrastive Elicitation, Household Inertia)
├── experiments/
│   ├── eu_jp/                 # EU27 (27 member states, N=1,350/mo) & Japan (8 regions, N=1,200/mo)
│   ├── us_baselines/          # Information-matched ML (+news embeddings), Direct LLM, open-loop,
│   │                          # ablations, subgroups, scale curves, downstream activity, weekly paths
│   ├── robustness/            # 4 LLM families, 4 prompt templates, 3 random seeds
│   ├── mechanisms/            # 3-tier exposure decomposition, negative controls, orthogonal
│   │                          # 144-persona counterfactuals, observable-input exogenous salience
│   └── memory_audit/          # Undated-snapshot date/ICS recall probe & error-inheritance audit
├── results/                   # Frozen reproducible CSV/JSON outputs (zero API cost)
├── scripts/
│   ├── reproduce_tables.py    # One-command deterministic verification of Tables 1–3, E1–E2, G4–G7
│   ├── build_site_data_v4.py  # Generates site/data/consumersim_site_data.csv (legacy + v4)
│   └── refresh_site_data.py   # Validates and normalizes site CSV for GitHub Pages CI
├── site/                      # Static forecast dashboard deployed to GitHub Pages
├── live_pipeline/             # Monthly push-sync helper for live inference servers (e.g., sociox)
├── consumersim_mcp_proxy/     # MCP server & local web preview interface
└── .github/workflows/         # Automated GitHub Pages deployment (main + live-data overlay)
```

---

## Quick Start: Zero-API Verification of All US / EU27 / Japan Tables

Reproduce **Table 1** (US / EU27 / Japan, 147-month full sample and 2023–2026 out-of-sample), **Table 2** (top-10 U.S. shocks), **Table 3** (component ablations), **Tables E1–E2** (multi-month open-loop rollout and memory audit), and **Tables G4–G7** (robustness, three-tier decomposition, orthogonal counterfactuals, and exogenous salience) deterministically without API calls:

```bash
pip install -r requirements.txt
python scripts/reproduce_tables.py
```

---

## Running the Full Simulation & Data-Fetch Pipelines

Third-party raw datasets (University of Michigan Surveys of Consumers microdata, *New York Times* Archive articles, and *The Guardian* Open Platform articles) are subject to provider terms of use and are not redistributed directly. To re-run simulations from scratch:

1. Copy `.env.example` to `.env` and populate `OPENAI_API_KEY` (plus `OPENAI_BASE_URL` if using an OpenAI-compatible gateway), `FRED_API_KEY`, `NYT_API_KEY`, and `GUARDIAN_API_KEY`.
2. Download official Michigan microdata from https://data.sca.isr.umich.edu into `CONSUMERSIM_DATA_DIR`.
3. Fetch EU27 and Japan official targets, macro indicators, and news:
   ```bash
   python experiments/eu_jp/fetch_eu_jp_data.py
   python experiments/eu_jp/fetch_eu_jp_supplement.py
   python experiments/eu_jp/fetch_guardian_2014_2019.py
   python experiments/eu_jp/run_eu_jp.py
   ```
4. Run U.S. strong baselines, robustness, mechanism tests, and memory audit:
   ```bash
   python experiments/us_baselines/run_strong_baselines.py
   python experiments/robustness/run_robustness.py
   python experiments/mechanisms/run_mechanisms.py
   python experiments/mechanisms/rerun_5c_ortho.py
   python experiments/memory_audit/run_date_probe.py
   python experiments/memory_audit/analyze_date_probe.py
   ```

---

## Forecast Website, Live Monthly Sync & MCP Interface

- **Static dashboard (`site/`)**: Preserves the original ConsumerSim forecast site layout while storing both the legacy (`?as_of=2026-07-02`) and updated v4 (`2026-09-30`, 147-month zero-$\lambda$ US/EU27/Japan) snapshots in `site/data/consumersim_site_data.csv`. Rebuild locally via `python scripts/build_site_data_v4.py`.
- **Live monthly sync**: See `live_pipeline/README.md` for running monthly inference on an external server (`sociox`) and pushing validated CSV updates to the `live-data` branch for automatic GitHub Pages redeployment.
- **MCP interface (`consumersim_mcp_proxy/`)**: Install with `pip install -e .` and run `consumersim-mcp` or `consumersim-web`.
