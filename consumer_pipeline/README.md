# `consumer_pipeline`

This Python package implements the shared monthly ConsumerSim core pipeline for JP, EU, and US. It excludes visualizations, method comparisons, ablation studies, and downstream tasks.

## Modules

- `orchestrator.py`: executes the complete pipeline in order.
- `population.py`: loads or synthesizes a population and draws a stratified core sample.
- `information.py`: aggregates normalized news and indicators available by the `as-of` date.
- `prediction.py`: predicts the five survey responses for core consumers.
- `bayesian.py`: performs grouped Dirichlet updating, population expansion, and weighted aggregation.
- `debias.py`: applies a shrunk residual from the immediately preceding calendar month.
- `regions.py`: defines regional profiles and priors for JP, EU, and US.
- `models.py`: defines shared data structures and field names.
- `config.py`: validates configuration, resolves paths, and reads credentials from environment variables.
- `output.py`: writes the summary JSON and population-response CSV.
- `cli.py`: provides the command-line entry point.

## Main Flow

```text
Population sampling
  -> News and other information indicators
  -> Core-sample prediction
  -> Bayesian updating and population aggregation
  -> Previous-month residual debiasing
  -> Standard output artifacts
```

`debias.py` reads only `predicted` and `actual` values from the calendar month immediately preceding the target. If the exact t-1 record is unavailable, no correction is applied. Future outcomes and arbitrary older months are never substituted.

## Run

From the `pipeline/` directory:

```powershell
python -m consumer_pipeline.cli --config configs/us.yaml --month 2026-06 --as-of 2026-06-20
```

See the parent [README](../README.md) for configuration details, input contracts, Bayesian formulas, and credential-handling rules.
