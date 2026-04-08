# Benchmarks

This directory contains an initial benchmark suite for resolved Polymarket markets.

The current benchmark package targets three tasks:

1. `multi-horizon terminal forecasting`
   Predict the final binary resolution of a market from the state observed `24h`, `72h`, or `168h` before `end_date`.

2. `trustworthiness / selective prediction`
   Score whether the market probability is likely to be reliable at `24h` before resolution and evaluate abstention by coverage.

3. `large repricing prediction`
   Predict whether a market probability will move by at least a fixed threshold over the next `24h`.

## Files

- `benchmark_utils.py`
  Shared data loading, feature extraction, dataset builders, and time-split utilities.

- `run_benchmarks.py`
  Reproducible benchmark runner that exports datasets, per-fold metrics, aggregated summaries, and a markdown report.

- `01_multi_horizon_terminal_forecasting.ipynb`
- `02_trustworthiness_selective_prediction.ipynb`
- `03_large_repricing_prediction.ipynb`
- `04_cross_market_information_uptake.ipynb`
- `05_hybrid_crypto_terminal_forecasting.ipynb`
  Notebook versions for exploratory analysis and visualization.

## Evaluation Protocol

The benchmark runner uses:

- resolved markets from `db/resolved_probability_dataset.sqlite`
- strict out-of-time rolling splits
- a snapshot-staleness filter for terminal/trust tasks
- per-task baseline suites

Current defaults:

- domain: `geopolitics`
- terminal horizons: `24h`, `72h`, `168h`
- trust horizon: `24h`
- repricing horizon: `24h`
- repricing threshold: absolute move `>= 0.15`

## Running

Using the existing conda environment:

```bash
conda activate polymarket
python benchmarks/run_benchmarks.py --domain geopolitics
```

Outputs are written under:

```text
benchmarks/results/<domain>/
```

Artifacts include:

- `terminal_dataset.csv`
- `repricing_dataset.csv`
- `terminal_metrics.csv`
- `terminal_summary.csv`
- `trust_metrics.csv`
- `trust_summary.csv`
- `repricing_metrics.csv`
- `repricing_summary.csv`
- `benchmark_config.json`
- `benchmark_report.md`
- `summary.json`

## Notes

- The benchmark is currently strongest on terminal forecasting and repricing.
- The trustworthiness task is intentionally difficult and currently uses a compact feature set derived only from market history and metadata.
- This package is designed to be extended with external covariates, text evidence, and cross-market structure without changing the task definitions.
- The two crypto notebooks are exploratory extensions: they show how to align external `BTC/ETH` covariates with the benchmark tasks before promoting those experiments into the main runner.
