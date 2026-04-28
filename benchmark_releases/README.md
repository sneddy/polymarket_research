# Benchmark Releases

This directory stores frozen local benchmark artifacts.

Expected layout:

```text
benchmark_releases/
  polymarket/
    terminal/v1/
    decisiveness/v1/
    repricing/v1/
    reports/v1/
  kalshi/
    terminal/v1/
    decisiveness/v1/
    repricing/v1/
    reports/v1/
```

Each task bundle contains:

- `manifest.json`
- `examples.parquet` with leakage-safe observable input rows
- `market_timeseries.parquet`
- `targets.parquet` with labels and auxiliary target fields
- `README.md`

Each source report directory contains:

- `release_report.json`
- `release_report.md`

Internal build caches such as raw snapshots and `CanonicalDataset` parquet saves belong under `internal_artifacts/`, not here.
