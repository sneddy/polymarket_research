---
license: cc-by-4.0
task_categories:
  - time-series-forecasting
  - tabular-classification
tags:
  - prediction-markets
  - polymarket
  - forecasting
  - benchmark
pretty_name: Polymarket Terminal 24h Benchmark
---

# Polymarket Terminal 24h Benchmark

Frozen terminal-outcome benchmark built from Polymarket market histories.

## Summary
- Release: `polymarket-terminal-24h`
- Examples: 10156
- Markets: 10156
- Market-timeseries rows: 144202396
- Splits: test=2238, train=7918
- Horizons (hours): 24
- Split policy: `end_date` via empirical 0.8 quantile
- Resolved split timestamp: `2026-01-31T00:00:00+00:00`

## Protocol
Only rows from market_timeseries with matching market_id and timestamp_utc <= cutoff_timestamp_utc are admissible for a given example.

## Files
- `examples.parquet`: frozen example manifest with stable `example_id` values and no labels
- `targets.parquet`: held-out targets keyed by `example_id`
- `market_timeseries.parquet`: normalized market-level probability histories keyed by `market_id`
- `manifest.json`: benchmark metadata and build config

## Build Manifest
```json
{
  "config": {
    "horizons_hours": [
      24
    ],
    "max_snapshot_staleness_hours": 12.0,
    "max_snapshot_staleness_hours_by_horizon": {
      "168": 24.0,
      "24": 12.0,
      "72": 18.0
    },
    "show_progress": true,
    "split_on": "end_date",
    "split_timestamp_utc": null,
    "train_fraction": 0.8
  },
  "example_columns": [
    "market_id",
    "market_slug",
    "question",
    "created_at",
    "end_date",
    "platform_category",
    "research_category",
    "family_id",
    "horizon_hours",
    "cutoff_timestamp_utc",
    "label",
    "split"
  ],
  "horizons_hours": [
    24
  ],
  "market_timeseries_columns": [
    "market_id",
    "timestamp_utc",
    "yes_probability"
  ],
  "market_timeseries_rows": 144202396,
  "markets": 10156,
  "name": "terminal_benchmark",
  "observable_information": "target market metadata and market-level probability history up to cutoff",
  "observable_prefix_rule": "Only rows from market_timeseries with matching market_id and timestamp_utc <= cutoff_timestamp_utc are admissible for a given example.",
  "protocol_example_columns": [
    "example_id",
    "market_id",
    "market_slug",
    "question",
    "created_at",
    "end_date",
    "platform_category",
    "research_category",
    "family_id",
    "horizon_hours",
    "cutoff_timestamp_utc",
    "split"
  ],
  "protocol_version": "terminal_hf_bundle_v1",
  "release_name": "polymarket-terminal-24h",
  "resolved_split_timestamp_utc": "2026-01-31T00:00:00+00:00",
  "rows": 10156,
  "split_counts": {
    "test": 2238,
    "train": 7918
  },
  "target_columns": [
    "example_id",
    "market_id",
    "horizon_hours",
    "cutoff_timestamp_utc",
    "label",
    "split"
  ],
  "target_type": "final resolved outcome",
  "targets_rows": 10156,
  "task": "terminal_outcome_prediction"
}
```
