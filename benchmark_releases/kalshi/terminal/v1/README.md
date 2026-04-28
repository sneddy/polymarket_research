# Kalshi Terminal Benchmark

Source: `kalshi`
Release: `kalshi-terminal-24h-72h-168h`
Examples: 6115
Market-timeseries rows: 94203240
Split policy: `end_date`

## Files
- `manifest.json`
- `examples.parquet`: leakage-safe observable input rows
- `market_timeseries.parquet`
- `targets.parquet`: labels and auxiliary target fields

## Manifest
```json
{
  "config": {
    "horizons_hours": [
      24,
      72,
      168
    ],
    "max_snapshot_staleness_hours": 12.0,
    "max_snapshot_staleness_hours_by_horizon": {
      "168": 24.0,
      "24": 12.0,
      "336": 48.0
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
    "prefix_rows",
    "cutoff_age_hours",
    "current_yes_probability",
    "current_probability_timestamp_utc",
    "current_probability_staleness_hours",
    "split"
  ],
  "families": 1671,
  "horizons_hours": [
    24,
    72,
    168
  ],
  "label_stats": {
    "by_split": {
      "test": {
        "positive_rate": 0.26715686274509803,
        "rows": 1224
      },
      "train": {
        "positive_rate": 0.21692905336332038,
        "rows": 4891
      }
    },
    "overall": {
      "positive_rate": 0.22698282910874898,
      "rows": 6115
    }
  },
  "market_timeseries_columns": [
    "market_id",
    "timestamp_utc",
    "yes_probability"
  ],
  "market_timeseries_rows": 94203240,
  "markets": 3480,
  "name": "terminal_benchmark",
  "observable_information": "target market metadata and market-level probability history up to cutoff",
  "release_name": "kalshi-terminal-24h-72h-168h",
  "rows": 6115,
  "rows_by_horizon_and_split": {
    "test": {
      "168": 820,
      "24": 200,
      "72": 204
    },
    "train": {
      "168": 2531,
      "24": 1182,
      "72": 1178
    }
  },
  "schema_version": 1,
  "source": "kalshi",
  "split_audit": {
    "family_overlap": {
      "families_by_split": {
        "test": 466,
        "train": 1265
      },
      "families_with_multiple_splits": 60,
      "overlapping_family_ids_sample": [
        "unknown::::2 us netflix movie on feb",
        "unknown::::2 us netflix show on feb",
        "unknown::::ali khamenei leave office before 2026",
        "unknown::::any individual connection with minnesota daycare",
        "unknown::::any member trump s cabinet leave",
        "unknown::::artist with most monthly spotify listeners",
        "unknown::::average gas prices above 2 90",
        "unknown::::average gas prices above 2 95",
        "unknown::::average gas prices above 3 00",
        "unknown::::average gas prices above 3 05",
        "unknown::::average gas prices above 3 10",
        "unknown::::average gas prices above 3 15",
        "unknown::::average gas prices above 3 20",
        "unknown::::average gas prices above 3 30",
        "unknown::::average gas prices above 3 40",
        "unknown::::average gas prices above 3 50",
        "unknown::::average gas prices above 4 30",
        "unknown::::average gas prices above 4 60",
        "unknown::::average gas prices above 4 70",
        "unknown::::average gas prices above 4 90"
      ],
      "pairwise_family_overlap": {
        "test__train": 60
      }
    },
    "pairwise_unit_overlap": {
      "test__train": 0
    },
    "rows_by_split": {
      "test": 1224,
      "train": 4891
    },
    "split_unit": "market_id",
    "units_by_split": {
      "test": 846,
      "train": 2634
    },
    "units_with_multiple_splits": 0
  },
  "split_counts": {
    "test": 1224,
    "train": 4891
  },
  "split_policy": {
    "assignment_rule": "all horizons derived from one market_id inherit the market-level split computed from a single timestamp key",
    "split_on": "end_date",
    "split_unit": "market_id",
    "timestamp_key_definition": "market end_date when split_on=end_date; market cutoff timestamp when split_on=cutoff_timestamp_utc"
  },
  "target_columns": [
    "market_id",
    "horizon_hours",
    "label",
    "split"
  ],
  "target_type": "final resolved outcome",
  "task": "terminal_outcome_prediction"
}
```
