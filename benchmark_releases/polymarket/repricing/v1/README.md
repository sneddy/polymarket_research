# Polymarket Repricing Benchmark

Source: `polymarket`
Release: `polymarket-repricing-24h`
Examples: 1316457
Market-timeseries rows: 206101450
Future horizon (hours): 24

## Files
- `manifest.json`
- `examples.parquet`: leakage-safe observable input rows
- `market_timeseries.parquet`
- `targets.parquet`: labels and auxiliary target fields

## Manifest
```json
{
  "config": {
    "attach_external_shocks": true,
    "future_horizon_hours": 24,
    "lookback_hours": 24,
    "move_threshold": 0.15,
    "sample_every_hours": 12,
    "show_progress": true,
    "split_on": "timestamp_utc",
    "split_timestamp_utc": null,
    "target_market_only": true,
    "train_fraction": 0.8
  },
  "example_columns": [
    "market_id",
    "timestamp_utc",
    "created_at",
    "end_date",
    "question",
    "platform_category",
    "research_category",
    "family_id",
    "future_horizon_hours",
    "current_yes_probability",
    "split"
  ],
  "families": 11374,
  "future_horizon_hours": 24,
  "label_stats": {
    "by_split": {
      "test": {
        "positive_rate": 0.05174746812694251,
        "rows": 94594
      },
      "train": {
        "positive_rate": 0.016825945298286304,
        "rows": 1221863
      }
    },
    "overall": {
      "positive_rate": 0.01933523085068483,
      "rows": 1316457
    }
  },
  "market_timeseries_columns": [
    "market_id",
    "timestamp_utc",
    "yes_probability"
  ],
  "market_timeseries_rows": 206101450,
  "markets": 13995,
  "name": "repricing_benchmark",
  "observable_information": "target market metadata and market-level probability history up to the prediction timestamp",
  "release_name": "polymarket-repricing-24h",
  "rows": 1316457,
  "schema_version": 1,
  "source": "polymarket",
  "split_audit": {
    "family_overlap": {
      "families_by_split": {
        "test": 2104,
        "train": 9479
      },
      "families_with_multiple_splits": 209,
      "overlapping_family_ids_sample": [
        "unknown::::advanced micro devices amd beat quarterly",
        "unknown::::al gore named newly released epstein",
        "unknown::::alex honnold free solo taipei 101",
        "unknown::::alexandria ocasio cortez named newly released",
        "unknown::::alibaba have best ai model at",
        "unknown::::alibaba have second best ai model",
        "unknown::::alphabet googl beat quarterly earnings",
        "unknown::::american express axp beat quarterly earnings",
        "unknown::::andr ventura win 1st round 2026",
        "unknown::::another 7 0 or above earthquake",
        "unknown::::ant nio jos seguro win 1st",
        "unknown::::anthropic have 2 ai model at",
        "unknown::::anthropic have 3 ai model at",
        "unknown::::anthropic have best ai model at",
        "unknown::::anthropic have second best ai model",
        "unknown::::anthropic have third best ai model",
        "unknown::::anthropic run ad during super bowl",
        "unknown::::any presidential candidate win outright first",
        "unknown::::apple aapl beat quarterly earnings",
        "unknown::::apple run ad during super bowl"
      ],
      "pairwise_family_overlap": {
        "test__train": 209
      }
    },
    "pairwise_unit_overlap": {
      "test__train": 0
    },
    "rows_by_split": {
      "test": 94594,
      "train": 1221863
    },
    "split_unit": "market_id",
    "units_by_split": {
      "test": 2801,
      "train": 11194
    },
    "units_with_multiple_splits": 0
  },
  "split_counts": {
    "test": 94594,
    "train": 1221863
  },
  "split_policy": {
    "assignment_rule": "all rolling repricing windows derived from one market_id inherit a single market-level split",
    "split_on": "timestamp_utc",
    "split_unit": "market_id",
    "timestamp_key_definition": "first admissible repricing timestamp when split_on=timestamp_utc; market end_date when split_on=end_date"
  },
  "target_columns": [
    "market_id",
    "timestamp_utc",
    "future_horizon_hours",
    "label",
    "future_move",
    "split"
  ],
  "target_type": "large future repricing indicator",
  "task": "large_future_repricing_prediction"
}
```
