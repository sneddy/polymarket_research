# Polymarket Decisiveness Benchmark

Source: `polymarket`
Release: `polymarket-decisive-belief-tau95`
Examples: 924923
Markets: 13460
Market-timeseries rows: 176867529

## Files
- `manifest.json`
- `examples.parquet`: leakage-safe observable input rows
- `market_timeseries.parquet`
- `targets.parquet`: labels and auxiliary target fields

## Manifest
```json
{
  "config": {
    "decisive_threshold": 0.95,
    "min_history_points": 24,
    "min_prefix_age_hours": 6.0,
    "min_time_to_decisive_hours": 1.0,
    "ordinal_bin_edges_hours": [
      24.0,
      72.0
    ],
    "ordinal_bin_labels": [
      "short",
      "medium",
      "long"
    ],
    "sample_every_hours": 12,
    "show_progress": true,
    "split_on": "decisive_timestamp_utc",
    "split_timestamp_utc": null,
    "target_market_only": true,
    "train_fraction": 0.8
  },
  "decisive_side_distribution": {
    "by_split": {
      "test": {
        "no": 130548,
        "yes": 46600
      },
      "train": {
        "no": 579436,
        "yes": 168339
      }
    },
    "overall": {
      "no": 709984,
      "yes": 214939
    }
  },
  "example_columns": [
    "market_id",
    "cutoff_timestamp_utc",
    "market_slug",
    "question",
    "created_at",
    "end_date",
    "platform_category",
    "research_category",
    "family_id",
    "prefix_rows",
    "cutoff_age_hours",
    "current_yes_probability",
    "confidence_margin",
    "distance_to_yes_decisive",
    "distance_to_no_decisive",
    "decisive_threshold",
    "split"
  ],
  "families": 10836,
  "market_timeseries_columns": [
    "market_id",
    "timestamp_utc",
    "yes_probability"
  ],
  "market_timeseries_rows": 176867529,
  "markets": 13460,
  "name": "decisiveness_benchmark",
  "observable_information": "target market metadata and market-level probability history up to the cutoff timestamp",
  "ordinal_bin_edges_hours": [
    24.0,
    72.0
  ],
  "ordinal_bin_labels": [
    "short",
    "medium",
    "long"
  ],
  "ordinal_label_distribution": {
    "by_split": {
      "test": {
        "long": 159620,
        "medium": 9776,
        "short": 7752
      },
      "train": {
        "long": 680085,
        "medium": 36769,
        "short": 30921
      }
    },
    "overall": {
      "long": 839705,
      "medium": 46545,
      "short": 38673
    }
  },
  "release_name": "polymarket-decisive-belief-tau95",
  "rows": 924923,
  "schema_version": 1,
  "source": "polymarket",
  "split_audit": {
    "family_overlap": {
      "families_by_split": {
        "test": 2149,
        "train": 8914
      },
      "families_with_multiple_splits": 227,
      "overlapping_family_ids_sample": [
        "unknown::::10 year treasury yield hit 4",
        "unknown::::2026 u s house election republican",
        "unknown::::advanced micro devices amd beat quarterly",
        "unknown::::agatha christie s seven dials 2",
        "unknown::::al gore named newly released epstein",
        "unknown::::alexandria ocasio cortez named newly released",
        "unknown::::alphabet googl beat quarterly earnings",
        "unknown::::american express axp beat quarterly earnings",
        "unknown::::another 7 0 or above earthquake",
        "unknown::::another mrbeast video get 100m week",
        "unknown::::ant nio jos seguro win second",
        "unknown::::anthropic have 3 ai model at",
        "unknown::::anthropic have best ai model at",
        "unknown::::anthropic have best ai model for",
        "unknown::::anthropic have second best ai model",
        "unknown::::anthropic have third best ai model",
        "unknown::::anthropic have top ai model at",
        "unknown::::anthropic run ad during super bowl",
        "unknown::::any presidential candidate win outright first",
        "unknown::::anyone charged over daycare fraud minnesota"
      ],
      "pairwise_family_overlap": {
        "test__train": 227
      }
    },
    "pairwise_unit_overlap": {
      "test__train": 0
    },
    "rows_by_split": {
      "test": 177148,
      "train": 747775
    },
    "split_unit": "market_id",
    "units_by_split": {
      "test": 2692,
      "train": 10768
    },
    "units_with_multiple_splits": 0
  },
  "split_counts": {
    "test": 177148,
    "train": 747775
  },
  "split_policy": {
    "assignment_rule": "all cutoff examples derived from one market_id inherit a single market-level split",
    "split_on": "decisive_timestamp_utc",
    "split_unit": "market_id",
    "timestamp_key_definition": "durable decisive timestamp when split_on=decisive_timestamp_utc; market end_date when split_on=end_date"
  },
  "target_columns": [
    "market_id",
    "cutoff_timestamp_utc",
    "decisive_timestamp_utc",
    "label",
    "label_name",
    "hours_to_decisive",
    "decisive_side",
    "split"
  ],
  "target_type": "ordinal decisive-horizon label plus continuous hours-to-decisive auxiliary target",
  "task": "durable_decisive_belief_formation_prediction"
}
```
