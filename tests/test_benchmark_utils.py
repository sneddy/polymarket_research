from __future__ import annotations

import unittest

import pandas as pd

from benchmarks.benchmark_utils import extract_snapshot_features
from benchmarks.benchmark_utils import rolling_time_splits


class BenchmarkUtilsTests(unittest.TestCase):
    def test_extract_snapshot_features_rejects_stale_snapshot(self) -> None:
        panel = pd.DataFrame(
            {
                "market_id": ["m1", "m1"],
                "timestamp_utc": pd.to_datetime(
                    ["2025-01-01T00:00:00Z", "2025-01-01T06:00:00Z"],
                    utc=True,
                ),
                "yes_probability": [0.4, 0.7],
                "observed_trade": [1, 1],
                "trade_count": [1, 2],
                "total_size": [10.0, 20.0],
                "last_trade_price": [0.4, 0.7],
            }
        )

        features = extract_snapshot_features(
            panel,
            cutoff=pd.Timestamp("2025-01-01T18:00:00Z"),
            max_snapshot_staleness_hours=6.0,
        )
        self.assertIsNone(features)

        fresh = extract_snapshot_features(
            panel,
            cutoff=pd.Timestamp("2025-01-01T10:00:00Z"),
            max_snapshot_staleness_hours=6.0,
        )
        self.assertIsNotNone(fresh)
        assert fresh is not None
        self.assertAlmostEqual(fresh["snapshot_staleness_hours"], 4.0)

    def test_rolling_time_splits_are_monotone_and_non_overlapping(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2025-01-01", periods=12, freq="D", tz="UTC"),
                "value": list(range(12)),
            }
        )

        splits = rolling_time_splits(df, time_col="timestamp_utc", n_splits=3, min_train_fraction=0.5)

        self.assertEqual(len(splits), 3)
        previous_test_end = None
        for fold, (train_df, test_df, meta) in enumerate(splits):
            self.assertGreater(len(train_df), 0)
            self.assertGreater(len(test_df), 0)
            self.assertLess(train_df["timestamp_utc"].max(), test_df["timestamp_utc"].min())
            self.assertEqual(meta["fold"], fold)
            if previous_test_end is not None:
                self.assertGreater(test_df["timestamp_utc"].min(), previous_test_end)
            previous_test_end = test_df["timestamp_utc"].max()


if __name__ == "__main__":
    unittest.main()
