"""Reference baselines for the frozen decisiveness benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from polymarket_research.benchmarks.baselines.common import (
    benchmark_targets_frame,
    dataclass_manifest,
    numeric_mode,
)
from polymarket_research.benchmarks.schemas.decisiveness import DecisivenessBenchmark


@dataclass(frozen=True)
class DecisivenessMajorityBaseline:
    """Majority-label plus median-hours baseline fit from the train split."""

    pred_label: int
    pred_label_name: str | None
    pred_hours_to_decisive: float
    train_rows: int
    train_split: str = "train"

    def predict(
        self,
        benchmark: DecisivenessBenchmark,
        *,
        split: str = "test",
    ) -> pd.DataFrame:
        targets = benchmark_targets_frame(
            benchmark,
            split=split,
            baseline_name="decisiveness_majority_baseline",
        )
        predictions = targets.loc[:, ["market_id", "cutoff_timestamp_utc"]].copy()
        predictions["pred_label"] = int(self.pred_label)
        predictions["pred_hours_to_decisive"] = float(self.pred_hours_to_decisive)
        return predictions

    def evaluate(
        self,
        benchmark: DecisivenessBenchmark,
        *,
        split: str = "test",
    ) -> dict[str, pd.DataFrame]:
        return benchmark.evaluate(self.predict(benchmark, split=split), split=split)

    def manifest(self) -> dict[str, Any]:
        return dataclass_manifest(self, name="decisiveness_majority_baseline", train_split=self.train_split)


def fit_decisiveness_majority_baseline(
    benchmark: DecisivenessBenchmark,
    *,
    split: str = "train",
) -> DecisivenessMajorityBaseline:
    """Fit the majority decisiveness baseline from one split of frozen targets."""
    targets = benchmark_targets_frame(
        benchmark,
        split=split,
        baseline_name="decisiveness_majority_baseline",
    )
    pred_label = numeric_mode(targets["label"])

    label_name_series = targets.loc[pd.to_numeric(targets["label"], errors="coerce") == pred_label, "label_name"]
    pred_label_name = None if label_name_series.empty else str(label_name_series.mode().iloc[0])
    pred_hours_to_decisive = float(pd.to_numeric(targets["hours_to_decisive"], errors="coerce").median())

    return DecisivenessMajorityBaseline(
        pred_label=pred_label,
        pred_label_name=pred_label_name,
        pred_hours_to_decisive=pred_hours_to_decisive,
        train_rows=int(len(targets)),
        train_split=split,
    )
