"""Reference baselines for the frozen repricing benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from polymarket_research.benchmarks.baselines.common import (
    benchmark_targets_frame,
    dataclass_manifest,
)
from polymarket_research.benchmarks.schemas.repricing import RepricingBenchmark


@dataclass(frozen=True)
class RepricingTrainRateBaseline:
    """Constant-probability baseline fit from the train split repricing rate."""

    pred_prob: float
    train_rows: int
    positive_rate: float
    train_split: str = "train"

    def predict(
        self,
        benchmark: RepricingBenchmark,
        *,
        split: str = "test",
    ) -> pd.DataFrame:
        targets = benchmark_targets_frame(
            benchmark,
            split=split,
            baseline_name="repricing_train_rate_baseline",
        )
        predictions = targets.loc[:, ["market_id", "timestamp_utc"]].copy()
        predictions["pred_prob"] = float(self.pred_prob)
        return predictions

    def evaluate(
        self,
        benchmark: RepricingBenchmark,
        *,
        split: str = "test",
    ) -> dict[str, pd.DataFrame]:
        return benchmark.evaluate(self.predict(benchmark, split=split), split=split)

    def manifest(self) -> dict[str, Any]:
        return dataclass_manifest(self, name="repricing_train_rate_baseline", train_split=self.train_split)


def fit_repricing_train_rate_baseline(
    benchmark: RepricingBenchmark,
    *,
    split: str = "train",
) -> RepricingTrainRateBaseline:
    """Fit a constant repricing baseline from the empirical train split label rate."""
    targets = benchmark_targets_frame(
        benchmark,
        split=split,
        baseline_name="repricing_train_rate_baseline",
    )
    positive_rate = float(pd.to_numeric(targets["label"], errors="coerce").mean())
    return RepricingTrainRateBaseline(
        pred_prob=positive_rate,
        train_rows=int(len(targets)),
        positive_rate=positive_rate,
        train_split=split,
    )
