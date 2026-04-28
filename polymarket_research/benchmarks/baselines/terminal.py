"""Reference baselines for the frozen terminal benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from polymarket_research.benchmarks.baselines.common import (
    benchmark_targets_frame,
    dataclass_manifest,
)
from polymarket_research.benchmarks.schemas.terminal import TerminalBenchmark


@dataclass(frozen=True)
class TerminalTrainRateBaseline:
    """Constant-probability baseline fit from the train split label rate."""

    pred_prob: float
    train_rows: int
    positive_rate: float
    train_split: str = "train"

    def predict(
        self,
        benchmark: TerminalBenchmark,
        *,
        split: str = "test",
    ) -> pd.DataFrame:
        targets = benchmark_targets_frame(
            benchmark,
            split=split,
            baseline_name="terminal_train_rate_baseline",
        )
        predictions = targets.loc[:, ["market_id", "horizon_hours"]].copy()
        predictions["pred_prob"] = float(self.pred_prob)
        return predictions

    def evaluate(
        self,
        benchmark: TerminalBenchmark,
        *,
        split: str = "test",
    ) -> dict[str, pd.DataFrame]:
        return benchmark.evaluate(self.predict(benchmark, split=split), split=split)

    def manifest(self) -> dict[str, Any]:
        return dataclass_manifest(self, name="terminal_train_rate_baseline", train_split=self.train_split)


@dataclass(frozen=True)
class TerminalLastProbabilityBaseline:
    """Market-implied terminal baseline using the latest probability at each cutoff."""

    train_split: str = "train"

    def predict(
        self,
        benchmark: TerminalBenchmark,
        *,
        split: str = "test",
    ) -> pd.DataFrame:
        benchmark_targets_frame(
            benchmark,
            split=split,
            baseline_name="terminal_last_probability_baseline",
        )
        predictions = benchmark.market_cutoff_probabilities(split=split)
        return predictions.rename(columns={"market_pred_prob": "pred_prob"})

    def evaluate(
        self,
        benchmark: TerminalBenchmark,
        *,
        split: str = "test",
    ) -> dict[str, pd.DataFrame]:
        return benchmark.evaluate(self.predict(benchmark, split=split), split=split)

    def manifest(self) -> dict[str, Any]:
        return dataclass_manifest(self, name="terminal_last_probability_baseline", train_split=self.train_split)


def fit_terminal_train_rate_baseline(
    benchmark: TerminalBenchmark,
    *,
    split: str = "train",
) -> TerminalTrainRateBaseline:
    """Fit a constant terminal baseline from the empirical label rate on one split."""
    targets = benchmark_targets_frame(
        benchmark,
        split=split,
        baseline_name="terminal_train_rate_baseline",
    )
    positive_rate = float(pd.to_numeric(targets["label"], errors="coerce").mean())
    return TerminalTrainRateBaseline(
        pred_prob=positive_rate,
        train_rows=int(len(targets)),
        positive_rate=positive_rate,
        train_split=split,
    )


def fit_terminal_last_probability_baseline(
    benchmark: TerminalBenchmark,
    *,
    split: str = "train",
) -> TerminalLastProbabilityBaseline:
    """Return the market-implied terminal baseline; split is recorded for manifest symmetry."""
    benchmark_targets_frame(
        benchmark,
        split=split,
        baseline_name="terminal_last_probability_baseline",
    )
    return TerminalLastProbabilityBaseline(train_split=split)
