"""Stable benchmark evaluators for frozen benchmark objects."""

from __future__ import annotations

import pandas as pd

from polymarket_research.benchmarks.schemas.decisiveness import DecisivenessBenchmark
from polymarket_research.benchmarks.schemas.repricing import RepricingBenchmark
from polymarket_research.benchmarks.schemas.terminal import TerminalBenchmark


def evaluate_terminal(
    benchmark: TerminalBenchmark,
    predictions: pd.DataFrame | pd.Series,
    *,
    split: str = "test",
) -> dict[str, pd.DataFrame]:
    """Evaluate predictions against a frozen terminal benchmark."""
    return benchmark.evaluate(predictions, split=split)


def evaluate_decisiveness(
    benchmark: DecisivenessBenchmark,
    predictions: pd.DataFrame | pd.Series,
    *,
    split: str = "test",
) -> dict[str, pd.DataFrame]:
    """Evaluate predictions against a frozen decisiveness benchmark."""
    return benchmark.evaluate(predictions, split=split)


def evaluate_repricing(
    benchmark: RepricingBenchmark,
    predictions: pd.DataFrame,
    *,
    split: str = "test",
) -> dict[str, pd.DataFrame]:
    """Evaluate predictions against a frozen repricing benchmark."""
    return benchmark.evaluate(predictions, split=split)
