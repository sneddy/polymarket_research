"""ML-facing tabular facade for frozen benchmark views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from polymarket_research.benchmarks.common import (
    SplitFrame,
    evaluate_binary_predictions,
    evaluate_multiclass_predictions,
    evaluate_regression_predictions,
)
from polymarket_research.views import BenchmarkView


@dataclass(frozen=True)
class TabularBenchmark:
    """Single-frame benchmark facade for Kaggle-style ML workflows."""

    name: str
    frame: pd.DataFrame
    target_col: str
    feature_columns: list[str]
    time_col: str
    entity_id_col: str = "market_id"
    split_col: str = "split"
    metadata: dict[str, Any] | None = None
    evaluation_group_col: str | None = None

    @classmethod
    def from_view(
        cls,
        view: BenchmarkView,
        *,
        evaluation_group_col: str | None = None,
    ) -> "TabularBenchmark":
        if "split" not in view.frame.columns:
            raise ValueError("benchmark view must include a 'split' column for tabular access.")
        return cls(
            name=view.name,
            frame=view.frame.copy(),
            target_col=view.target_col,
            feature_columns=view.task.feature_columns,
            time_col=view.time_col,
            entity_id_col=view.entity_id_col,
            metadata=dict(view.metadata),
            evaluation_group_col=evaluation_group_col,
        )

    @property
    def train_df(self) -> pd.DataFrame:
        return self.frame.loc[self.frame[self.split_col] == "train"].reset_index(drop=True).copy()

    @property
    def test_df(self) -> pd.DataFrame:
        return self.frame.loc[self.frame[self.split_col] == "test"].reset_index(drop=True).copy()

    @property
    def train(self) -> SplitFrame:
        return SplitFrame(frame=self.train_df, target_col=self.target_col, feature_columns=self.feature_columns)

    @property
    def test(self) -> SplitFrame:
        return SplitFrame(frame=self.test_df, target_col=self.target_col, feature_columns=self.feature_columns)

    @property
    def X_train(self) -> pd.DataFrame:
        return self.train.X

    @property
    def y_train(self) -> pd.Series:
        return self.train.y

    @property
    def X_test(self) -> pd.DataFrame:
        return self.test.X

    @property
    def y_test(self) -> pd.Series:
        return self.test.y

    def evaluate_predictions(
        self,
        predictions: pd.DataFrame | pd.Series,
        *,
        split: str = "test",
    ) -> dict[str, pd.DataFrame]:
        gold_columns = ["example_id", self.target_col]
        if self.evaluation_group_col is not None and self.evaluation_group_col in self.frame.columns:
            gold_columns.append(self.evaluation_group_col)
        gold = self.frame.loc[self.frame[self.split_col] == split, gold_columns].copy()
        group_col = self.evaluation_group_col if self.evaluation_group_col in gold.columns else None

        if isinstance(predictions, pd.DataFrame):
            prediction_columns = set(predictions.columns)
            if "pred_target" in prediction_columns:
                gold = gold.rename(columns={self.target_col: "target"})
                return evaluate_regression_predictions(
                    gold,
                    predictions,
                    split=split,
                    value_col="target",
                    pred_col="pred_target",
                    group_col=group_col,
                )
            if "pred_label" in prediction_columns:
                gold = gold.rename(columns={self.target_col: "label"})
                return evaluate_multiclass_predictions(
                    gold,
                    predictions,
                    split=split,
                    group_col=group_col,
                )

        gold = gold.rename(columns={self.target_col: "label"})
        target_values = gold["label"].dropna().unique().tolist()
        is_binary_target = set(target_values).issubset({0, 1})
        if not is_binary_target:
            return evaluate_multiclass_predictions(
                gold,
                predictions,
                split=split,
                group_col=group_col,
            )
        return evaluate_binary_predictions(
            gold,
            predictions,
            split=split,
            group_col=group_col,
        )
