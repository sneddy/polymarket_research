"""Repricing benchmark built around a frozen manifest and normalized market histories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from polymarket_research.benchmarks.common import (
    assign_time_splits,
    evaluate_binary_predictions,
    format_repricing_example_ids,
    normalize_utc_timestamp,
    to_json_ready,
)
from polymarket_research.benchmarks.tabular import TabularBenchmark
from polymarket_research.data.canonical.dataset import CanonicalDataset
from polymarket_research.data.representations.repricing import RepricingPanelBuilder


REPRICING_EXAMPLE_COLUMNS = [
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
    "future_move",
    "label",
    "split",
]

REPRICING_MARKET_TIMESERIES_COLUMNS = [
    "market_id",
    "timestamp_utc",
    "yes_probability",
]


@dataclass(frozen=True)
class RepricingBenchmarkConfig:
    """Configuration for the frozen repricing benchmark."""

    future_horizon_hours: int = 24
    lookback_hours: int = 24
    sample_every_hours: int = 12
    move_threshold: float = 0.15
    attach_external_shocks: bool = True
    target_market_only: bool = True
    split_on: str = "timestamp_utc"
    split_timestamp_utc: pd.Timestamp | None = None
    train_fraction: float = 0.8
    show_progress: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "future_horizon_hours", int(self.future_horizon_hours))
        object.__setattr__(self, "lookback_hours", int(self.lookback_hours))
        object.__setattr__(self, "sample_every_hours", int(self.sample_every_hours))
        object.__setattr__(self, "split_timestamp_utc", normalize_utc_timestamp(self.split_timestamp_utc))

    def as_dict(self) -> dict[str, Any]:
        return to_json_ready(asdict(self))


@dataclass(frozen=True)
class RepricingBenchmark:
    """Frozen repricing manifest plus normalized market-level probability histories."""

    config: RepricingBenchmarkConfig
    examples: pd.DataFrame
    market_timeseries: pd.DataFrame
    canonical: CanonicalDataset | None = None

    @property
    def release_name(self) -> str:
        return f"polymarket-repricing-{int(self.config.future_horizon_hours)}h"

    @classmethod
    def build(
        cls,
        canonical: CanonicalDataset,
        *,
        config: RepricingBenchmarkConfig | None = None,
    ) -> "RepricingBenchmark":
        cfg = config or RepricingBenchmarkConfig()
        panel = RepricingPanelBuilder(
            canonical=canonical,
            future_horizon_hours=cfg.future_horizon_hours,
            lookback_hours=cfg.lookback_hours,
            sample_every_hours=cfg.sample_every_hours,
            move_threshold=cfg.move_threshold,
            attach_external_shocks=cfg.attach_external_shocks,
            show_progress=cfg.show_progress,
        ).build().frame

        if panel.empty:
            examples = pd.DataFrame(columns=REPRICING_EXAMPLE_COLUMNS)
            market_timeseries = pd.DataFrame(columns=REPRICING_MARKET_TIMESERIES_COLUMNS)
            return cls(config=cfg, examples=examples, market_timeseries=market_timeseries, canonical=canonical)

        examples = panel.loc[
            :,
            [
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
                "future_move",
                "target",
            ],
        ].copy()
        examples = examples.rename(columns={"target": "label"})
        examples["label"] = examples["label"].astype(int)
        examples["split"] = assign_time_splits(
            examples,
            split_on=cfg.split_on,
            valid_columns={"timestamp_utc", "end_date"},
            split_timestamp_utc=cfg.split_timestamp_utc,
            train_fraction=cfg.train_fraction,
        )
        examples["_split_order"] = examples["split"].map({"train": 0, "test": 1}).fillna(2)
        examples = examples.sort_values(
            ["_split_order", "timestamp_utc", "market_id"],
            kind="stable",
        ).reset_index(drop=True)
        examples = examples.loc[:, REPRICING_EXAMPLE_COLUMNS]

        market_ids = set(examples["market_id"].astype(str))
        market_timeseries = canonical.probabilities.loc[
            canonical.probabilities["market_id"].astype(str).isin(market_ids),
            REPRICING_MARKET_TIMESERIES_COLUMNS,
        ].copy()
        market_timeseries["market_id"] = market_timeseries["market_id"].astype(str)
        market_timeseries["timestamp_utc"] = pd.to_datetime(market_timeseries["timestamp_utc"], utc=True, errors="coerce")
        market_timeseries = market_timeseries.loc[market_timeseries["timestamp_utc"].notna(), REPRICING_MARKET_TIMESERIES_COLUMNS].reset_index(drop=True)

        return cls(config=cfg, examples=examples, market_timeseries=market_timeseries, canonical=canonical)

    @classmethod
    def from_canonical(
        cls,
        canonical: CanonicalDataset,
        *,
        config: RepricingBenchmarkConfig | None = None,
    ) -> "RepricingBenchmark":
        return cls.build(canonical, config=config)

    def split_examples(self, split: str) -> pd.DataFrame:
        return self.examples.loc[self.examples["split"] == split].reset_index(drop=True).copy()

    def targets(self, split: str | None = None) -> pd.DataFrame:
        frame = self.examples if split is None else self.examples.loc[self.examples["split"] == split]
        return frame.loc[:, ["market_id", "timestamp_utc", "future_horizon_hours", "label", "split"]].reset_index(drop=True).copy()

    def resolve_market_snapshot(self, market_id: str, timestamp_utc: pd.Timestamp | str) -> pd.Series:
        ts = normalize_utc_timestamp(pd.Timestamp(timestamp_utc))
        match = self.examples.loc[
            (self.examples["market_id"] == str(market_id))
            & (pd.to_datetime(self.examples["timestamp_utc"], utc=True) == ts)
        ]
        if match.empty:
            raise KeyError(f"Unknown repricing snapshot for market_id={market_id} at timestamp_utc={ts}")
        return match.iloc[0].copy()

    def market_history(self, market_id: str) -> pd.DataFrame:
        history = self.market_timeseries.loc[self.market_timeseries["market_id"] == str(market_id)].reset_index(drop=True).copy()
        if history.empty:
            raise KeyError(f"Missing market history for market_id={market_id}")
        return history

    def history_until(self, market_id: str, timestamp_utc: pd.Timestamp | str) -> pd.DataFrame:
        ts = normalize_utc_timestamp(pd.Timestamp(timestamp_utc))
        history = self.market_history(market_id)
        return history.loc[history["timestamp_utc"] <= ts].reset_index(drop=True).copy()

    def evaluate(
        self,
        predictions: pd.DataFrame,
        *,
        split: str = "test",
    ) -> dict[str, pd.DataFrame]:
        gold = self.targets(split=split).copy()
        results = evaluate_binary_predictions(
            gold=gold,
            predictions=predictions,
            split=split,
            group_col="future_horizon_hours",
            id_columns=("market_id", "timestamp_utc"),
        )
        if "by_future_horizon" in results:
            results["by_horizon"] = results.pop("by_future_horizon")
        return results

    def save(self, directory: str | Path) -> dict[str, Path]:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "examples": out_dir / "examples.parquet",
            "market_timeseries": out_dir / "market_timeseries.parquet",
            "manifest": out_dir / "manifest.json",
        }
        self.examples.to_parquet(paths["examples"], index=False)
        self.market_timeseries.to_parquet(paths["market_timeseries"], index=False)
        paths["manifest"].write_text(json.dumps(self.manifest(), indent=2, sort_keys=True), encoding="utf-8")
        return paths

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        canonical: CanonicalDataset | None = None,
    ) -> "RepricingBenchmark":
        source_dir = Path(directory)
        manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
        config_dict = dict(manifest.get("config", {}))
        if config_dict.get("split_timestamp_utc") is not None:
            config_dict["split_timestamp_utc"] = pd.Timestamp(config_dict["split_timestamp_utc"])

        config = RepricingBenchmarkConfig(**config_dict)
        examples = pd.read_parquet(source_dir / "examples.parquet")
        market_timeseries = pd.read_parquet(source_dir / "market_timeseries.parquet")

        if not examples.empty:
            examples["timestamp_utc"] = pd.to_datetime(examples["timestamp_utc"], utc=True, errors="coerce")
            examples["created_at"] = pd.to_datetime(examples["created_at"], utc=True, errors="coerce")
            examples["end_date"] = pd.to_datetime(examples["end_date"], utc=True, errors="coerce")
            examples = examples.loc[:, REPRICING_EXAMPLE_COLUMNS]
        else:
            examples = pd.DataFrame(columns=REPRICING_EXAMPLE_COLUMNS)

        if not market_timeseries.empty:
            market_timeseries["timestamp_utc"] = pd.to_datetime(market_timeseries["timestamp_utc"], utc=True, errors="coerce")
            market_timeseries = market_timeseries.loc[:, REPRICING_MARKET_TIMESERIES_COLUMNS]
        else:
            market_timeseries = pd.DataFrame(columns=REPRICING_MARKET_TIMESERIES_COLUMNS)

        return cls(config=config, examples=examples, market_timeseries=market_timeseries, canonical=canonical)

    def manifest(self) -> dict[str, Any]:
        split_counts = self.examples["split"].value_counts(dropna=False).sort_index().to_dict()
        return {
            "name": "repricing_benchmark",
            "release_name": self.release_name,
            "task": "large_future_repricing_prediction",
            "observable_information": "target market metadata and market-level probability history up to the prediction timestamp",
            "target_type": "large future repricing indicator",
            "config": self.config.as_dict(),
            "rows": int(len(self.examples)),
            "market_timeseries_rows": int(len(self.market_timeseries)),
            "split_counts": {str(key): int(value) for key, value in split_counts.items()},
            "future_horizon_hours": int(self.config.future_horizon_hours),
            "example_columns": REPRICING_EXAMPLE_COLUMNS,
            "market_timeseries_columns": REPRICING_MARKET_TIMESERIES_COLUMNS,
        }

    def _examples_for_view(self) -> pd.DataFrame:
        frame = self.examples.loc[:, ["market_id", "timestamp_utc", "future_horizon_hours", "split"]].copy()
        frame["example_id"] = format_repricing_example_ids(frame)
        return frame

    def build_reference_view(self) -> pd.DataFrame:
        return self.view().frame.copy()

    def view(self):
        if self.canonical is None:
            raise ValueError("RepricingBenchmark.view() requires a canonical dataset; build with RepricingBenchmark.build(...).")
        return self.canonical.repricing_benchmark_view(
            future_horizon_hours=self.config.future_horizon_hours,
            lookback_hours=self.config.lookback_hours,
            sample_every_hours=self.config.sample_every_hours,
            move_threshold=self.config.move_threshold,
            attach_external_shocks=self.config.attach_external_shocks,
            show_progress=self.config.show_progress,
            examples=self._examples_for_view(),
        )

    def tabular(self) -> TabularBenchmark:
        return TabularBenchmark.from_view(self.view(), evaluation_group_col="future_horizon_hours")
