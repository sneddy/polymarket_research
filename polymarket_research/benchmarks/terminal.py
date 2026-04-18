"""Terminal benchmark built around a frozen manifest and normalized market histories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from polymarket_research.benchmarks.common import (
    assign_time_splits,
    evaluate_binary_predictions,
    normalize_utc_timestamp,
    to_json_ready,
)
from polymarket_research.data.canonical.dataset import CanonicalDataset


TERMINAL_EXAMPLE_COLUMNS = [
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
    "label",
    "split",
]

TERMINAL_MARKET_TIMESERIES_COLUMNS = [
    "market_id",
    "timestamp_utc",
    "yes_probability",
]

TERMINAL_REQUIRED_PROBABILITY_COLUMNS = {"market_id", "timestamp_utc", "yes_probability"}
TERMINAL_REQUIRED_MARKET_COLUMNS = {"market_id", "created_at", "end_date", "final_yes_probability", "question"}


@dataclass(frozen=True)
class TerminalBenchmarkConfig:
    """Configuration for the frozen terminal benchmark."""

    horizons_hours: tuple[int, ...] = (24, 72, 168)
    max_snapshot_staleness_hours: float = 12.0
    max_snapshot_staleness_hours_by_horizon: dict[int, float] | None = None
    split_on: str = "end_date"
    split_timestamp_utc: pd.Timestamp | None = None
    train_fraction: float = 0.8
    show_progress: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizons_hours", tuple(int(value) for value in self.horizons_hours))
        horizon_staleness = self.max_snapshot_staleness_hours_by_horizon
        if horizon_staleness is not None:
            object.__setattr__(
                self,
                "max_snapshot_staleness_hours_by_horizon",
                {int(key): float(value) for key, value in horizon_staleness.items()},
            )
        object.__setattr__(self, "split_timestamp_utc", normalize_utc_timestamp(self.split_timestamp_utc))

    def as_dict(self) -> dict[str, Any]:
        return to_json_ready(asdict(self))

    def max_staleness_for_horizon(self, horizon_hours: int) -> float:
        if self.max_snapshot_staleness_hours_by_horizon is None:
            return float(self.max_snapshot_staleness_hours)
        return float(self.max_snapshot_staleness_hours_by_horizon.get(int(horizon_hours), self.max_snapshot_staleness_hours))


@dataclass(frozen=True)
class TerminalBenchmark:
    """Frozen manifest plus normalized market-level probability histories for terminal prediction."""

    config: TerminalBenchmarkConfig
    examples: pd.DataFrame
    market_timeseries: pd.DataFrame

    @property
    def horizon_hours(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.config.horizons_hours)

    @property
    def release_name(self) -> str:
        if len(self.horizon_hours) == 1:
            return f"polymarket-terminal-{self.horizon_hours[0]}h"
        horizon_tag = "-".join(f"{value}h" for value in self.horizon_hours)
        return f"polymarket-terminal-{horizon_tag}"

    @classmethod
    def build(
        cls,
        canonical: CanonicalDataset,
        config: TerminalBenchmarkConfig | None = None,
    ) -> "TerminalBenchmark":
        cfg = config or TerminalBenchmarkConfig()
        cls._log(
            cfg,
            "starting "
            f"(markets={len(canonical.markets)}, probability_rows={len(canonical.probabilities)}, "
            f"horizons={list(cfg.horizons_hours)}, split_on={cfg.split_on})",
        )
        cls._validate_canonical(canonical)
        cls._log(cfg, "validated canonical tables")
        markets = cls._prepare_markets(canonical.markets)
        probabilities = cls._prepare_probabilities(canonical.probabilities)
        cls._log(cfg, f"prepared inputs (markets={len(markets)}, probability_rows={len(probabilities)})")
        market_slices = cls._build_market_slices(probabilities["market_id"])
        cls._log(cfg, f"indexed contiguous market blocks (markets_with_history={len(market_slices)})")
        examples = cls._build_examples(markets, probabilities, market_slices, cfg)
        cls._log(
            cfg,
            "built example manifest "
            f"(examples={len(examples)}, train={int((examples['split'] == 'train').sum()) if not examples.empty else 0}, "
            f"test={int((examples['split'] == 'test').sum()) if not examples.empty else 0})",
        )
        market_timeseries = cls._build_market_timeseries(probabilities, examples, cfg)
        cls._log(cfg, f"built market histories (market_timeseries_rows={len(market_timeseries)})")
        cls._log(cfg, "done")
        return cls(config=cfg, examples=examples, market_timeseries=market_timeseries)

    @staticmethod
    def _log(config: TerminalBenchmarkConfig, message: str) -> None:
        if config.show_progress:
            print(f"[terminal benchmark] {message}")

    @staticmethod
    def _validate_canonical(canonical: CanonicalDataset) -> None:
        market_cols = set(canonical.markets.columns)
        probability_cols = set(canonical.probabilities.columns)
        missing_market = sorted(TERMINAL_REQUIRED_MARKET_COLUMNS - market_cols)
        missing_probability = sorted(TERMINAL_REQUIRED_PROBABILITY_COLUMNS - probability_cols)

        if missing_market:
            raise ValueError(f"Canonical markets table is missing required columns: {missing_market}")
        if missing_probability:
            raise ValueError(f"Canonical probabilities table is missing required columns: {missing_probability}")

    @staticmethod
    def _prepare_markets(markets: pd.DataFrame) -> pd.DataFrame:
        prepared = markets.copy()
        prepared["market_id"] = prepared["market_id"].astype(str)
        prepared["created_at"] = pd.to_datetime(prepared["created_at"], utc=True, errors="coerce")
        prepared["end_date"] = pd.to_datetime(prepared["end_date"], utc=True, errors="coerce")
        return prepared

    @staticmethod
    def _prepare_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
        prepared = probabilities.loc[:, TERMINAL_MARKET_TIMESERIES_COLUMNS].copy()
        prepared["market_id"] = prepared["market_id"].astype(str)
        prepared["timestamp_utc"] = pd.to_datetime(prepared["timestamp_utc"], utc=True, errors="coerce")
        prepared = prepared.loc[prepared["timestamp_utc"].notna()].reset_index(drop=True)
        return prepared

    @staticmethod
    def _build_market_slices(market_ids: pd.Series) -> dict[str, slice]:
        values = market_ids.astype(str).tolist()
        if not values:
            return {}

        slices: dict[str, slice] = {}
        seen_markets = {values[0]}
        start = 0
        current_market = values[0]

        for index in range(1, len(values) + 1):
            if index < len(values) and values[index] == current_market:
                continue

            slices[current_market] = slice(start, index)
            if index == len(values):
                break

            next_market = values[index]
            if next_market in seen_markets:
                raise ValueError(
                    "canonical.probabilities must be grouped by market_id in contiguous blocks "
                    "and sorted by timestamp_utc within each market."
                )
            seen_markets.add(next_market)
            current_market = next_market
            start = index

        return slices

    @staticmethod
    def _build_examples(
        markets: pd.DataFrame,
        probabilities: pd.DataFrame,
        market_slices: dict[str, slice],
        config: TerminalBenchmarkConfig,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []

        market_iter = markets.itertuples(index=False)
        if config.show_progress:
            market_iter = tqdm(market_iter, total=len(markets), desc="terminal examples", unit="market")
        for market in market_iter:
            if pd.isna(market.created_at) or pd.isna(market.end_date):
                continue

            market_slice = market_slices.get(str(market.market_id))
            if market_slice is None:
                continue

            market_history = probabilities.iloc[market_slice]
            timestamps = pd.DatetimeIndex(market_history["timestamp_utc"])
            if len(timestamps) == 0:
                continue

            created_at = pd.Timestamp(market.created_at)
            end_date = pd.Timestamp(market.end_date)
            for horizon_hours in config.horizons_hours:
                cutoff = end_date - pd.Timedelta(hours=int(horizon_hours))
                if cutoff <= created_at:
                    continue

                prefix_length = int(timestamps.searchsorted(cutoff, side="right"))
                if prefix_length == 0:
                    continue

                last_timestamp = timestamps[prefix_length - 1]
                staleness_hours = float((cutoff - last_timestamp).total_seconds() / 3600.0)
                if staleness_hours > config.max_staleness_for_horizon(int(horizon_hours)):
                    continue

                rows.append(
                    {
                        "market_id": str(market.market_id),
                        "market_slug": getattr(market, "market_slug", None),
                        "question": getattr(market, "question", None),
                        "created_at": created_at,
                        "end_date": end_date,
                        "platform_category": getattr(market, "platform_category", None),
                        "research_category": getattr(market, "research_category", None),
                        "family_id": getattr(market, "family_id", None),
                        "horizon_hours": int(horizon_hours),
                        "cutoff_timestamp_utc": cutoff,
                        "label": int(float(market.final_yes_probability) >= 0.5),
                    }
                )

        if not rows:
            return pd.DataFrame(columns=TERMINAL_EXAMPLE_COLUMNS)

        examples = pd.DataFrame(rows)
        examples["split"] = assign_time_splits(
            examples,
            split_on=config.split_on,
            valid_columns={"end_date", "cutoff_timestamp_utc"},
            split_timestamp_utc=config.split_timestamp_utc,
            train_fraction=config.train_fraction,
        )
        examples["_split_order"] = examples["split"].map({"train": 0, "test": 1}).fillna(2)
        examples = examples.sort_values(
            ["_split_order", "cutoff_timestamp_utc", "market_id", "horizon_hours"],
            kind="stable",
        ).reset_index(drop=True)
        return examples.loc[:, TERMINAL_EXAMPLE_COLUMNS]

    @staticmethod
    def _build_market_timeseries(
        probabilities: pd.DataFrame,
        examples: pd.DataFrame,
        config: TerminalBenchmarkConfig,
    ) -> pd.DataFrame:
        if examples.empty:
            return pd.DataFrame(columns=TERMINAL_MARKET_TIMESERIES_COLUMNS)

        admissible_markets = set(examples["market_id"].astype(str))
        grouped_probabilities = list(probabilities.groupby("market_id", sort=False))
        market_iter = grouped_probabilities
        if config.show_progress:
            market_iter = tqdm(grouped_probabilities, total=len(grouped_probabilities), desc="terminal market histories", unit="market")

        rows: list[pd.DataFrame] = []
        for market_id, market_history in market_iter:
            if str(market_id) not in admissible_markets:
                continue
            rows.append(market_history.loc[:, TERMINAL_MARKET_TIMESERIES_COLUMNS].reset_index(drop=True))

        if not rows:
            return pd.DataFrame(columns=TERMINAL_MARKET_TIMESERIES_COLUMNS)
        return pd.concat(rows, ignore_index=True).loc[:, TERMINAL_MARKET_TIMESERIES_COLUMNS]

    def split_examples(self, split: str) -> pd.DataFrame:
        return self.examples.loc[self.examples["split"] == split].reset_index(drop=True).copy()

    def targets(self, split: str | None = None) -> pd.DataFrame:
        frame = self.examples if split is None else self.examples.loc[self.examples["split"] == split]
        return frame.loc[:, ["market_id", "horizon_hours", "label", "split"]].reset_index(drop=True).copy()

    def resolve_market_id(self, market_id: str) -> pd.Series:
        match = self.examples.loc[self.examples["market_id"] == str(market_id)]
        if match.empty:
            raise KeyError(f"Unknown market_id: {market_id}")
        return match.iloc[0].copy()

    def market_history(self, market_id: str) -> pd.DataFrame:
        history = self.market_timeseries.loc[self.market_timeseries["market_id"] == str(market_id)].reset_index(drop=True).copy()
        if history.empty:
            raise KeyError(f"Missing market history for market_id={market_id}")
        return history

    def evaluate(
        self,
        predictions: pd.DataFrame | pd.Series,
        split: str = "test",
    ) -> dict[str, pd.DataFrame]:
        gold = self.targets(split=split).copy().rename(columns={"market_id": "example_id"})
        if isinstance(predictions, pd.Series):
            pred_frame = predictions.rename("pred_prob").reset_index().rename(columns={"index": "example_id"})
        else:
            pred_frame = predictions.copy().rename(columns={"market_id": "example_id"})
        return evaluate_binary_predictions(predictions=pred_frame, gold=gold, split=split, group_col="horizon_hours")

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
    def load(cls, directory: str | Path) -> "TerminalBenchmark":
        source_dir = Path(directory)
        manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
        config_dict = dict(manifest.get("config", {}))
        if config_dict.get("split_timestamp_utc") is not None:
            config_dict["split_timestamp_utc"] = pd.Timestamp(config_dict["split_timestamp_utc"])

        config = TerminalBenchmarkConfig(**config_dict)
        examples = pd.read_parquet(source_dir / "examples.parquet")
        market_timeseries = pd.read_parquet(source_dir / "market_timeseries.parquet")

        if not examples.empty:
            examples["created_at"] = pd.to_datetime(examples["created_at"], utc=True, errors="coerce")
            examples["end_date"] = pd.to_datetime(examples["end_date"], utc=True, errors="coerce")
            examples["cutoff_timestamp_utc"] = pd.to_datetime(examples["cutoff_timestamp_utc"], utc=True, errors="coerce")
            examples = examples.loc[:, TERMINAL_EXAMPLE_COLUMNS]
        else:
            examples = pd.DataFrame(columns=TERMINAL_EXAMPLE_COLUMNS)

        if not market_timeseries.empty:
            market_timeseries["timestamp_utc"] = pd.to_datetime(market_timeseries["timestamp_utc"], utc=True, errors="coerce")
            market_timeseries = market_timeseries.loc[:, TERMINAL_MARKET_TIMESERIES_COLUMNS]
        else:
            market_timeseries = pd.DataFrame(columns=TERMINAL_MARKET_TIMESERIES_COLUMNS)

        return cls(config=config, examples=examples, market_timeseries=market_timeseries)

    def manifest(self) -> dict[str, Any]:
        split_counts = self.examples["split"].value_counts(dropna=False).sort_index().to_dict()
        return {
            "name": "terminal_benchmark",
            "release_name": self.release_name,
            "task": "terminal_outcome_prediction",
            "observable_information": "target market metadata and market-level probability history up to cutoff",
            "target_type": "final resolved outcome",
            "config": self.config.as_dict(),
            "rows": int(len(self.examples)),
            "market_timeseries_rows": int(len(self.market_timeseries)),
            "split_counts": {str(key): int(value) for key, value in split_counts.items()},
            "horizons_hours": [int(value) for value in sorted(self.examples["horizon_hours"].dropna().unique())],
            "example_columns": TERMINAL_EXAMPLE_COLUMNS,
            "market_timeseries_columns": TERMINAL_MARKET_TIMESERIES_COLUMNS,
        }
