"""Terminal benchmark built around a frozen manifest and normalized market histories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from polymarket_research.benchmarks.audit.reporting import (
    binary_label_stats,
    counts_by_split_and_group,
    split_audit,
)
from polymarket_research.benchmarks.evaluation.metrics import evaluate_binary_predictions
from polymarket_research.benchmarks.utils.serialization import (
    SCHEMA_VERSION,
    render_bundle_readme,
    to_json_ready,
)
from polymarket_research.benchmarks.utils.sources import (
    infer_source_from_release_path,
    normalize_source_name,
    source_display_name,
)
from polymarket_research.benchmarks.utils.splits import assign_time_splits, select_split_rows
from polymarket_research.benchmarks.utils.time import normalize_utc_timestamp
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
    "prefix_rows",
    "cutoff_age_hours",
    "current_yes_probability",
    "current_probability_timestamp_utc",
    "current_probability_staleness_hours",
    "split",
]

TERMINAL_MARKET_TIMESERIES_COLUMNS = [
    "market_id",
    "timestamp_utc",
    "yes_probability",
]

TERMINAL_TARGET_COLUMNS = [
    "market_id",
    "horizon_hours",
    "label",
    "split",
]

TERMINAL_BUILD_COLUMNS = [
    *TERMINAL_EXAMPLE_COLUMNS[:-1],
    "label",
    "split",
]

TERMINAL_REQUIRED_PROBABILITY_COLUMNS = {"market_id", "timestamp_utc", "yes_probability"}
TERMINAL_REQUIRED_MARKET_COLUMNS = {"market_id", "created_at", "end_date", "final_yes_probability", "question"}


@dataclass(frozen=True)
class TerminalBenchmarkConfig:
    """Configuration for the frozen terminal benchmark."""

    horizons_hours: tuple[int, ...] = (24, 168, 336)
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
    targets_frame: pd.DataFrame
    source: str = "polymarket"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", normalize_source_name(self.source))

    @property
    def horizon_hours(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.config.horizons_hours)

    @property
    def release_name(self) -> str:
        if len(self.horizon_hours) == 1:
            return f"{self.source}-terminal-{self.horizon_hours[0]}h"
        horizon_tag = "-".join(f"{value}h" for value in self.horizon_hours)
        return f"{self.source}-terminal-{horizon_tag}"

    @classmethod
    def example_columns(cls) -> list[str]:
        return list(TERMINAL_EXAMPLE_COLUMNS)

    @classmethod
    def market_timeseries_columns(cls) -> list[str]:
        return list(TERMINAL_MARKET_TIMESERIES_COLUMNS)

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
        prepared["yes_probability"] = pd.to_numeric(prepared["yes_probability"], errors="coerce")
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
                current_yes_probability = float(market_history["yes_probability"].iloc[prefix_length - 1])
                if pd.isna(current_yes_probability):
                    continue

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
                        "prefix_rows": int(prefix_length),
                        "cutoff_age_hours": float((cutoff - created_at).total_seconds() / 3600.0),
                        "current_yes_probability": current_yes_probability,
                        "current_probability_timestamp_utc": last_timestamp,
                        "current_probability_staleness_hours": staleness_hours,
                        "label": int(float(market.final_yes_probability) >= 0.5),
                    }
                )

        if not rows:
            return pd.DataFrame(columns=TERMINAL_BUILD_COLUMNS)

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
        return examples.loc[:, TERMINAL_BUILD_COLUMNS]

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

    @staticmethod
    def _cutoff_feature_frame(examples: pd.DataFrame, market_timeseries: pd.DataFrame) -> pd.DataFrame:
        feature_columns = [
            "prefix_rows",
            "cutoff_age_hours",
            "current_yes_probability",
            "current_probability_timestamp_utc",
            "current_probability_staleness_hours",
        ]
        if examples.empty:
            return pd.DataFrame(columns=feature_columns, index=examples.index)

        features = pd.DataFrame(index=examples.index)
        features["prefix_rows"] = pd.NA
        features["cutoff_age_hours"] = (
            pd.to_datetime(examples["cutoff_timestamp_utc"], utc=True, errors="coerce")
            - pd.to_datetime(examples["created_at"], utc=True, errors="coerce")
        ).dt.total_seconds() / 3600.0
        features["current_yes_probability"] = pd.NA
        features["current_probability_timestamp_utc"] = pd.Series(
            pd.NaT,
            index=examples.index,
            dtype="datetime64[ns, UTC]",
        )
        features["current_probability_staleness_hours"] = pd.NA

        if market_timeseries.empty:
            return features

        histories = {
            str(market_id): market_history.sort_values("timestamp_utc", kind="stable")
            for market_id, market_history in market_timeseries.groupby("market_id", sort=False)
        }
        for market_id, market_examples in examples.groupby("market_id", sort=False):
            history = histories.get(str(market_id))
            if history is None or history.empty:
                continue

            timestamps = pd.DatetimeIndex(pd.to_datetime(history["timestamp_utc"], utc=True, errors="coerce"))
            probabilities = pd.to_numeric(history["yes_probability"], errors="coerce").to_numpy(dtype=float)
            cutoffs = pd.DatetimeIndex(pd.to_datetime(market_examples["cutoff_timestamp_utc"], utc=True, errors="coerce"))
            positions = timestamps.searchsorted(cutoffs, side="right") - 1
            valid = positions >= 0
            valid_index = market_examples.index[valid]
            if len(valid_index) == 0:
                continue

            observed_at = pd.Series(timestamps[positions[valid]], index=valid_index)
            cutoff_series = pd.Series(cutoffs[valid], index=valid_index)
            features.loc[valid_index, "prefix_rows"] = positions[valid] + 1
            features.loc[valid_index, "current_yes_probability"] = probabilities[positions[valid]]
            features.loc[valid_index, "current_probability_timestamp_utc"] = observed_at
            features.loc[valid_index, "current_probability_staleness_hours"] = (
                cutoff_series - observed_at
            ).dt.total_seconds() / 3600.0

        return features

    @classmethod
    def _ensure_example_observable_columns(
        cls,
        examples: pd.DataFrame,
        market_timeseries: pd.DataFrame,
    ) -> pd.DataFrame:
        missing_columns = [column for column in TERMINAL_EXAMPLE_COLUMNS if column not in examples.columns]
        if not missing_columns:
            return examples.loc[:, TERMINAL_EXAMPLE_COLUMNS]

        enriched = examples.copy()
        cutoff_features = cls._cutoff_feature_frame(enriched, market_timeseries)
        for column in cutoff_features.columns:
            if column not in enriched.columns:
                enriched[column] = cutoff_features[column]
        return enriched.loc[:, TERMINAL_EXAMPLE_COLUMNS]

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

    def history_until(self, market_id: str, timestamp_utc: pd.Timestamp | str) -> pd.DataFrame:
        """Return a leakage-safe market history prefix ending at ``timestamp_utc``."""
        ts = normalize_utc_timestamp(pd.Timestamp(timestamp_utc))
        history = self.market_history(market_id)
        return history.loc[history["timestamp_utc"] <= ts].reset_index(drop=True).copy()

    def input_frame(self, *, split: str | None = None) -> pd.DataFrame:
        """Return leakage-safe terminal example inputs."""
        return select_split_rows(self.examples, split)

    def targets(self, *, split: str | None = None) -> pd.DataFrame:
        """Return terminal target rows."""
        return select_split_rows(self.targets_frame, split)

    def _market_timeseries_slices(self) -> dict[str, slice]:
        cached = getattr(self, "_market_timeseries_slice_cache", None)
        if cached is not None:
            return cached

        market_ids = self.market_timeseries["market_id"].to_numpy()
        if len(market_ids) == 0:
            slices: dict[str, slice] = {}
            object.__setattr__(self, "_market_timeseries_slice_cache", slices)
            return slices

        boundary_positions = np.flatnonzero(market_ids[1:] != market_ids[:-1]) + 1
        starts = np.concatenate(([0], boundary_positions))
        ends = np.concatenate((boundary_positions, [len(market_ids)]))

        slices = {}
        for start, end in zip(starts, ends, strict=True):
            market_id = str(market_ids[start])
            if market_id in slices:
                raise ValueError("market_timeseries must be grouped by market_id in contiguous blocks.")
            slices[market_id] = slice(int(start), int(end))

        object.__setattr__(self, "_market_timeseries_slice_cache", slices)
        return slices

    def market_cutoff_probabilities(self, *, split: str | None = None) -> pd.DataFrame:
        """Return market-implied probabilities from the latest observation at each terminal cutoff."""
        cache_key = split or "__all__"
        cached = getattr(self, "_market_cutoff_probability_cache", None)
        if cached is not None and cache_key in cached:
            return cached[cache_key].copy()

        examples = select_split_rows(self.examples, split).copy()
        if examples.empty:
            return pd.DataFrame(columns=["market_id", "horizon_hours", "market_pred_prob"])

        if "current_yes_probability" in examples.columns:
            out = examples.loc[:, ["market_id", "horizon_hours", "current_yes_probability"]].copy()
            out = out.rename(columns={"current_yes_probability": "market_pred_prob"})
            out["market_pred_prob"] = pd.to_numeric(out["market_pred_prob"], errors="coerce")
            if not out["market_pred_prob"].isna().any():
                out = out.sort_values(["market_id", "horizon_hours"], kind="stable").reset_index(drop=True)
                next_cache = dict(cached or {})
                next_cache[cache_key] = out
                object.__setattr__(self, "_market_cutoff_probability_cache", next_cache)
                return out.copy()

        examples = examples.loc[:, ["market_id", "horizon_hours", "cutoff_timestamp_utc"]].copy()

        examples["market_id"] = examples["market_id"].astype(str)
        examples["horizon_hours"] = pd.to_numeric(examples["horizon_hours"], errors="coerce").astype(int)
        examples["cutoff_timestamp_utc"] = pd.to_datetime(examples["cutoff_timestamp_utc"], utc=True, errors="coerce")

        market_slices = self._market_timeseries_slices()
        rows: list[pd.DataFrame] = []
        missing = 0
        for market_id, market_examples in examples.groupby("market_id", sort=False):
            market_slice = market_slices.get(str(market_id))
            if market_slice is None:
                missing += len(market_examples)
                continue

            market_history = self.market_timeseries.iloc[market_slice]
            timestamps = pd.DatetimeIndex(pd.to_datetime(market_history["timestamp_utc"], utc=True, errors="coerce"))
            probabilities = pd.to_numeric(market_history["yes_probability"], errors="coerce").to_numpy(dtype=float)

            cutoffs = pd.DatetimeIndex(market_examples["cutoff_timestamp_utc"])
            positions = timestamps.searchsorted(cutoffs, side="right") - 1
            valid = positions >= 0
            if not bool(np.all(valid)):
                missing += int(np.sum(~valid))

            out = market_examples.loc[valid, ["market_id", "horizon_hours"]].copy()
            out["market_pred_prob"] = probabilities[positions[valid]]
            rows.append(out)

        if missing:
            raise ValueError(f"Missing pre-cutoff probabilities for {missing} terminal examples.")

        if not rows:
            out = pd.DataFrame(columns=["market_id", "horizon_hours", "market_pred_prob"])
        else:
            out = pd.concat(rows, ignore_index=True)
            if out["market_pred_prob"].isna().any():
                missing = int(out["market_pred_prob"].isna().sum())
                raise ValueError(f"Missing pre-cutoff probabilities for {missing} terminal examples.")
            out = out.sort_values(["market_id", "horizon_hours"], kind="stable").reset_index(drop=True)

        next_cache = dict(cached or {})
        next_cache[cache_key] = out
        object.__setattr__(self, "_market_cutoff_probability_cache", next_cache)
        return out.copy()

    def evaluate(
        self,
        predictions: pd.DataFrame | pd.Series,
        split: str = "test",
    ) -> dict[str, pd.DataFrame]:
        gold = select_split_rows(self.targets_frame, split)
        market_probabilities = self.market_cutoff_probabilities(split=split)
        gold = gold.merge(market_probabilities, on=["market_id", "horizon_hours"], how="left")
        if gold["market_pred_prob"].isna().any():
            missing = int(gold["market_pred_prob"].isna().sum())
            raise ValueError(f"Missing market cutoff probabilities for {missing} terminal examples.")
        return evaluate_binary_predictions(
            predictions=predictions,
            gold=gold,
            split=split,
            group_col="horizon_hours",
            id_columns=("market_id", "horizon_hours"),
            reference_prob_col="market_pred_prob",
        )

    def save(self, directory: str | Path) -> dict[str, Path]:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "examples": out_dir / "examples.parquet",
            "market_timeseries": out_dir / "market_timeseries.parquet",
            "targets": out_dir / "targets.parquet",
            "manifest": out_dir / "manifest.json",
            "readme": out_dir / "README.md",
        }
        self.examples.to_parquet(paths["examples"], index=False)
        self.market_timeseries.to_parquet(paths["market_timeseries"], index=False)
        self.targets_frame.to_parquet(paths["targets"], index=False)
        manifest = self.manifest()
        paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        paths["readme"].write_text(
            render_bundle_readme(
                title=f"{source_display_name(self.source)} Terminal Benchmark",
                summary_lines=[
                    f"Source: `{self.source}`",
                    f"Release: `{self.release_name}`",
                    f"Examples: {manifest['rows']}",
                    f"Market-timeseries rows: {manifest['market_timeseries_rows']}",
                    f"Split policy: `{self.config.split_on}`",
                ],
                manifest=manifest,
            ),
            encoding="utf-8",
        )
        return paths

    @classmethod
    def load(cls, directory: str | Path) -> "TerminalBenchmark":
        source_dir = Path(directory)
        manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
        source = normalize_source_name(manifest.get("source") or infer_source_from_release_path(source_dir))
        config_dict = dict(manifest.get("config", {}))
        if config_dict.get("split_timestamp_utc") is not None:
            config_dict["split_timestamp_utc"] = pd.Timestamp(config_dict["split_timestamp_utc"])

        config = TerminalBenchmarkConfig(**config_dict)
        examples = pd.read_parquet(source_dir / "examples.parquet")
        market_timeseries = pd.read_parquet(source_dir / "market_timeseries.parquet")
        targets_frame = pd.read_parquet(source_dir / "targets.parquet")

        if not market_timeseries.empty:
            market_timeseries["timestamp_utc"] = pd.to_datetime(market_timeseries["timestamp_utc"], utc=True, errors="coerce")
            market_timeseries["yes_probability"] = pd.to_numeric(market_timeseries["yes_probability"], errors="coerce")
            market_timeseries = market_timeseries.loc[:, TERMINAL_MARKET_TIMESERIES_COLUMNS]
        else:
            market_timeseries = pd.DataFrame(columns=TERMINAL_MARKET_TIMESERIES_COLUMNS)

        if not examples.empty:
            examples["created_at"] = pd.to_datetime(examples["created_at"], utc=True, errors="coerce")
            examples["end_date"] = pd.to_datetime(examples["end_date"], utc=True, errors="coerce")
            examples["cutoff_timestamp_utc"] = pd.to_datetime(examples["cutoff_timestamp_utc"], utc=True, errors="coerce")
            if "current_probability_timestamp_utc" in examples.columns:
                examples["current_probability_timestamp_utc"] = pd.to_datetime(
                    examples["current_probability_timestamp_utc"],
                    utc=True,
                    errors="coerce",
                )
            examples = cls._ensure_example_observable_columns(examples, market_timeseries)
            for column in ["prefix_rows", "horizon_hours"]:
                if column in examples.columns:
                    examples[column] = pd.to_numeric(examples[column], errors="coerce").astype(int)
            for column in ["cutoff_age_hours", "current_yes_probability", "current_probability_staleness_hours"]:
                if column in examples.columns:
                    examples[column] = pd.to_numeric(examples[column], errors="coerce")
        else:
            examples = pd.DataFrame(columns=TERMINAL_EXAMPLE_COLUMNS)

        if not targets_frame.empty:
            targets_frame["horizon_hours"] = pd.to_numeric(targets_frame["horizon_hours"], errors="coerce").astype(int)
            targets_frame["label"] = pd.to_numeric(targets_frame["label"], errors="coerce").astype(int)
            targets_frame = targets_frame.loc[:, TERMINAL_TARGET_COLUMNS]
        else:
            targets_frame = pd.DataFrame(columns=TERMINAL_TARGET_COLUMNS)

        return cls(
            config=config,
            examples=examples,
            market_timeseries=market_timeseries,
            targets_frame=targets_frame,
            source=source,
        )

    def manifest(self) -> dict[str, Any]:
        split_counts = self.examples["split"].value_counts(dropna=False).sort_index().to_dict()
        families = (
            self.examples["family_id"].dropna().astype(str).str.strip()
            if "family_id" in self.examples.columns
            else pd.Series(dtype="string")
        )
        families = families.loc[families != ""]
        return {
            "schema_version": SCHEMA_VERSION,
            "source": self.source,
            "name": "terminal_benchmark",
            "release_name": self.release_name,
            "task": "terminal_outcome_prediction",
            "observable_information": "target market metadata and market-level probability history up to cutoff",
            "target_type": "final resolved outcome",
            "config": self.config.as_dict(),
            "rows": int(len(self.examples)),
            "markets": int(self.examples["market_id"].nunique()) if not self.examples.empty else 0,
            "families": int(families.nunique()) if not families.empty else 0,
            "market_timeseries_rows": int(len(self.market_timeseries)),
            "split_counts": {str(key): int(value) for key, value in split_counts.items()},
            "split_policy": {
                "split_unit": "market_id",
                "split_on": str(self.config.split_on),
                "assignment_rule": (
                    "all horizons derived from one market_id inherit the market-level split computed "
                    "from a single timestamp key"
                ),
                "timestamp_key_definition": (
                    "market end_date when split_on=end_date; market cutoff timestamp when split_on=cutoff_timestamp_utc"
                ),
            },
            "split_audit": split_audit(self.examples, split_unit_col="market_id", family_col="family_id"),
            "label_stats": binary_label_stats(self.targets_frame, label_col="label"),
            "rows_by_horizon_and_split": counts_by_split_and_group(self.examples, group_col="horizon_hours"),
            "horizons_hours": [int(value) for value in sorted(self.examples["horizon_hours"].dropna().unique())],
            "example_columns": TERMINAL_EXAMPLE_COLUMNS,
            "market_timeseries_columns": TERMINAL_MARKET_TIMESERIES_COLUMNS,
            "target_columns": TERMINAL_TARGET_COLUMNS,
        }
