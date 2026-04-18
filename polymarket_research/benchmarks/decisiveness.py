"""Durable-decisiveness benchmark built around frozen market prefixes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from polymarket_research.benchmarks.common import (
    assign_time_splits,
    evaluate_multiclass_predictions,
    evaluate_regression_predictions,
    format_decisiveness_example_ids,
    normalize_utc_timestamp,
    to_json_ready,
)
from polymarket_research.benchmarks.tabular import TabularBenchmark
from polymarket_research.data.canonical.dataset import CanonicalDataset


DECISIVENESS_EXAMPLE_COLUMNS = [
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
    "decisive_side",
    "decisive_timestamp_utc",
    "hours_to_decisive",
    "label",
    "label_name",
    "split",
]

DECISIVENESS_MARKET_TIMESERIES_COLUMNS = [
    "market_id",
    "timestamp_utc",
    "yes_probability",
]

DECISIVENESS_REQUIRED_PROBABILITY_COLUMNS = {"market_id", "timestamp_utc", "yes_probability"}
DECISIVENESS_REQUIRED_MARKET_COLUMNS = {"market_id", "created_at", "end_date", "question"}


@dataclass(frozen=True)
class DecisivenessBenchmarkConfig:
    """Configuration for the durable-decisiveness benchmark."""

    decisive_threshold: float = 0.95
    sample_every_hours: int = 12
    min_history_points: int = 24
    min_prefix_age_hours: float = 6.0
    min_time_to_decisive_hours: float = 1.0
    ordinal_bin_edges_hours: tuple[float, ...] = (24.0, 72.0)
    ordinal_bin_labels: tuple[str, ...] = ("short", "medium", "long")
    target_market_only: bool = True
    split_on: str = "decisive_timestamp_utc"
    split_timestamp_utc: pd.Timestamp | None = None
    train_fraction: float = 0.8
    show_progress: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "decisive_threshold", float(self.decisive_threshold))
        object.__setattr__(self, "sample_every_hours", int(self.sample_every_hours))
        object.__setattr__(self, "min_history_points", int(self.min_history_points))
        object.__setattr__(self, "min_prefix_age_hours", float(self.min_prefix_age_hours))
        object.__setattr__(self, "min_time_to_decisive_hours", float(self.min_time_to_decisive_hours))
        object.__setattr__(
            self,
            "ordinal_bin_edges_hours",
            tuple(float(edge) for edge in self.ordinal_bin_edges_hours),
        )
        object.__setattr__(
            self,
            "ordinal_bin_labels",
            tuple(str(label) for label in self.ordinal_bin_labels),
        )
        object.__setattr__(self, "split_timestamp_utc", normalize_utc_timestamp(self.split_timestamp_utc))

        if not 0.5 < self.decisive_threshold < 1.0:
            raise ValueError("decisive_threshold must lie strictly between 0.5 and 1.0.")
        if self.sample_every_hours <= 0:
            raise ValueError("sample_every_hours must be positive.")
        if self.min_history_points <= 0:
            raise ValueError("min_history_points must be positive.")
        if self.min_prefix_age_hours < 0.0:
            raise ValueError("min_prefix_age_hours must be non-negative.")
        if self.min_time_to_decisive_hours <= 0.0:
            raise ValueError("min_time_to_decisive_hours must be positive.")
        if tuple(sorted(self.ordinal_bin_edges_hours)) != self.ordinal_bin_edges_hours:
            raise ValueError("ordinal_bin_edges_hours must be sorted in increasing order.")
        if len(self.ordinal_bin_labels) != len(self.ordinal_bin_edges_hours) + 1:
            raise ValueError("ordinal_bin_labels must have exactly one more entry than ordinal_bin_edges_hours.")

    def as_dict(self) -> dict[str, Any]:
        return to_json_ready(asdict(self))

    def label_for_hours(self, hours_to_decisive: float) -> tuple[int, str]:
        for index, edge in enumerate(self.ordinal_bin_edges_hours):
            if float(hours_to_decisive) <= float(edge):
                return index, self.ordinal_bin_labels[index]
        return len(self.ordinal_bin_labels) - 1, self.ordinal_bin_labels[-1]


@dataclass(frozen=True)
class DecisivenessBenchmark:
    """Frozen decisive-belief manifest plus normalized market-level probability histories."""

    config: DecisivenessBenchmarkConfig
    examples: pd.DataFrame
    market_timeseries: pd.DataFrame
    canonical: CanonicalDataset | None = None

    @property
    def release_name(self) -> str:
        tau_tag = str(int(round(self.config.decisive_threshold * 100.0)))
        return f"polymarket-decisive-belief-tau{tau_tag}"

    @classmethod
    def build(
        cls,
        canonical: CanonicalDataset,
        *,
        config: DecisivenessBenchmarkConfig | None = None,
    ) -> "DecisivenessBenchmark":
        cfg = config or DecisivenessBenchmarkConfig()
        cls._log(
            cfg,
            "starting "
            f"(markets={len(canonical.markets)}, probability_rows={len(canonical.probabilities)}, "
            f"tau={cfg.decisive_threshold:.2f}, split_on={cfg.split_on})",
        )
        cls._validate_canonical(canonical)
        cls._log(cfg, "validated canonical tables")
        markets = cls._prepare_markets(canonical.markets)
        probabilities = cls._prepare_probabilities(canonical.probabilities)
        cls._log(cfg, f"prepared inputs (markets={len(markets)}, probability_rows={len(probabilities)})")

        grouped_probabilities = {
            market_id: frame.reset_index(drop=True)
            for market_id, frame in probabilities.groupby("market_id", sort=False)
        }
        cls._log(cfg, f"indexed market histories (markets_with_history={len(grouped_probabilities)})")

        rows: list[dict[str, object]] = []
        market_iter = markets.itertuples(index=False)
        if cfg.show_progress:
            market_iter = tqdm(market_iter, total=len(markets), desc="decisiveness examples", unit="market")
        for market in market_iter:
            market_id = str(market.market_id)
            market_history = grouped_probabilities.get(market_id)
            if market_history is None or market_history.empty:
                continue

            decisive_entry = cls._durable_decisive_entry(market_history, threshold=cfg.decisive_threshold)
            if decisive_entry is None:
                continue
            decisive_idx, decisive_timestamp_utc, decisive_side = decisive_entry

            sampled_indices = cls._sample_cutoff_indices(
                market_history,
                decisive_index=decisive_idx,
                created_at=pd.Timestamp(market.created_at),
                config=cfg,
            )
            if not sampled_indices:
                continue

            for cutoff_index in sampled_indices:
                cutoff_row = market_history.iloc[int(cutoff_index)]
                cutoff_timestamp_utc = pd.Timestamp(cutoff_row["timestamp_utc"])
                current_probability = float(cutoff_row["yes_probability"])
                hours_to_decisive = float((decisive_timestamp_utc - cutoff_timestamp_utc).total_seconds() / 3600.0)
                label, label_name = cfg.label_for_hours(hours_to_decisive)
                rows.append(
                    {
                        "market_id": market_id,
                        "cutoff_timestamp_utc": cutoff_timestamp_utc,
                        "market_slug": getattr(market, "market_slug", None),
                        "question": getattr(market, "question", None),
                        "created_at": pd.Timestamp(market.created_at),
                        "end_date": pd.Timestamp(market.end_date),
                        "platform_category": getattr(market, "platform_category", None),
                        "research_category": getattr(market, "research_category", None),
                        "family_id": getattr(market, "family_id", None),
                        "prefix_rows": int(cutoff_index) + 1,
                        "cutoff_age_hours": float((cutoff_timestamp_utc - pd.Timestamp(market.created_at)).total_seconds() / 3600.0),
                        "current_yes_probability": current_probability,
                        "confidence_margin": abs(current_probability - 0.5),
                        "distance_to_yes_decisive": max(float(cfg.decisive_threshold) - current_probability, 0.0),
                        "distance_to_no_decisive": max(current_probability - float(1.0 - cfg.decisive_threshold), 0.0),
                        "decisive_threshold": float(cfg.decisive_threshold),
                        "decisive_side": decisive_side,
                        "decisive_timestamp_utc": decisive_timestamp_utc,
                        "hours_to_decisive": hours_to_decisive,
                        "label": int(label),
                        "label_name": label_name,
                    }
                )

        if not rows:
            examples = pd.DataFrame(columns=DECISIVENESS_EXAMPLE_COLUMNS)
            market_timeseries = pd.DataFrame(columns=DECISIVENESS_MARKET_TIMESERIES_COLUMNS)
            return cls(config=cfg, examples=examples, market_timeseries=market_timeseries, canonical=canonical)

        examples = pd.DataFrame(rows)
        split_source = (
            examples.loc[:, ["market_id", cfg.split_on]]
            .drop_duplicates(subset=["market_id"])
            .reset_index(drop=True)
        )
        split_source["split"] = assign_time_splits(
            split_source,
            split_on=cfg.split_on,
            valid_columns={"decisive_timestamp_utc", "end_date"},
            split_timestamp_utc=cfg.split_timestamp_utc,
            train_fraction=cfg.train_fraction,
        )
        examples = examples.merge(split_source.loc[:, ["market_id", "split"]], on="market_id", how="left")
        examples["_split_order"] = examples["split"].map({"train": 0, "test": 1}).fillna(2)
        examples = examples.sort_values(
            ["_split_order", "cutoff_timestamp_utc", "market_id"],
            kind="stable",
        ).reset_index(drop=True)
        examples = examples.loc[:, DECISIVENESS_EXAMPLE_COLUMNS]
        cls._log(
            cfg,
            "built example manifest "
            f"(examples={len(examples)}, train={int((examples['split'] == 'train').sum())}, "
            f"test={int((examples['split'] == 'test').sum())}, markets={examples['market_id'].nunique()})",
        )

        market_ids = set(examples["market_id"].astype(str))
        market_timeseries = probabilities.loc[
            probabilities["market_id"].astype(str).isin(market_ids),
            DECISIVENESS_MARKET_TIMESERIES_COLUMNS,
        ].reset_index(drop=True)
        cls._log(cfg, f"built market histories (market_timeseries_rows={len(market_timeseries)})")
        cls._log(cfg, "done")
        return cls(config=cfg, examples=examples, market_timeseries=market_timeseries, canonical=canonical)

    @classmethod
    def from_canonical(
        cls,
        canonical: CanonicalDataset,
        *,
        config: DecisivenessBenchmarkConfig | None = None,
    ) -> "DecisivenessBenchmark":
        return cls.build(canonical, config=config)

    @staticmethod
    def _log(config: DecisivenessBenchmarkConfig, message: str) -> None:
        if config.show_progress:
            print(f"[decisiveness benchmark] {message}")

    @staticmethod
    def _validate_canonical(canonical: CanonicalDataset) -> None:
        market_cols = set(canonical.markets.columns)
        probability_cols = set(canonical.probabilities.columns)
        missing_market = sorted(DECISIVENESS_REQUIRED_MARKET_COLUMNS - market_cols)
        missing_probability = sorted(DECISIVENESS_REQUIRED_PROBABILITY_COLUMNS - probability_cols)
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
        prepared = prepared.loc[prepared["created_at"].notna() & prepared["end_date"].notna()].reset_index(drop=True)
        return prepared

    @staticmethod
    def _prepare_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
        prepared = probabilities.loc[:, DECISIVENESS_MARKET_TIMESERIES_COLUMNS].copy()
        prepared["market_id"] = prepared["market_id"].astype(str)
        prepared["timestamp_utc"] = pd.to_datetime(prepared["timestamp_utc"], utc=True, errors="coerce")
        prepared["yes_probability"] = pd.to_numeric(prepared["yes_probability"], errors="coerce")
        prepared = prepared.loc[
            prepared["timestamp_utc"].notna() & prepared["yes_probability"].notna(),
            DECISIVENESS_MARKET_TIMESERIES_COLUMNS,
        ].copy()
        prepared = prepared.sort_values(["market_id", "timestamp_utc"], kind="stable").reset_index(drop=True)
        return prepared

    @staticmethod
    def _durable_decisive_entry(
        market_history: pd.DataFrame,
        *,
        threshold: float,
    ) -> tuple[int, pd.Timestamp, str] | None:
        history = market_history.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
        if history.empty:
            return None

        yes_mask = history["yes_probability"] >= float(threshold)
        no_mask = history["yes_probability"] <= float(1.0 - threshold)

        if bool(yes_mask.iloc[-1]):
            durable_mask = yes_mask.to_numpy(dtype=bool)
            decisive_side = "yes"
        elif bool(no_mask.iloc[-1]):
            durable_mask = no_mask.to_numpy(dtype=bool)
            decisive_side = "no"
        else:
            return None

        decisive_index = len(history) - 1
        while decisive_index > 0 and durable_mask[decisive_index - 1]:
            decisive_index -= 1
        decisive_timestamp_utc = pd.Timestamp(history.iloc[decisive_index]["timestamp_utc"])
        return decisive_index, decisive_timestamp_utc, decisive_side

    @staticmethod
    def _sample_cutoff_indices(
        market_history: pd.DataFrame,
        *,
        decisive_index: int,
        created_at: pd.Timestamp,
        config: DecisivenessBenchmarkConfig,
    ) -> list[int]:
        if decisive_index <= 0:
            return []

        candidate_history = market_history.iloc[:decisive_index].copy()
        candidate_history["prefix_rows"] = range(1, len(candidate_history) + 1)
        decisive_timestamp_utc = pd.Timestamp(market_history.iloc[decisive_index]["timestamp_utc"])
        candidate_history["cutoff_age_hours"] = (
            pd.to_datetime(candidate_history["timestamp_utc"], utc=True) - created_at
        ).dt.total_seconds() / 3600.0
        candidate_history["hours_to_decisive"] = (
            decisive_timestamp_utc - pd.to_datetime(candidate_history["timestamp_utc"], utc=True)
        ).dt.total_seconds() / 3600.0

        eligible = candidate_history.loc[
            (candidate_history["prefix_rows"] >= int(config.min_history_points))
            & (candidate_history["cutoff_age_hours"] >= float(config.min_prefix_age_hours))
            & (candidate_history["hours_to_decisive"] >= float(config.min_time_to_decisive_hours))
        ].copy()
        if eligible.empty:
            return []

        sampled_indices: list[int] = []
        last_kept_timestamp: pd.Timestamp | None = None
        for row in eligible.itertuples():
            row_timestamp = pd.Timestamp(row.timestamp_utc)
            if last_kept_timestamp is None:
                sampled_indices.append(int(row.Index))
                last_kept_timestamp = row_timestamp
                continue

            elapsed_hours = float((row_timestamp - last_kept_timestamp).total_seconds() / 3600.0)
            if elapsed_hours >= float(config.sample_every_hours):
                sampled_indices.append(int(row.Index))
                last_kept_timestamp = row_timestamp

        final_index = int(eligible.index[-1])
        if final_index not in sampled_indices:
            sampled_indices.append(final_index)
        return sampled_indices

    def split_examples(self, split: str) -> pd.DataFrame:
        return self.examples.loc[self.examples["split"] == split].reset_index(drop=True).copy()

    def targets(self, split: str | None = None) -> pd.DataFrame:
        frame = self.examples if split is None else self.examples.loc[self.examples["split"] == split]
        return (
            frame.loc[
                :,
                [
                    "market_id",
                    "cutoff_timestamp_utc",
                    "label",
                    "label_name",
                    "hours_to_decisive",
                    "decisive_side",
                    "split",
                ],
            ]
            .reset_index(drop=True)
            .copy()
        )

    def resolve_market_snapshot(self, market_id: str, cutoff_timestamp_utc: pd.Timestamp | str) -> pd.Series:
        cutoff_ts = normalize_utc_timestamp(pd.Timestamp(cutoff_timestamp_utc))
        match = self.examples.loc[
            (self.examples["market_id"] == str(market_id))
            & (pd.to_datetime(self.examples["cutoff_timestamp_utc"], utc=True) == cutoff_ts)
        ]
        if match.empty:
            raise KeyError(f"Unknown decisiveness snapshot for market_id={market_id} at cutoff_timestamp_utc={cutoff_ts}")
        return match.iloc[0].copy()

    def market_history(self, market_id: str) -> pd.DataFrame:
        history = self.market_timeseries.loc[self.market_timeseries["market_id"] == str(market_id)].reset_index(drop=True).copy()
        if history.empty:
            raise KeyError(f"Missing market history for market_id={market_id}")
        return history

    def history_until(self, market_id: str, cutoff_timestamp_utc: pd.Timestamp | str) -> pd.DataFrame:
        cutoff_ts = normalize_utc_timestamp(pd.Timestamp(cutoff_timestamp_utc))
        history = self.market_history(market_id)
        return history.loc[history["timestamp_utc"] <= cutoff_ts].reset_index(drop=True).copy()

    def evaluate(
        self,
        predictions: pd.DataFrame | pd.Series,
        *,
        split: str = "test",
    ) -> dict[str, pd.DataFrame]:
        target_frame = self.targets(split=split).copy()
        results = evaluate_multiclass_predictions(
            gold=target_frame.loc[:, ["market_id", "cutoff_timestamp_utc", "label", "decisive_side"]],
            predictions=predictions,
            split=split,
            group_col="decisive_side",
            id_columns=("market_id", "cutoff_timestamp_utc"),
        )

        if isinstance(predictions, pd.DataFrame) and "pred_hours_to_decisive" in predictions.columns:
            regression = evaluate_regression_predictions(
                gold=target_frame.loc[:, ["market_id", "cutoff_timestamp_utc", "hours_to_decisive", "decisive_side"]],
                predictions=predictions,
                split=split,
                value_col="hours_to_decisive",
                pred_col="pred_hours_to_decisive",
                group_col="decisive_side",
                id_columns=("market_id", "cutoff_timestamp_utc"),
            )
            results["continuous_overall"] = regression["overall"]
            if "by_decisive_side" in regression:
                results["continuous_by_side"] = regression["by_decisive_side"]
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
    ) -> "DecisivenessBenchmark":
        source_dir = Path(directory)
        manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
        config_dict = dict(manifest.get("config", {}))
        if config_dict.get("split_timestamp_utc") is not None:
            config_dict["split_timestamp_utc"] = pd.Timestamp(config_dict["split_timestamp_utc"])

        config = DecisivenessBenchmarkConfig(**config_dict)
        examples = pd.read_parquet(source_dir / "examples.parquet")
        market_timeseries = pd.read_parquet(source_dir / "market_timeseries.parquet")

        if not examples.empty:
            for column in ("created_at", "end_date", "cutoff_timestamp_utc", "decisive_timestamp_utc"):
                examples[column] = pd.to_datetime(examples[column], utc=True, errors="coerce")
            examples["label"] = pd.to_numeric(examples["label"], errors="coerce").astype(int)
            examples["hours_to_decisive"] = pd.to_numeric(examples["hours_to_decisive"], errors="coerce")
            examples = examples.loc[:, DECISIVENESS_EXAMPLE_COLUMNS]
        else:
            examples = pd.DataFrame(columns=DECISIVENESS_EXAMPLE_COLUMNS)

        if not market_timeseries.empty:
            market_timeseries["timestamp_utc"] = pd.to_datetime(market_timeseries["timestamp_utc"], utc=True, errors="coerce")
            market_timeseries["yes_probability"] = pd.to_numeric(market_timeseries["yes_probability"], errors="coerce")
            market_timeseries = market_timeseries.loc[:, DECISIVENESS_MARKET_TIMESERIES_COLUMNS]
        else:
            market_timeseries = pd.DataFrame(columns=DECISIVENESS_MARKET_TIMESERIES_COLUMNS)

        return cls(config=config, examples=examples, market_timeseries=market_timeseries, canonical=canonical)

    def manifest(self) -> dict[str, Any]:
        split_counts = self.examples["split"].value_counts(dropna=False).sort_index().to_dict()
        return {
            "name": "decisiveness_benchmark",
            "release_name": self.release_name,
            "task": "durable_decisive_belief_formation_prediction",
            "observable_information": "target market metadata and market-level probability history up to the cutoff timestamp",
            "target_type": "ordinal decisive-horizon label plus continuous hours-to-decisive auxiliary target",
            "config": self.config.as_dict(),
            "rows": int(len(self.examples)),
            "markets": int(self.examples["market_id"].nunique()) if not self.examples.empty else 0,
            "market_timeseries_rows": int(len(self.market_timeseries)),
            "split_counts": {str(key): int(value) for key, value in split_counts.items()},
            "ordinal_bin_edges_hours": [float(edge) for edge in self.config.ordinal_bin_edges_hours],
            "ordinal_bin_labels": list(self.config.ordinal_bin_labels),
            "example_columns": DECISIVENESS_EXAMPLE_COLUMNS,
            "market_timeseries_columns": DECISIVENESS_MARKET_TIMESERIES_COLUMNS,
        }

    def _examples_for_view(self) -> pd.DataFrame:
        frame = self.examples.loc[:, ["market_id", "cutoff_timestamp_utc", "split"]].copy()
        frame["example_id"] = format_decisiveness_example_ids(frame)
        return frame

    def build_reference_view(self) -> pd.DataFrame:
        return self.view().frame.copy()

    def view(self):
        from polymarket_research.views import DecisivenessBenchmarkView

        return DecisivenessBenchmarkView.from_examples(self.examples, config=self.config)

    def tabular(self) -> TabularBenchmark:
        return TabularBenchmark.from_view(self.view(), evaluation_group_col="decisive_side")
