"""Repricing benchmark built around a frozen manifest and normalized market histories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from polymarket_research.benchmarks.audit.reporting import (
    binary_label_stats,
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
from polymarket_research.benchmarks.utils.splits import select_split_rows
from polymarket_research.benchmarks.utils.time import normalize_utc_timestamp


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
    "split",
]

REPRICING_MARKET_TIMESERIES_COLUMNS = [
    "market_id",
    "timestamp_utc",
    "yes_probability",
]

REPRICING_TARGET_COLUMNS = [
    "market_id",
    "timestamp_utc",
    "future_horizon_hours",
    "label",
    "future_move",
    "split",
]

REPRICING_BUILD_COLUMNS = [
    *REPRICING_EXAMPLE_COLUMNS[:-1],
    "future_move",
    "label",
    "split",
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
    targets_frame: pd.DataFrame
    source: str = "polymarket"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", normalize_source_name(self.source))

    @property
    def release_name(self) -> str:
        return f"{self.source}-repricing-{int(self.config.future_horizon_hours)}h"

    @classmethod
    def example_columns(cls) -> list[str]:
        return list(REPRICING_EXAMPLE_COLUMNS)

    @classmethod
    def market_timeseries_columns(cls) -> list[str]:
        return list(REPRICING_MARKET_TIMESERIES_COLUMNS)

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

    def input_frame(self, *, split: str | None = None) -> pd.DataFrame:
        """Return leakage-safe repricing example inputs."""
        return select_split_rows(self.examples, split)

    def targets(self, *, split: str | None = None) -> pd.DataFrame:
        """Return repricing target rows."""
        return select_split_rows(self.targets_frame, split)

    def evaluate(
        self,
        predictions: pd.DataFrame,
        *,
        split: str = "test",
    ) -> dict[str, pd.DataFrame]:
        gold = select_split_rows(self.targets_frame, split)
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
                title=f"{source_display_name(self.source)} Repricing Benchmark",
                summary_lines=[
                    f"Source: `{self.source}`",
                    f"Release: `{self.release_name}`",
                    f"Examples: {manifest['rows']}",
                    f"Market-timeseries rows: {manifest['market_timeseries_rows']}",
                    f"Future horizon (hours): {manifest['future_horizon_hours']}",
                ],
                manifest=manifest,
            ),
            encoding="utf-8",
        )
        return paths

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "RepricingBenchmark":
        source_dir = Path(directory)
        manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
        source = normalize_source_name(manifest.get("source") or infer_source_from_release_path(source_dir))
        config_dict = dict(manifest.get("config", {}))
        if config_dict.get("split_timestamp_utc") is not None:
            config_dict["split_timestamp_utc"] = pd.Timestamp(config_dict["split_timestamp_utc"])

        config = RepricingBenchmarkConfig(**config_dict)
        examples = pd.read_parquet(source_dir / "examples.parquet")
        market_timeseries = pd.read_parquet(source_dir / "market_timeseries.parquet")
        targets_frame = pd.read_parquet(source_dir / "targets.parquet")
        raw_examples = examples.copy()

        if not examples.empty:
            examples["timestamp_utc"] = pd.to_datetime(examples["timestamp_utc"], utc=True, errors="coerce")
            examples["created_at"] = pd.to_datetime(examples["created_at"], utc=True, errors="coerce")
            examples["end_date"] = pd.to_datetime(examples["end_date"], utc=True, errors="coerce")
            examples["future_horizon_hours"] = pd.to_numeric(examples["future_horizon_hours"], errors="coerce").astype(int)
            examples["current_yes_probability"] = pd.to_numeric(examples["current_yes_probability"], errors="coerce")
            examples = examples.loc[:, REPRICING_EXAMPLE_COLUMNS]
        else:
            examples = pd.DataFrame(columns=REPRICING_EXAMPLE_COLUMNS)

        if not market_timeseries.empty:
            market_timeseries["timestamp_utc"] = pd.to_datetime(market_timeseries["timestamp_utc"], utc=True, errors="coerce")
            market_timeseries = market_timeseries.loc[:, REPRICING_MARKET_TIMESERIES_COLUMNS]
        else:
            market_timeseries = pd.DataFrame(columns=REPRICING_MARKET_TIMESERIES_COLUMNS)

        if not targets_frame.empty:
            targets_frame["timestamp_utc"] = pd.to_datetime(targets_frame["timestamp_utc"], utc=True, errors="coerce")
            targets_frame["future_horizon_hours"] = pd.to_numeric(targets_frame["future_horizon_hours"], errors="coerce").astype(int)
            if "future_move" not in targets_frame.columns and "future_move" in raw_examples.columns:
                future_moves = raw_examples.loc[
                    :,
                    ["market_id", "timestamp_utc", "future_horizon_hours", "future_move"],
                ].copy()
                future_moves["timestamp_utc"] = pd.to_datetime(future_moves["timestamp_utc"], utc=True, errors="coerce")
                future_moves["future_horizon_hours"] = pd.to_numeric(
                    future_moves["future_horizon_hours"],
                    errors="coerce",
                ).astype(int)
                future_moves["future_move"] = pd.to_numeric(future_moves["future_move"], errors="coerce")
                future_moves = future_moves.drop_duplicates(["market_id", "timestamp_utc", "future_horizon_hours"])
                targets_frame = targets_frame.merge(
                    future_moves,
                    on=["market_id", "timestamp_utc", "future_horizon_hours"],
                    how="left",
                )
            targets_frame["label"] = pd.to_numeric(targets_frame["label"], errors="coerce").astype(int)
            targets_frame["future_move"] = pd.to_numeric(targets_frame["future_move"], errors="coerce")
            targets_frame = targets_frame.loc[:, REPRICING_TARGET_COLUMNS]
        else:
            targets_frame = pd.DataFrame(columns=REPRICING_TARGET_COLUMNS)

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
            "name": "repricing_benchmark",
            "release_name": self.release_name,
            "task": "large_future_repricing_prediction",
            "observable_information": "target market metadata and market-level probability history up to the prediction timestamp",
            "target_type": "large future repricing indicator",
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
                    "all rolling repricing windows derived from one market_id inherit a single market-level split"
                ),
                "timestamp_key_definition": (
                    "first admissible repricing timestamp when split_on=timestamp_utc; market end_date when split_on=end_date"
                ),
            },
            "split_audit": split_audit(self.examples, split_unit_col="market_id", family_col="family_id"),
            "label_stats": binary_label_stats(self.targets_frame, label_col="label"),
            "future_horizon_hours": int(self.config.future_horizon_hours),
            "example_columns": REPRICING_EXAMPLE_COLUMNS,
            "market_timeseries_columns": REPRICING_MARKET_TIMESERIES_COLUMNS,
            "target_columns": REPRICING_TARGET_COLUMNS,
        }
