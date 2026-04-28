"""Build the repricing benchmark and local analysis frames from the canonical layer."""

from __future__ import annotations

import pandas as pd

from polymarket_research.benchmarks.builders.common import attach_analysis_columns
from polymarket_research.benchmarks.schemas.repricing import (
    REPRICING_BUILD_COLUMNS,
    REPRICING_TARGET_COLUMNS,
    RepricingBenchmark,
    RepricingBenchmarkConfig,
)
from polymarket_research.benchmarks.utils.ids import format_repricing_example_ids
from polymarket_research.benchmarks.utils.splits import assign_group_time_splits
from polymarket_research.data.canonical.dataset import CanonicalDataset
from polymarket_research.data.representations.repricing import RepricingPanelBuilder


def build_repricing_from_canonical(
    canonical: CanonicalDataset,
    config: RepricingBenchmarkConfig | None = None,
    *,
    source: str = "polymarket",
) -> RepricingBenchmark:
    """Materialize a frozen repricing benchmark from a canonical dataset."""
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
        examples = pd.DataFrame(columns=RepricingBenchmark.example_columns())
        market_timeseries = pd.DataFrame(columns=RepricingBenchmark.market_timeseries_columns())
        targets_frame = pd.DataFrame(columns=REPRICING_TARGET_COLUMNS)
        return RepricingBenchmark(
            config=cfg,
            examples=examples,
            market_timeseries=market_timeseries,
            targets_frame=targets_frame,
            source=source,
        )

    build_frame = panel.loc[
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
    build_frame = build_frame.rename(columns={"target": "label"})
    build_frame["label"] = build_frame["label"].astype(int)
    build_frame["split"] = assign_group_time_splits(
        build_frame,
        group_col="market_id",
        split_on=cfg.split_on,
        valid_columns={"timestamp_utc", "end_date"},
        split_timestamp_utc=cfg.split_timestamp_utc,
        train_fraction=cfg.train_fraction,
        group_timestamp_agg="min",
    )
    build_frame["_split_order"] = build_frame["split"].map({"train": 0, "test": 1}).fillna(2)
    build_frame = build_frame.sort_values(
        ["_split_order", "timestamp_utc", "market_id"],
        kind="stable",
    ).reset_index(drop=True)
    build_frame = build_frame.loc[:, REPRICING_BUILD_COLUMNS]
    examples = build_frame.loc[:, RepricingBenchmark.example_columns()].reset_index(drop=True).copy()

    market_ids = set(examples["market_id"].astype(str))
    market_timeseries = canonical.probabilities.loc[
        canonical.probabilities["market_id"].astype(str).isin(market_ids),
        RepricingBenchmark.market_timeseries_columns(),
    ].copy()
    market_timeseries["market_id"] = market_timeseries["market_id"].astype(str)
    market_timeseries["timestamp_utc"] = pd.to_datetime(market_timeseries["timestamp_utc"], utc=True, errors="coerce")
    market_timeseries = market_timeseries.loc[
        market_timeseries["timestamp_utc"].notna(),
        RepricingBenchmark.market_timeseries_columns(),
    ].reset_index(drop=True)
    targets_frame = build_frame.loc[:, REPRICING_TARGET_COLUMNS].reset_index(drop=True).copy()

    return RepricingBenchmark(
        config=cfg,
        examples=examples,
        market_timeseries=market_timeseries,
        targets_frame=targets_frame,
        source=source,
    )


def build_repricing_analysis_frame(benchmark: RepricingBenchmark) -> pd.DataFrame:
    """Build a lightweight local audit frame for notebook analysis."""
    frame = benchmark.examples.copy()
    if frame.empty:
        return frame

    targets = benchmark.targets_frame.loc[
        :,
        ["market_id", "timestamp_utc", "future_horizon_hours", "label", "future_move", "split"],
    ].copy()
    if not targets.empty:
        frame = frame.merge(
            targets,
            on=["market_id", "timestamp_utc", "future_horizon_hours", "split"],
            how="left",
        )

    frame = attach_analysis_columns(
        frame,
        probability_col="current_yes_probability",
        example_id=format_repricing_example_ids(frame),
        row_id_col="repricing_row_id",
    )
    frame["admissible_repricing"] = 1
    frame["target"] = frame["label"].astype(int)
    return frame.sort_values(["timestamp_utc", "market_id"], kind="stable").reset_index(drop=True)
