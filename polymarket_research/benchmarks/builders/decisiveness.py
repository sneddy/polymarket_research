"""Build the decisiveness benchmark and local analysis frames from the canonical layer."""

from __future__ import annotations

import pandas as pd

from polymarket_research.benchmarks.builders.common import attach_analysis_columns
from polymarket_research.benchmarks.schemas.decisiveness import (
    DECISIVENESS_BUILD_COLUMNS,
    DECISIVENESS_TARGET_COLUMNS,
    DecisivenessBenchmark,
    DecisivenessBenchmarkConfig,
)
from polymarket_research.benchmarks.utils.ids import format_decisiveness_example_ids
from polymarket_research.benchmarks.utils.splits import assign_time_splits
from polymarket_research.data.canonical.dataset import CanonicalDataset


def build_decisiveness_from_canonical(
    canonical: CanonicalDataset,
    config: DecisivenessBenchmarkConfig | None = None,
    *,
    source: str = "polymarket",
) -> DecisivenessBenchmark:
    """Materialize a frozen decisiveness benchmark from a canonical dataset."""
    cfg = config or DecisivenessBenchmarkConfig()
    DecisivenessBenchmark._log(
        cfg,
        "starting "
        f"(markets={len(canonical.markets)}, probability_rows={len(canonical.probabilities)}, "
        f"tau={cfg.decisive_threshold:.2f}, split_on={cfg.split_on})",
    )
    DecisivenessBenchmark._validate_canonical(canonical)
    DecisivenessBenchmark._log(cfg, "validated canonical tables")
    markets = DecisivenessBenchmark._prepare_markets(canonical.markets)
    probabilities = DecisivenessBenchmark._prepare_probabilities(canonical.probabilities)
    DecisivenessBenchmark._log(cfg, f"prepared inputs (markets={len(markets)}, probability_rows={len(probabilities)})")

    grouped_probabilities = {
        market_id: frame.reset_index(drop=True)
        for market_id, frame in probabilities.groupby("market_id", sort=False)
    }
    DecisivenessBenchmark._log(cfg, f"indexed market histories (markets_with_history={len(grouped_probabilities)})")

    rows: list[dict[str, object]] = []
    market_iter = markets.itertuples(index=False)
    if cfg.show_progress:
        from tqdm.auto import tqdm

        market_iter = tqdm(market_iter, total=len(markets), desc="decisiveness examples", unit="market")
    for market in market_iter:
        market_id = str(market.market_id)
        market_history = grouped_probabilities.get(market_id)
        if market_history is None or market_history.empty:
            continue

        decisive_entry = DecisivenessBenchmark._durable_decisive_entry(market_history, threshold=cfg.decisive_threshold)
        if decisive_entry is None:
            continue
        decisive_idx, decisive_timestamp_utc, decisive_side = decisive_entry

        sampled_indices = DecisivenessBenchmark._sample_cutoff_indices(
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
        examples = pd.DataFrame(columns=DecisivenessBenchmark.example_columns())
        market_timeseries = pd.DataFrame(columns=DecisivenessBenchmark.market_timeseries_columns())
        targets_frame = pd.DataFrame(columns=DECISIVENESS_TARGET_COLUMNS)
        return DecisivenessBenchmark(
            config=cfg,
            examples=examples,
            market_timeseries=market_timeseries,
            targets_frame=targets_frame,
            source=source,
        )

    build_frame = pd.DataFrame(rows)
    split_source = (
        build_frame.loc[:, ["market_id", cfg.split_on]]
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
    build_frame = build_frame.merge(split_source.loc[:, ["market_id", "split"]], on="market_id", how="left")
    build_frame["_split_order"] = build_frame["split"].map({"train": 0, "test": 1}).fillna(2)
    build_frame = build_frame.sort_values(
        ["_split_order", "cutoff_timestamp_utc", "market_id"],
        kind="stable",
    ).reset_index(drop=True)
    build_frame = build_frame.loc[:, DECISIVENESS_BUILD_COLUMNS]
    examples = build_frame.loc[:, DecisivenessBenchmark.example_columns()].reset_index(drop=True).copy()
    DecisivenessBenchmark._log(
        cfg,
        "built example manifest "
        f"(examples={len(examples)}, train={int((examples['split'] == 'train').sum())}, "
        f"test={int((examples['split'] == 'test').sum())}, markets={examples['market_id'].nunique()})",
    )

    market_ids = set(examples["market_id"].astype(str))
    market_timeseries = probabilities.loc[
        probabilities["market_id"].astype(str).isin(market_ids),
        DecisivenessBenchmark.market_timeseries_columns(),
    ].reset_index(drop=True)
    targets_frame = build_frame.loc[:, DECISIVENESS_TARGET_COLUMNS].reset_index(drop=True).copy()
    DecisivenessBenchmark._log(cfg, f"built market histories (market_timeseries_rows={len(market_timeseries)})")
    DecisivenessBenchmark._log(cfg, "done")
    return DecisivenessBenchmark(
        config=cfg,
        examples=examples,
        market_timeseries=market_timeseries,
        targets_frame=targets_frame,
        source=source,
    )


def build_decisiveness_analysis_frame(benchmark: DecisivenessBenchmark) -> pd.DataFrame:
    """Build a lightweight local audit frame for notebook analysis."""
    frame = benchmark.examples.copy()
    if frame.empty:
        return frame

    target_columns = [
        "market_id",
        "cutoff_timestamp_utc",
        "decisive_timestamp_utc",
        "label",
        "label_name",
        "hours_to_decisive",
        "decisive_side",
        "split",
    ]
    targets = benchmark.targets_frame.loc[
        :,
        [column for column in target_columns if column in benchmark.targets_frame.columns],
    ].copy()
    merge_columns = ["market_id", "cutoff_timestamp_utc", "split"]
    if not targets.empty:
        frame = frame.merge(targets, on=merge_columns, how="left")

    frame = attach_analysis_columns(
        frame,
        probability_col="current_yes_probability",
        example_id=format_decisiveness_example_ids(frame),
        row_id_col="decisiveness_row_id",
    )
    frame["admissible_decisiveness"] = 1
    return frame.sort_values(["cutoff_timestamp_utc", "market_id"], kind="stable").reset_index(drop=True)
