"""Build the terminal benchmark from the canonical layer."""

from __future__ import annotations

from polymarket_research.benchmarks.schemas.terminal import (
    TERMINAL_TARGET_COLUMNS,
    TerminalBenchmark,
    TerminalBenchmarkConfig,
)
from polymarket_research.data.canonical.dataset import CanonicalDataset


def build_terminal_from_canonical(
    canonical: CanonicalDataset,
    config: TerminalBenchmarkConfig | None = None,
    *,
    source: str = "polymarket",
) -> TerminalBenchmark:
    """Materialize a frozen terminal benchmark from a canonical dataset."""
    cfg = config or TerminalBenchmarkConfig()
    TerminalBenchmark._log(
        cfg,
        "starting "
        f"(markets={len(canonical.markets)}, probability_rows={len(canonical.probabilities)}, "
        f"horizons={list(cfg.horizons_hours)}, split_on={cfg.split_on})",
    )
    TerminalBenchmark._validate_canonical(canonical)
    TerminalBenchmark._log(cfg, "validated canonical tables")
    markets = TerminalBenchmark._prepare_markets(canonical.markets)
    probabilities = TerminalBenchmark._prepare_probabilities(canonical.probabilities)
    TerminalBenchmark._log(cfg, f"prepared inputs (markets={len(markets)}, probability_rows={len(probabilities)})")
    market_slices = TerminalBenchmark._build_market_slices(probabilities["market_id"])
    TerminalBenchmark._log(cfg, f"indexed contiguous market blocks (markets_with_history={len(market_slices)})")
    build_frame = TerminalBenchmark._build_examples(markets, probabilities, market_slices, cfg)
    examples = build_frame.loc[:, TerminalBenchmark.example_columns()].reset_index(drop=True).copy()
    TerminalBenchmark._log(
        cfg,
        "built example manifest "
        f"(examples={len(build_frame)}, train={int((build_frame['split'] == 'train').sum()) if not build_frame.empty else 0}, "
        f"test={int((build_frame['split'] == 'test').sum()) if not build_frame.empty else 0})",
    )
    market_timeseries = TerminalBenchmark._build_market_timeseries(probabilities, examples, cfg)
    targets_frame = build_frame.loc[:, TERMINAL_TARGET_COLUMNS].reset_index(drop=True).copy()
    TerminalBenchmark._log(cfg, f"built market histories (market_timeseries_rows={len(market_timeseries)})")
    TerminalBenchmark._log(cfg, "done")
    return TerminalBenchmark(
        config=cfg,
        examples=examples,
        market_timeseries=market_timeseries,
        targets_frame=targets_frame,
        source=source,
    )
