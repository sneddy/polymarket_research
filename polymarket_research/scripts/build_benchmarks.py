"""CLI for materializing frozen benchmark releases from a canonical dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from polymarket_research.benchmarks import (
    DecisivenessBenchmarkConfig,
    RepricingBenchmarkConfig,
    TerminalBenchmarkConfig,
)
from polymarket_research.benchmarks.builders import (
    build_decisiveness_from_canonical,
    build_repricing_from_canonical,
    build_terminal_from_canonical,
)
from polymarket_research.benchmarks.io.paths import (
    DEFAULT_BENCHMARK_RELEASE_VERSION,
    benchmark_release_dir,
)
from polymarket_research.data.canonical import CanonicalDataset
from polymarket_research.scripts.common import (
    VALID_BENCHMARK_TASKS,
    VALID_SOURCES,
    default_canonical_cache_dir,
    log_message,
    log_stage,
    normalize_tasks,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
    parse_int_float_map,
    parse_optional_timestamp,
    print_frame,
    resolve_repo_root,
)


DEFAULT_TERMINAL_HORIZONS = (24, 168, 336)
DEFAULT_TERMINAL_STALENESS_BY_HORIZON = {24: 12.0, 168: 24.0, 336: 48.0}
DEFAULT_DECISIVENESS_BINS = (24.0, 72.0)
DEFAULT_DECISIVENESS_LABELS = ("short", "medium", "long")


@dataclass(frozen=True)
class BenchmarkBuildResult:
    """Structured result returned by the benchmark build script helpers."""

    outputs: dict[str, Path]
    report_paths: dict[str, Path] | None = None


def make_terminal_config(
    *,
    show_progress: bool,
    horizons_hours: tuple[int, ...] = DEFAULT_TERMINAL_HORIZONS,
    max_snapshot_staleness_hours: float = 12.0,
    max_snapshot_staleness_hours_by_horizon: dict[int, float] | None = DEFAULT_TERMINAL_STALENESS_BY_HORIZON,
    split_on: str = "end_date",
    split_timestamp_utc=None,
    train_fraction: float = 0.8,
) -> TerminalBenchmarkConfig:
    """Build the default terminal release config with optional overrides."""
    return TerminalBenchmarkConfig(
        horizons_hours=horizons_hours,
        max_snapshot_staleness_hours=max_snapshot_staleness_hours,
        max_snapshot_staleness_hours_by_horizon=max_snapshot_staleness_hours_by_horizon,
        split_on=split_on,
        split_timestamp_utc=split_timestamp_utc,
        train_fraction=train_fraction,
        show_progress=show_progress,
    )


def make_decisiveness_config(
    *,
    show_progress: bool,
    decisive_threshold: float = 0.95,
    sample_every_hours: int = 12,
    min_history_points: int = 24,
    min_prefix_age_hours: float = 6.0,
    min_time_to_decisive_hours: float = 1.0,
    ordinal_bin_edges_hours: tuple[float, ...] = DEFAULT_DECISIVENESS_BINS,
    ordinal_bin_labels: tuple[str, ...] = DEFAULT_DECISIVENESS_LABELS,
    split_on: str = "decisive_timestamp_utc",
    split_timestamp_utc=None,
    train_fraction: float = 0.8,
) -> DecisivenessBenchmarkConfig:
    """Build the default decisiveness release config with optional overrides."""
    return DecisivenessBenchmarkConfig(
        decisive_threshold=decisive_threshold,
        sample_every_hours=sample_every_hours,
        min_history_points=min_history_points,
        min_prefix_age_hours=min_prefix_age_hours,
        min_time_to_decisive_hours=min_time_to_decisive_hours,
        ordinal_bin_edges_hours=ordinal_bin_edges_hours,
        ordinal_bin_labels=ordinal_bin_labels,
        split_on=split_on,
        split_timestamp_utc=split_timestamp_utc,
        train_fraction=train_fraction,
        show_progress=show_progress,
    )


def make_repricing_config(
    *,
    show_progress: bool,
    future_horizon_hours: int = 24,
    lookback_hours: int = 24,
    sample_every_hours: int = 12,
    move_threshold: float = 0.15,
    attach_external_shocks: bool = True,
    split_on: str = "timestamp_utc",
    split_timestamp_utc=None,
    train_fraction: float = 0.8,
) -> RepricingBenchmarkConfig:
    """Build the default repricing release config with optional overrides."""
    return RepricingBenchmarkConfig(
        future_horizon_hours=future_horizon_hours,
        lookback_hours=lookback_hours,
        sample_every_hours=sample_every_hours,
        move_threshold=move_threshold,
        attach_external_shocks=attach_external_shocks,
        split_on=split_on,
        split_timestamp_utc=split_timestamp_utc,
        train_fraction=train_fraction,
        show_progress=show_progress,
    )


def build_benchmark_releases(
    *,
    canonical: CanonicalDataset | None = None,
    canonical_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    source: str = "polymarket",
    db_path: str | Path | None = None,
    tasks: tuple[str, ...] = VALID_BENCHMARK_TASKS,
    version: str = DEFAULT_BENCHMARK_RELEASE_VERSION,
    terminal_config: TerminalBenchmarkConfig | None = None,
    decisiveness_config: DecisivenessBenchmarkConfig | None = None,
    repricing_config: RepricingBenchmarkConfig | None = None,
) -> BenchmarkBuildResult:
    """Build one or more benchmark releases from a canonical dataset."""
    prefix = "[build_benchmarks]"
    root = resolve_repo_root(repo_root)
    normalized_source = str(source)
    benchmark_tasks = normalize_tasks(tasks)
    log_message(
        prefix,
        f"config: source={normalized_source} tasks={list(benchmark_tasks)} version={version}",
    )
    canonical_dataset = canonical
    if canonical_dataset is None:
        load_dir = Path(canonical_dir) if canonical_dir is not None else default_canonical_cache_dir(root, normalized_source)
        log_message(prefix, f"canonical_dir={load_dir}")
        with log_stage(prefix, "load canonical dataset"):
            canonical_dataset = CanonicalDataset.from_parquet(load_dir)
    else:
        log_message(prefix, "using in-memory canonical dataset")
    print_frame(f"{prefix} canonical summary", canonical_dataset.summary())

    outputs: dict[str, Path] = {}
    built_benchmarks: dict[str, object] = {}
    if "terminal" in benchmark_tasks:
        config = terminal_config or make_terminal_config(show_progress=True)
        out_dir = benchmark_release_dir(root, source=normalized_source, task="terminal", version=version)
        with log_stage(prefix, f"build terminal benchmark -> {out_dir}"):
            built = build_terminal_from_canonical(
                canonical_dataset,
                config=config,
                source=normalized_source,
            )
            built.save(out_dir)
        outputs["terminal"] = out_dir
        built_benchmarks["terminal"] = built
        log_message(prefix, f"terminal rows={len(built.examples)} market_timeseries_rows={len(built.market_timeseries)}")

    if "decisiveness" in benchmark_tasks:
        config = decisiveness_config or make_decisiveness_config(show_progress=True)
        out_dir = benchmark_release_dir(root, source=normalized_source, task="decisiveness", version=version)
        with log_stage(prefix, f"build decisiveness benchmark -> {out_dir}"):
            built = build_decisiveness_from_canonical(
                canonical_dataset,
                config=config,
                source=normalized_source,
            )
            built.save(out_dir)
        outputs["decisiveness"] = out_dir
        built_benchmarks["decisiveness"] = built
        log_message(prefix, f"decisiveness rows={len(built.examples)} market_timeseries_rows={len(built.market_timeseries)}")

    if "repricing" in benchmark_tasks:
        config = repricing_config or make_repricing_config(show_progress=True)
        out_dir = benchmark_release_dir(root, source=normalized_source, task="repricing", version=version)
        with log_stage(prefix, f"build repricing benchmark -> {out_dir}"):
            built = build_repricing_from_canonical(
                canonical_dataset,
                config=config,
                source=normalized_source,
            )
            built.save(out_dir)
        outputs["repricing"] = out_dir
        built_benchmarks["repricing"] = built
        log_message(prefix, f"repricing rows={len(built.examples)} market_timeseries_rows={len(built.market_timeseries)}")

    from polymarket_research.scripts.release_report import write_release_report

    with log_stage(prefix, "write source-level release report"):
        report_paths = write_release_report(
            repo_root=root,
            source=normalized_source,
            version=version,
            canonical=canonical_dataset,
            benchmarks=built_benchmarks,
            db_path=db_path,
        )
    log_message(prefix, f"release_report={report_paths['json']}")
    return BenchmarkBuildResult(outputs=outputs, report_paths=report_paths)


def build_parser(*, include_help: bool = True) -> argparse.ArgumentParser:
    """Create the CLI parser for benchmark materialization."""
    parser = argparse.ArgumentParser(description=__doc__, add_help=include_help)
    parser.add_argument("--repo-root", type=Path, default=None, help="Repository root. Defaults to discovery from cwd.")
    parser.add_argument("--source", choices=VALID_SOURCES, default="polymarket")
    parser.add_argument("--db-path", type=Path, default=None, help="Optional SQLite path used to enrich the source-level release report.")
    parser.add_argument("--canonical-dir", type=Path, default=None, help="Canonical input directory. Defaults to the internal canonical cache path.")
    parser.add_argument("--tasks", default="all", help="Comma-separated benchmark tasks. One of: terminal, decisiveness, repricing, all.")
    parser.add_argument("--version", default=DEFAULT_BENCHMARK_RELEASE_VERSION)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--terminal-horizons-hours", default="24,168,336")
    parser.add_argument("--terminal-max-snapshot-staleness-hours", type=float, default=12.0)
    parser.add_argument("--terminal-max-snapshot-staleness-hours-by-horizon", default="24=12,168=24,336=48")
    parser.add_argument("--terminal-split-on", default="end_date")
    parser.add_argument("--terminal-split-timestamp-utc", default=None)
    parser.add_argument("--terminal-train-fraction", type=float, default=0.8)

    parser.add_argument("--decisive-threshold", type=float, default=0.95)
    parser.add_argument("--decisiveness-sample-every-hours", type=int, default=12)
    parser.add_argument("--decisiveness-min-history-points", type=int, default=24)
    parser.add_argument("--decisiveness-min-prefix-age-hours", type=float, default=6.0)
    parser.add_argument("--decisiveness-min-time-to-decisive-hours", type=float, default=1.0)
    parser.add_argument("--decisiveness-ordinal-bin-edges-hours", default="24,72")
    parser.add_argument("--decisiveness-ordinal-bin-labels", default="short,medium,long")
    parser.add_argument("--decisiveness-split-on", default="decisive_timestamp_utc")
    parser.add_argument("--decisiveness-split-timestamp-utc", default=None)
    parser.add_argument("--decisiveness-train-fraction", type=float, default=0.8)

    parser.add_argument("--repricing-future-horizon-hours", type=int, default=24)
    parser.add_argument("--repricing-lookback-hours", type=int, default=24)
    parser.add_argument("--repricing-sample-every-hours", type=int, default=12)
    parser.add_argument("--repricing-move-threshold", type=float, default=0.15)
    parser.add_argument("--repricing-attach-external-shocks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repricing-split-on", default="timestamp_utc")
    parser.add_argument("--repricing-split-timestamp-utc", default=None)
    parser.add_argument("--repricing-train-fraction", type=float, default=0.8)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark build CLI."""
    args = build_parser().parse_args(argv)
    tasks = normalize_tasks(parse_csv_strings(args.tasks))
    terminal_config = make_terminal_config(
        show_progress=args.show_progress,
        horizons_hours=parse_csv_ints(args.terminal_horizons_hours),
        max_snapshot_staleness_hours=args.terminal_max_snapshot_staleness_hours,
        max_snapshot_staleness_hours_by_horizon=parse_int_float_map(args.terminal_max_snapshot_staleness_hours_by_horizon),
        split_on=args.terminal_split_on,
        split_timestamp_utc=parse_optional_timestamp(args.terminal_split_timestamp_utc),
        train_fraction=args.terminal_train_fraction,
    )
    decisiveness_config = make_decisiveness_config(
        show_progress=args.show_progress,
        decisive_threshold=args.decisive_threshold,
        sample_every_hours=args.decisiveness_sample_every_hours,
        min_history_points=args.decisiveness_min_history_points,
        min_prefix_age_hours=args.decisiveness_min_prefix_age_hours,
        min_time_to_decisive_hours=args.decisiveness_min_time_to_decisive_hours,
        ordinal_bin_edges_hours=parse_csv_floats(args.decisiveness_ordinal_bin_edges_hours),
        ordinal_bin_labels=parse_csv_strings(args.decisiveness_ordinal_bin_labels),
        split_on=args.decisiveness_split_on,
        split_timestamp_utc=parse_optional_timestamp(args.decisiveness_split_timestamp_utc),
        train_fraction=args.decisiveness_train_fraction,
    )
    repricing_config = make_repricing_config(
        show_progress=args.show_progress,
        future_horizon_hours=args.repricing_future_horizon_hours,
        lookback_hours=args.repricing_lookback_hours,
        sample_every_hours=args.repricing_sample_every_hours,
        move_threshold=args.repricing_move_threshold,
        attach_external_shocks=args.repricing_attach_external_shocks,
        split_on=args.repricing_split_on,
        split_timestamp_utc=parse_optional_timestamp(args.repricing_split_timestamp_utc),
        train_fraction=args.repricing_train_fraction,
    )
    result = build_benchmark_releases(
        repo_root=args.repo_root,
        source=args.source,
        db_path=args.db_path,
        canonical_dir=args.canonical_dir,
        tasks=tasks,
        version=args.version,
        terminal_config=terminal_config,
        decisiveness_config=decisiveness_config,
        repricing_config=repricing_config,
    )
    print("[build_benchmarks] wrote benchmark releases")
    print_frame(
        "[build_benchmarks] outputs",
        pd.DataFrame(
            [{"task": task, "output_dir": str(path)} for task, path in result.outputs.items()]
        ),
    )
    if result.report_paths:
        print_frame(
            "[build_benchmarks] release report",
            pd.DataFrame(
                [{"name": name, "path": str(path)} for name, path in result.report_paths.items()]
            ),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
