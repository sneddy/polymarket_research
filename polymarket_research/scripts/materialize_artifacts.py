"""CLI for building the canonical dataset plus benchmark releases in one run."""

from __future__ import annotations

import argparse
from pathlib import Path

from polymarket_research.scripts.build_benchmarks import (
    build_benchmark_releases,
    build_parser as build_benchmarks_parser,
    make_decisiveness_config,
    make_repricing_config,
    make_terminal_config,
)
from polymarket_research.scripts.build_canonical import (
    build_canonical_dataset,
    build_parser as build_canonical_parser,
)
from polymarket_research.scripts.common import (
    log_message,
    log_stage,
    normalize_tasks,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
    parse_int_float_map,
    parse_optional_timestamp,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the orchestration parser by combining canonical and benchmark args."""
    canonical_parent = build_canonical_parser(include_help=False)
    benchmark_parent = build_benchmarks_parser(include_help=False)
    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[canonical_parent, benchmark_parent],
        conflict_handler="resolve",
    )
    for action in list(parser._actions):
        if action.dest != "canonical_dir":
            continue
        parser._actions.remove(action)
        for option_string in action.option_strings:
            parser._option_string_actions.pop(option_string, None)
    for group in parser._action_groups:
        group._group_actions = [action for action in group._group_actions if action.dest != "canonical_dir"]
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the full internal artifact materialization pipeline."""
    prefix = "[materialize_artifacts]"
    args = build_parser().parse_args(argv)
    tasks = normalize_tasks(parse_csv_strings(args.tasks))
    log_message(
        prefix,
        f"start: source={args.source} tasks={list(tasks)} version={args.version}",
    )
    with log_stage(prefix, "materialize canonical dataset"):
        canonical_result = build_canonical_dataset(
            repo_root=args.repo_root,
            source=args.source,
            db_path=args.db_path,
            output_dir=args.output_dir if hasattr(args, "output_dir") else None,
            market_limit=args.market_limit,
            market_order=args.market_order,
            include_raw_trades=args.include_raw_trades,
            include_external_covariates=args.include_external_covariates,
            external_covariates_path=args.external_covariates_path,
            resolved_only=args.resolved_only,
            save_raw_snapshot=args.save_raw_snapshot,
            raw_snapshot_dir=args.raw_snapshot_dir,
            show_progress=args.show_progress,
        )
    with log_stage(prefix, "materialize benchmark releases"):
        benchmark_result = build_benchmark_releases(
            canonical=canonical_result.canonical,
            repo_root=args.repo_root,
            source=args.source,
            db_path=canonical_result.db_path,
            tasks=tasks,
            version=args.version,
            terminal_config=make_terminal_config(
                show_progress=args.show_progress,
                horizons_hours=parse_csv_ints(args.terminal_horizons_hours),
                max_snapshot_staleness_hours=args.terminal_max_snapshot_staleness_hours,
                max_snapshot_staleness_hours_by_horizon=parse_int_float_map(args.terminal_max_snapshot_staleness_hours_by_horizon),
                split_on=args.terminal_split_on,
                split_timestamp_utc=parse_optional_timestamp(args.terminal_split_timestamp_utc),
                train_fraction=args.terminal_train_fraction,
            ),
            decisiveness_config=make_decisiveness_config(
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
            ),
            repricing_config=make_repricing_config(
                show_progress=args.show_progress,
                future_horizon_hours=args.repricing_future_horizon_hours,
                lookback_hours=args.repricing_lookback_hours,
                sample_every_hours=args.repricing_sample_every_hours,
                move_threshold=args.repricing_move_threshold,
                attach_external_shocks=args.repricing_attach_external_shocks,
                split_on=args.repricing_split_on,
                split_timestamp_utc=parse_optional_timestamp(args.repricing_split_timestamp_utc),
                train_fraction=args.repricing_train_fraction,
            ),
        )
    print(f"{prefix} canonical_dir={canonical_result.canonical_dir}")
    for task, output_dir in benchmark_result.outputs.items():
        print(f"{prefix} {task}_dir={output_dir}")
    if benchmark_result.report_paths:
        for name, path in benchmark_result.report_paths.items():
            print(f"{prefix} release_report_{name}={path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
