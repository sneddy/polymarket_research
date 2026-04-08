from __future__ import annotations

import argparse
import sys

from collectors.external_covariates_collector import ExternalCovariatesCollector
from configs.external_covariates_config import DEFAULT_EXTERNAL_SERIES


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download external market covariates into a normalized parquet dataset.")
    p.add_argument("--start-date", required=True, help="ISO start date.")
    p.add_argument("--end-date", required=True, help="ISO end date.")
    p.add_argument(
        "--series-id",
        action="append",
        dest="series_ids",
        default=None,
        help="Series id to download. Repeatable. Defaults to the configured default registry.",
    )
    p.add_argument("--out", required=True, help="Parquet path or partitioned dataset directory.")
    p.add_argument("--frame-type", choices=["pandas", "polars"], default=None)
    p.add_argument(
        "--binance-source",
        choices=["archive", "api"],
        default="archive",
        help="How to fetch Binance-backed crypto history. Default: archive.",
    )
    p.add_argument(
        "--archive-tail-days",
        type=int,
        default=45,
        help="When using Binance archives, also fetch daily files for the last N days to cover the freshest tail.",
    )
    p.add_argument(
        "--sec-user-agent-note",
        action="store_true",
        help="Print a reminder that sec_edgar series use the configured User-Agent from config.py.",
    )
    p.add_argument(
        "--partition-cols",
        nargs="*",
        default=["series_id"],
        help="Optional parquet partition columns. Default: series_id.",
    )
    p.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    selected = args.series_ids or list(DEFAULT_EXTERNAL_SERIES)
    if args.sec_user_agent_note or any(str(series_id).startswith("edgar_") for series_id in selected):
        print("Note: sec_edgar series use SecEdgarConfig.user_agent from config.py; set it to descriptive contact info for SEC access.")
    collector = ExternalCovariatesCollector()
    df = collector.download_many(
        selected,
        start_date=args.start_date,
        end_date=args.end_date,
        frame_type=args.frame_type,
        binance_source=args.binance_source,
        archive_tail_days=args.archive_tail_days,
        show_progress=not args.no_progress,
    )
    out = collector.save_to_parquet(df, args.out, partition_cols=args.partition_cols)
    print(f"Wrote {len(df)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
