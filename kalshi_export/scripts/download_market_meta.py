from __future__ import annotations

import argparse
from datetime import UTC, datetime
import logging
from pathlib import Path
import sqlite3
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clients.kalshi_client import KalshiClient
from kalshi_registry.refresh import refresh_raw_markets
from kalshi_registry.schema import ensure_schema
from scripts.common import DEFAULT_DB_PATH, DEFAULT_LOG_DIR, init_run_context


logger = logging.getLogger(__name__)


def _default_max_close_ts() -> int:
    """Default to the start of the current UTC day, excluding today's still-open horizon."""
    now = datetime.now(tz=UTC)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(today_start.timestamp())


def _parse_utc_date_start(value: str) -> int:
    """Parse `YYYY-MM-DD` into a UTC Unix timestamp at day start."""
    dt = datetime.strptime(value.strip(), "%Y-%m-%d").replace(tzinfo=UTC)
    return int(dt.timestamp())


def _resolve_close_date_range(args: argparse.Namespace) -> tuple[int | None, int]:
    if args.historical:
        if args.min_close_date is not None or args.max_close_date is not None:
            raise ValueError("`--min-close-date` and `--max-close-date` are only available for live `/markets` indexing.")
        return None, None
    min_close_ts = None if args.min_close_date is None else _parse_utc_date_start(args.min_close_date)
    max_close_ts = _default_max_close_ts() if args.max_close_date is None else _parse_utc_date_start(args.max_close_date)
    if min_close_ts is not None and min_close_ts > max_close_ts:
        raise ValueError("--min-close-date must be earlier than or equal to --max-close-date.")
    return min_close_ts, max_close_ts


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and upsert the Kalshi raw markets table from `/markets` or `/historical/markets`.")
    p.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="SQLite database path for the shared Kalshi market registry.",
    )
    p.add_argument(
        "--page-limit",
        type=int,
        default=200,
        help="Kalshi `/markets` page size.",
    )
    p.add_argument(
        "--max-index-pages",
        type=int,
        default=None,
        help="Optional cap on `/markets` pages to scan.",
    )
    p.add_argument(
        "--write-batch-pages",
        type=int,
        default=100,
        help="Number of fetched API pages to buffer before upserting into SQLite. Default: 100.",
    )
    p.add_argument(
        "--status",
        default=None,
        help="Optional `/markets` status filter. Default: unset.",
    )
    p.add_argument(
        "--series-ticker",
        default=None,
        help="Optional `/markets` series_ticker filter.",
    )
    p.add_argument(
        "--event-ticker",
        default=None,
        help="Optional `/markets` event_ticker filter.",
    )
    p.add_argument(
        "--min-close-date",
        default=None,
        help="Optional lower bound on market close date in `YYYY-MM-DD` (UTC day start).",
    )
    p.add_argument(
        "--max-close-date",
        default=None,
        help="Optional upper bound on market close date in `YYYY-MM-DD` (UTC day start). Defaults to today in UTC, which effectively means closed through yesterday.",
    )
    p.add_argument(
        "--historical",
        action="store_true",
        help="Index from `/historical/markets` instead of `/markets`. When enabled, live-only filters like `--status`, `--series-ticker`, and close-date filters are disabled.",
    )
    p.add_argument(
        "--force-remove",
        action="store_true",
        help="Clear `raw_markets` before indexing. By default the script preserves existing rows and upserts new ones.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    p.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help="Directory for per-run log files.",
    )
    return p.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    min_close_ts, max_close_ts = _resolve_close_date_range(args)
    db_path, log_path = init_run_context(
        log_level=args.log_level,
        log_dir=args.log_dir,
        log_stem="download_market_meta",
        db_path=args.db_path,
    )
    logger.info("run logging initialized | log_path=%s", log_path)

    kalshi = KalshiClient()
    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        refresh_stats = refresh_raw_markets(
            conn,
            kalshi=kalshi,
            historical=args.historical,
            limit=args.page_limit,
            max_pages=args.max_index_pages,
            force_remove=args.force_remove,
            write_batch_pages=args.write_batch_pages,
            status=args.status,
            series_ticker=args.series_ticker,
            event_ticker=args.event_ticker,
            min_close_ts=min_close_ts,
            max_close_ts=max_close_ts,
        )
        logger.info(
            "Kalshi raw markets download finished | source_endpoint=%s fetched_rows=%s total_raw_market_rows=%s db=%s",
            "historical/markets" if args.historical else "markets",
            refresh_stats["fetched_rows"],
            refresh_stats["table_rows"],
            db_path,
        )
        index_count = conn.execute("SELECT COUNT(*) FROM raw_markets").fetchone()[0]
        logger.info("Kalshi raw_markets table count | rows=%s", index_count)
        logger.info("Kalshi download_market_meta finished after populating raw_markets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
