from __future__ import annotations

import argparse
import logging
import sys

from collectors.orderbook_snapshot_collector import OrderBookSnapshotCollector
from storage.sqlite_orderbook_store import SqliteOrderBookStore


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll Polymarket order book snapshots for all open outcomes resolved from a URL."
    )
    parser.add_argument("--url", required=True, help="Polymarket /event/... or /market/... URL.")
    parser.add_argument("--db", required=True, help="SQLite database path.")
    parser.add_argument("--interval-seconds", type=float, default=10.0, help="Polling interval in seconds.")
    parser.add_argument("--levels", type=int, default=10, help="Order book depth to store per side (1..10).")
    parser.add_argument("--polls", type=int, default=None, help="Number of polling cycles to run. Omit to run forever.")
    parser.add_argument(
        "--include-closed",
        action="store_true",
        help="Also include closed/archived markets resolved from an event URL.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    collector = OrderBookSnapshotCollector(store=SqliteOrderBookStore())
    outcomes = collector.resolve_market_outcomes(args.url, include_closed=args.include_closed)
    print(f"Resolved {len(outcomes)} outcomes from URL")

    results = collector.run(
        url=args.url,
        db_path=args.db,
        interval_seconds=args.interval_seconds,
        levels=args.levels,
        max_polls=args.polls,
        include_closed=args.include_closed,
    )

    counts = SqliteOrderBookStore().get_counts(args.db)
    last_result = results[-1] if results else None
    print(f"Database: {args.db}")
    print(f"Counts: {counts}")
    if last_result is not None:
        print(
            "Last poll: "
            f"captured_at={last_result.captured_at_utc}, "
            f"ok={len(last_result.snapshots)}, "
            f"errors={len(last_result.errors)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
