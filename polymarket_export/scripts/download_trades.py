from __future__ import annotations

import argparse
import sys

from clients.gamma_client import GammaClient
from collectors.trades_collector import TradesCollector


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Polymarket historical trades for a condition id.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--market-id", help="Condition id (0x...).")
    src.add_argument("--url", help="Polymarket /event/... or /market/... URL.")
    p.add_argument("--start-date", default=None, help="Optional ISO start date (UTC assumed if no TZ).")
    p.add_argument("--out", required=True, help="Parquet path to write trades.")
    p.add_argument("--limit", type=int, default=500, help="Page size (subgraph GraphQL page size).")
    p.add_argument("--max-pages", type=int, default=None, help="Optional max pages to fetch.")
    p.add_argument(
        "--market-index",
        type=int,
        default=None,
        help="If --url is an /event/ URL with multiple markets, pick by 0-based index.",
    )
    p.add_argument(
        "--market-slug",
        default=None,
        help="If --url is an /event/ URL with multiple markets, pick by market slug.",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    client = GammaClient()
    collector = TradesCollector(client)

    if args.url is not None:
        df = collector.download_all_trades_from_url(
            args.url,
            start_date=args.start_date,
            limit=args.limit,
            max_pages=args.max_pages,
            market_index=args.market_index,
            market_slug=args.market_slug,
        )
    else:
        df = collector.download_all_trades(
            args.market_id,
            start_date=args.start_date,
            limit=args.limit,
            max_pages=args.max_pages,
        )

    collector.save_to_parquet(df, args.out)
    print(f"Wrote {len(df)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
