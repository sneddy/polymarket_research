from __future__ import annotations

import argparse
import sys

from clients.gamma_client import GammaClient
from collectors.orderbook_recorder import OrderBookRecorder


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record Polymarket CLOB order book updates for a short duration.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--id", help="Token id (decimal) OR condition id (0x...).")
    src.add_argument("--url", help="Polymarket /event/... or /market/... URL.")
    p.add_argument("--seconds", type=int, default=60, help="Recording duration in seconds.")
    p.add_argument("--max-messages", type=int, default=None, help="Stop after N messages (optional).")
    p.add_argument("--out", default=None, help="Write messages to parquet at this path (optional).")
    p.add_argument("--snapshot", action="store_true", help="Fetch and print a REST snapshot before recording.")
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

    gamma = GammaClient()
    recorder = OrderBookRecorder(gamma_client=gamma)

    recorder.connect()
    if args.url is not None:
        token_ids = recorder.subscribe_url(args.url, market_index=args.market_index, market_slug=args.market_slug)
    else:
        token_ids = recorder.subscribe(args.id)
    print(f"Subscribed token ids: {token_ids}")

    if args.snapshot:
        snap = recorder.get_snapshot()
        print("Snapshot:")
        print(snap)

    df = recorder.record(args.seconds, max_messages=args.max_messages)
    print(f"Recorded {len(df)} messages")

    if args.out:
        recorder.save_to_parquet(df, args.out)
        print(f"Wrote: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
