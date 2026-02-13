from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from clients.gamma_client import GammaClient
from collectors.markets_collector import MarketsCollector


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Polymarket-wide market metadata and ranking stats.")
    p.add_argument("--active-only", action="store_true", help="Only include active markets.")
    p.add_argument("--limit", type=int, default=200, help="Gamma page size.")
    p.add_argument("--max-pages", type=int, default=None, help="Optional max pages per active state.")
    p.add_argument("--top", type=int, default=25, help="Number of top markets to return.")
    p.add_argument("--min-liquidity", type=float, default=None, help="Optional min liquidity filter for ranking.")
    p.add_argument("--min-volume-24h", type=float, default=None, help="Optional min 24h volume filter for ranking.")
    p.add_argument(
        "--min-created-at",
        default=None,
        help="Optional ISO datetime lower bound for market createdAt (example: 2025-01-01T00:00:00Z).",
    )
    p.add_argument("--out-markets", default=None, help="Optional parquet path for the full market universe.")
    p.add_argument("--out-top", default=None, help="Optional parquet path for ranked top markets.")
    p.add_argument("--out-summary", default=None, help="Optional JSON path for summary statistics.")
    p.add_argument("--no-progress", action="store_true", help="Disable metadata download progress bar.")
    p.add_argument("--no-estimate", action="store_true", help="Disable total-count estimation for progress bars.")
    p.add_argument("--frame-type", choices=["pandas", "polars"], default=None, help="Optional output frame type.")
    return p.parse_args(argv)


def _to_records(df: Any) -> list[dict[str, Any]]:
    try:
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            return df.to_dict(orient="records")
    except Exception:
        pass

    try:
        import polars as pl

        if isinstance(df, pl.DataFrame):
            return df.to_dicts()
    except Exception:
        pass

    return []


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    client = GammaClient()
    collector = MarketsCollector(client)
    report = collector.download_market_meta(
        include_active=True,
        include_inactive=not args.active_only,
        limit=args.limit,
        max_pages=args.max_pages,
        top_n=args.top,
        min_liquidity=args.min_liquidity,
        min_volume_24h=args.min_volume_24h,
        min_created_at=args.min_created_at,
        frame_type=args.frame_type,
        show_progress=not args.no_progress,
        estimate_total=not args.no_estimate,
    )

    markets = report["markets"]
    summary = report["summary"]
    top = report["top_markets"]

    print("Summary:")
    print(json.dumps(summary, indent=2, default=str))

    top_rows = _to_records(top)
    print(f"Top markets (n={len(top_rows)}):")
    for i, row in enumerate(top_rows[: min(10, len(top_rows))], start=1):
        slug = row.get("slug")
        score = row.get("market_score")
        vol24 = row.get("volume_24h")
        liq = row.get("liquidity")
        try:
            score_s = f"{float(score):.4f}"
        except Exception:
            score_s = "nan"
        try:
            vol24_s = f"{float(vol24):.2f}"
        except Exception:
            vol24_s = "nan"
        try:
            liq_s = f"{float(liq):.2f}"
        except Exception:
            liq_s = "nan"
        print(f"{i:2d}. score={score_s} volume_24h={vol24_s} liquidity={liq_s} slug={slug}")

    if args.out_markets:
        collector.save_to_parquet(markets, args.out_markets)
        print(f"Wrote markets: {args.out_markets}")
    if args.out_top:
        collector.save_to_parquet(top, args.out_top)
        print(f"Wrote top markets: {args.out_top}")
    if args.out_summary:
        out = Path(args.out_summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"Wrote summary: {args.out_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
