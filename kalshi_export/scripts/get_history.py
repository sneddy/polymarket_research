from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clients.kalshi_client import KalshiClient
from collectors.markets_collector import _resolve_tqdm
from kalshi_registry.history import (
    build_pending_queue,
    build_probability_series_5m_from_candles,
    download_market_minute_candles,
    get_market_settled_cutoff,
    store_market_history,
)
from kalshi_registry.schema import ensure_schema
from scripts.common import DEFAULT_DB_PATH, DEFAULT_LOG_DIR, init_run_context


logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Kalshi minute candles and build 5-minute probability panels.")
    p.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    p.add_argument("--max-markets", type=int, default=None, help="Optional cap on the number of markets processed in this run.")
    p.add_argument(
        "--chunk-days",
        type=int,
        default=7,
        help="Candlestick request window per API call in days. Default: 7.",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="Redownload history even for markets already present in `added_markets`.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    p.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory for per-run log files.")
    return p.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    db_path, log_path = init_run_context(
        log_level=args.log_level,
        log_dir=args.log_dir,
        log_stem="get_history",
        db_path=args.db_path,
    )
    logger.info("run logging initialized | log_path=%s", log_path)

    kalshi = KalshiClient()
    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        pending_df = build_pending_queue(conn, force_refresh=args.force_refresh)
        if args.max_markets is not None:
            pending_df = pending_df.head(int(args.max_markets)).reset_index(drop=True)
        queue_total = len(pending_df)
        logger.info(
            "Kalshi get_history started | pending_markets=%s force_refresh=%s chunk_days=%s",
            queue_total,
            args.force_refresh,
            args.chunk_days,
        )
        if pending_df.empty:
            logger.info("Kalshi get_history finished | pending_markets=0 db=%s", db_path)
            return 0

        cutoff_ts = get_market_settled_cutoff(kalshi)
        cutoff_text = cutoff_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info("Kalshi market history cutoff loaded | market_settled_ts=%s", cutoff_text)

        tqdm = _resolve_tqdm(show_progress=True)
        pbar = (
            tqdm(total=queue_total, disable=False, unit="market", desc="Downloading Kalshi market history", leave=True)
            if tqdm is not None
            else None
        )

        completed = 0
        failed = 0
        written_prob_rows = 0
        try:
            for idx, (_, market_row) in enumerate(pending_df.iterrows(), start=1):
                market_id = str(market_row["market_id"])
                ticker = str(market_row["ticker"])
                logger.info(
                    "Kalshi history download started | market_index=%s/%s market_id=%s ticker=%s",
                    idx,
                    queue_total,
                    market_id,
                    ticker,
                )
                try:
                    minute_df, history_source_mode, warnings = download_market_minute_candles(
                        kalshi,
                        market_row=market_row,
                        cutoff_ts=cutoff_ts,
                        chunk_days=args.chunk_days,
                        show_progress=True,
                        progress_desc=f"Candles {idx}/{queue_total} {ticker[:32]}",
                    )
                    probability_df = build_probability_series_5m_from_candles(minute_df, market_id)
                    stats = store_market_history(
                        conn,
                        market_row=market_row,
                        minute_df=minute_df,
                        probability_df=probability_df,
                        storage_path=str(db_path),
                        history_source_mode=history_source_mode,
                        cutoff_ts_used=cutoff_text,
                        warnings=warnings,
                    )
                    completed += 1
                    written_prob_rows += int(stats["probability_rows"])
                    logger.info(
                        "Kalshi history download finished | market_index=%s/%s market_id=%s ticker=%s history_source_mode=%s minute_rows=%s probability_rows=%s warnings=%s",
                        idx,
                        queue_total,
                        market_id,
                        ticker,
                        history_source_mode,
                        stats["minute_rows"],
                        stats["probability_rows"],
                        len(warnings),
                    )
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    logger.exception(
                        "Kalshi history download failed | market_index=%s/%s market_id=%s ticker=%s error=%s",
                        idx,
                        queue_total,
                        market_id,
                        ticker,
                        exc,
                    )
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "completed": completed,
                            "failed": failed,
                            "prob_rows": written_prob_rows,
                        }
                    )
        finally:
            if pbar is not None:
                pbar.close()

        logger.info(
            "Kalshi get_history finished | completed_markets=%s failed_markets=%s probability_rows_written=%s db=%s",
            completed,
            failed,
            written_prob_rows,
            db_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
