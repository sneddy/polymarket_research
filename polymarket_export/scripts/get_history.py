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

from configs.resolved_dataset_domain_config import DOMAIN_PRIORITY
from clients.gamma_client import GammaClient
from polymarket_registry.history import build_pending_queue
from polymarket_registry.history import build_yes_probability_series_5m
from polymarket_registry.history import store_market_dataset
from polymarket_registry.schema import ensure_schema
from polymarket_registry.upsert import load_markets_for_category
from scripts.common import DEFAULT_DB_PATH
from scripts.common import DEFAULT_LOG_DIR
from scripts.common import setup_logging


logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download 5-minute probability histories for a prepared registry category."
    )
    p.add_argument(
        "--category",
        choices=DOMAIN_PRIORITY,
        default="geopolitics",
        help="Research domain to download from the prepared metadata registry.",
    )
    p.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="SQLite database path for markets / added_markets / probabilities.",
    )
    p.add_argument(
        "--max-markets",
        type=int,
        default=None,
        help="Optional cap on how many pending markets to process in this run.",
    )
    p.add_argument(
        "--trade-page-size",
        type=int,
        default=1000,
        help="Subgraph page size for trade download. Default: 1000.",
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
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_path = log_dir / f"get_history_{args.category}_{timestamp}.log"
    setup_logging(args.log_level, log_path=log_path)

    db_path = Path(args.db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("run logging initialized | log_path=%s", log_path)

    gamma = GammaClient()
    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        available_df = load_markets_for_category(conn, args.category)
        logger.info(
            "market registry loaded | category=%s available_markets=%s",
            args.category,
            len(available_df),
        )
        if available_df.empty:
            logger.warning(
                "no markets available for category=%s; run market_selection.py first",
                args.category,
            )
            return 0

        already_added = conn.execute(
            "SELECT COUNT(*) FROM added_markets WHERE primary_domain = ?",
            (args.category,),
        ).fetchone()[0]
        pending_df = build_pending_queue(conn, args.category)
        pending_count = len(pending_df)
        logger.info(
            "category download summary | category=%s total_markets=%s already_downloaded=%s pending_download=%s",
            args.category,
            len(available_df),
            already_added,
            pending_count,
        )
        queue_df = pending_df
        if args.max_markets is not None:
            queue_df = pending_df.head(int(args.max_markets)).reset_index(drop=True)

        completed = 0
        failed = 0
        queue_total = len(queue_df)
        from collectors.trades_collector import TradesCollector

        trades_collector = TradesCollector(gamma)
        for idx, (_, market_row) in enumerate(queue_df.iterrows(), start=1):
            market_id = str(market_row["market_id"])
            slug = str(market_row["market_slug"])
            logger.info(
                "dataset download started | category=%s market_index=%s/%s market_id=%s slug=%s",
                args.category,
                idx,
                queue_total,
                market_id,
                slug,
            )
            try:
                trades_df = trades_collector.download_all_trades(
                    str(market_row["condition_id"]),
                    limit=int(args.trade_page_size),
                    frame_type="pandas",
                    show_progress=True,
                    estimate_total=False,
                    progress_desc=f"Trades {idx}/{queue_total} {slug[:32]}",
                )
                trade_rows = int(len(trades_df))
                probability_df = build_yes_probability_series_5m(trades_df, market_id)
                store_market_dataset(
                    conn,
                    market_row=market_row,
                    probability_df=probability_df,
                    trade_rows=trade_rows,
                    storage_path=str(db_path),
                )
                completed += 1
                logger.info(
                    "dataset download finished | category=%s market_index=%s/%s market_id=%s slug=%s trade_rows=%s probability_rows=%s saved_to=%s",
                    args.category,
                    idx,
                    queue_total,
                    market_id,
                    slug,
                    trade_rows,
                    len(probability_df),
                    db_path,
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.exception(
                    "dataset download failed | category=%s market_index=%s/%s market_id=%s slug=%s error=%s",
                    args.category,
                    idx,
                    queue_total,
                    market_id,
                    slug,
                    exc,
                )
        logger.info(
            "run finished | category=%s completed_markets=%s failed_markets=%s db=%s",
            args.category,
            completed,
            failed,
            db_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
