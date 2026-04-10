from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
import logging
from pathlib import Path
import sqlite3
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clients.gamma_client import GammaClient
from polymarket_registry.history import build_pending_queue_all
from polymarket_registry.history import build_yes_probability_series_5m
from polymarket_registry.history import store_market_dataset
from polymarket_registry.schema import ensure_schema
from polymarket_registry.upsert import load_all_markets
from scripts.common import DEFAULT_DB_PATH
from scripts.common import DEFAULT_LOG_DIR
from scripts.common import setup_logging


logger = logging.getLogger(__name__)


def _parse_list_text(value: object) -> list[str]:
    """Parse a list-like text payload into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip().strip("\"'") for part in text.split(",") if part.strip().strip("\"'")]


def _resolve_local_trade_metadata(market_row: object) -> tuple[list[str], dict[str, str]]:
    """Extract local token ids and token outcomes from a selected market row."""
    raw_get = market_row.get if hasattr(market_row, "get") else lambda *_args, **_kwargs: None
    token_ids = _parse_list_text(raw_get("clob_token_ids"))
    outcomes = _parse_list_text(raw_get("outcomes"))
    token_outcomes: dict[str, str] = {}
    if token_ids and outcomes and len(token_ids) == len(outcomes):
        token_outcomes = {token_id: outcome for token_id, outcome in zip(token_ids, outcomes, strict=False)}
    return token_ids, token_outcomes


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download 5-minute probability histories for every market currently stored by market_selection."
    )
    p.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="SQLite database path for selected_markets / added_markets / probabilities / raw_trades.",
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
    p.add_argument(
        "--save_trades",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store normalized raw fill-level trades in SQLite alongside the 5-minute probability panel.",
    )
    return p.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_path = log_dir / f"get_history_{timestamp}.log"
    setup_logging(args.log_level, log_path=log_path)

    db_path = Path(args.db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("run logging initialized | log_path=%s", log_path)

    gamma = GammaClient()
    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        available_df = load_all_markets(conn)
        logger.info(
            "market registry loaded | available_markets=%s",
            len(available_df),
        )
        if available_df.empty:
            logger.warning("no markets available; run market_selection.py first")
            return 0

        already_added = conn.execute(
            """
            SELECT COUNT(*)
            FROM added_markets
            WHERE ? = 0 OR COALESCE(raw_trades_saved, 0) = 1
            """,
            (1 if args.save_trades else 0,),
        ).fetchone()[0]
        pending_df = build_pending_queue_all(conn, require_raw_trades=bool(args.save_trades))
        pending_count = len(pending_df)
        logger.info(
            "history download summary | total_markets=%s already_downloaded=%s pending_download=%s",
            len(available_df),
            already_added,
            pending_count,
        )
        logger.info("history download config | save_trades=%s", args.save_trades)
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
                "dataset download started | market_index=%s/%s market_id=%s slug=%s",
                idx,
                queue_total,
                market_id,
                slug,
            )
            try:
                token_ids, token_outcomes = _resolve_local_trade_metadata(market_row)
                logger.info(
                    "dataset trade metadata | market_index=%s/%s market_id=%s local_token_ids=%s",
                    idx,
                    queue_total,
                    market_id,
                    len(token_ids),
                )
                trades_df = trades_collector.download_all_trades(
                    str(market_row["condition_id"]),
                    limit=int(args.trade_page_size),
                    frame_type="pandas",
                    show_progress=True,
                    estimate_total=False,
                    progress_desc=f"Trades {idx}/{queue_total} {slug[:32]}",
                    token_ids=token_ids or None,
                    token_outcomes=token_outcomes or None,
                )
                trade_rows = int(len(trades_df))
                probability_df = build_yes_probability_series_5m(trades_df, market_id)
                store_market_dataset(
                    conn,
                    market_row=market_row,
                    probability_df=probability_df,
                    trade_rows=trade_rows,
                    storage_path=str(db_path),
                    trades_df=trades_df,
                    save_trades=bool(args.save_trades),
                )
                completed += 1
                logger.info(
                    "dataset download finished | market_index=%s/%s market_id=%s slug=%s trade_rows=%s probability_rows=%s saved_to=%s",
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
                    "dataset download failed | market_index=%s/%s market_id=%s slug=%s error=%s",
                    idx,
                    queue_total,
                    market_id,
                    slug,
                    exc,
                )
        logger.info(
            "run finished | completed_markets=%s failed_markets=%s db=%s",
            completed,
            failed,
            db_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
