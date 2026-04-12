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

from kalshi_registry.schema import ensure_schema
from scripts.common import DEFAULT_DB_PATH, DEFAULT_LOG_DIR, init_run_context


logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Placeholder for Kalshi candlesticks-first history download.")
    p.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    p.add_argument(
        "--save-trades",
        action="store_true",
        help="Reserved flag for optional raw-trades download. Default path remains candle-first without raw trades.",
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
    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        selected_count = conn.execute("SELECT COUNT(*) FROM selected_markets").fetchone()[0]
        enriched_count = conn.execute("SELECT COUNT(*) FROM market_universe").fetchone()[0]
        logger.info(
            "Kalshi get_history scaffold | selected_markets=%s market_universe=%s save_trades=%s",
            selected_count,
            enriched_count,
            args.save_trades,
        )
        logger.info(
            "Kalshi get_history is not implemented yet. Planned behavior: read `selected_markets`, fetch 1m candlesticks, write `probabilities`, and optionally write `raw_trades`."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
