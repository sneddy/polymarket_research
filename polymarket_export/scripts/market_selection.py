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

from clients.gamma_client import GammaClient
from configs.resolved_dataset_domain_config import DOMAIN_PRIORITY
from polymarket_registry.block_filters import DEFAULT_MIN_RESIDUAL_VOLUME
from polymarket_registry.refresh import select_market_registry_from_universe_all_categories
from polymarket_registry.schema import ensure_schema
from scripts.common import DEFAULT_DB_PATH
from scripts.common import DEFAULT_LOG_DIR
from scripts.common import setup_logging


logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the filtered market registry from the local market_universe table using the canonical block-filter protocol."
    )
    p.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="SQLite database path for the shared selected_markets registry.",
    )
    p.add_argument(
        "--min-created-at",
        default="2025-01-01T00:00:00Z",
        help="Lower bound for local market_universe createdAt when rebuilding the filtered registry.",
    )
    p.add_argument(
        "--min-resolved-volume",
        type=float,
        default=DEFAULT_MIN_RESIDUAL_VOLUME,
        help="Minimum cumulative market volume for the final residual volume screen.",
    )
    p.add_argument(
        "--categories",
        nargs="*",
        choices=DOMAIN_PRIORITY,
        default=list(DOMAIN_PRIORITY),
        help="Optional subset of research categories to store when tag enrichment is enabled.",
    )
    p.add_argument(
        "--tag_enrichment",
        "--tag-enrichment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to enrich selected markets with Gamma tags and keep domain categories. Default: false.",
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
    log_path = log_dir / f"market_selection_{timestamp}.log"
    setup_logging(args.log_level, log_path=log_path)

    db_path = Path(args.db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("run logging initialized | log_path=%s", log_path)
    logger.info(
        "market selection protocol | min_created_at=%s min_resolved_volume=%s tag_enrichment=%s",
        args.min_created_at,
        args.min_resolved_volume,
        args.tag_enrichment,
    )

    gamma = GammaClient() if args.tag_enrichment else None
    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        selected_df = select_market_registry_from_universe_all_categories(
            conn,
            gamma=gamma,
            min_created_at=args.min_created_at,
            min_resolved_volume=args.min_resolved_volume,
            categories=args.categories,
            tag_enrichment=args.tag_enrichment,
        )
        logger.info(
            "market selection finished | total_selected=%s db=%s",
            len(selected_df),
            db_path,
        )
        if args.tag_enrichment:
            for category in args.categories:
                count = conn.execute(
                    "SELECT COUNT(*) FROM selected_markets WHERE primary_domain = ?",
                    (category,),
                ).fetchone()[0]
                logger.info("category registry count | category=%s selected_markets=%s", category, count)
        else:
            count = conn.execute(
                "SELECT COUNT(*) FROM selected_markets WHERE primary_domain = ?",
                ("unassigned",),
            ).fetchone()[0]
            logger.info("registry count | primary_domain=unassigned selected_markets=%s", count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
