from __future__ import annotations

import argparse
import ast
from collections.abc import Sequence
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import sqlite3
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clients.gamma_client import GammaClient
from configs.resolved_dataset_domain_config import DOMAIN_KEYWORD_HINTS
from configs.resolved_dataset_domain_config import DOMAIN_PRIORITY
from configs.resolved_dataset_domain_config import DOMAIN_TAG_MAP
from collectors.markets_collector import MarketsCollector
from collectors.trades_collector import TradesCollector


logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("db") / "resolved_probability_dataset.sqlite"
DEFAULT_LOG_DIR = Path("logs")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download resolved Polymarket markets into a 5-minute probability dataset."
    )
    p.add_argument(
        "--category",
        choices=DOMAIN_PRIORITY,
        default="geopolitics",
        help="Research domain to download. Default: geopolitics.",
    )
    p.add_argument(
        "--update",
        action="store_true",
        help="Refresh the resolved market registry from Polymarket before downloading pending markets.",
    )
    p.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="SQLite database path for markets / added_markets / probabilities.",
    )
    p.add_argument(
        "--min-created-at",
        default="2025-01-01T00:00:00Z",
        help="Lower bound for market createdAt when rebuilding the market registry.",
    )
    p.add_argument(
        "--min-resolved-volume",
        type=float,
        default=100_000.0,
        help="Minimum cumulative market volume for resolved candidates.",
    )
    p.add_argument(
        "--max-metadata-pages",
        type=int,
        default=10,
        help="Maximum pages per active/inactive state when refreshing the market registry.",
    )
    p.add_argument(
        "--max-markets",
        type=int,
        default=None,
        help="Optional cap on how many pending markets to process in this run.",
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


def setup_logging(level: str, *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper()))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS markets (
            market_id TEXT PRIMARY KEY,
            condition_id TEXT NOT NULL UNIQUE,
            market_slug TEXT NOT NULL,
            question TEXT NOT NULL,
            description TEXT,
            resolution_source TEXT,
            created_at TEXT,
            end_date TEXT,
            active INTEGER NOT NULL DEFAULT 0,
            closed INTEGER NOT NULL DEFAULT 0,
            archived INTEGER NOT NULL DEFAULT 0,
            volume_num REAL,
            liquidity_num REAL,
            final_outcome TEXT NOT NULL,
            final_yes_probability REAL NOT NULL,
            tag_labels TEXT NOT NULL,
            matched_tags TEXT NOT NULL,
            matched_domains TEXT NOT NULL,
            primary_domain TEXT NOT NULL,
            synced_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS added_markets (
            market_id TEXT PRIMARY KEY,
            condition_id TEXT NOT NULL,
            market_slug TEXT NOT NULL,
            primary_domain TEXT NOT NULL,
            added_at_utc TEXT NOT NULL,
            trade_rows INTEGER NOT NULL,
            probability_rows INTEGER NOT NULL,
            probability_start_utc TEXT,
            probability_end_utc TEXT,
            storage_path TEXT NOT NULL,
            FOREIGN KEY (market_id) REFERENCES markets(market_id)
        );

        CREATE TABLE IF NOT EXISTS probabilities (
            market_id TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            yes_probability REAL NOT NULL,
            observed_trade INTEGER NOT NULL,
            trade_count INTEGER NOT NULL,
            total_size REAL NOT NULL,
            last_trade_price REAL,
            PRIMARY KEY (market_id, timestamp_utc),
            FOREIGN KEY (market_id) REFERENCES markets(market_id)
        );

        CREATE INDEX IF NOT EXISTS idx_markets_primary_domain
            ON markets(primary_domain, volume_num DESC, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_probabilities_timestamp
            ON probabilities(timestamp_utc);
        """
    )


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _parse_list(value: Any) -> list[Any] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return ast.literal_eval(text)
        except Exception:
            return None
    return None


def _parse_binary_prices(value: Any) -> list[float] | None:
    parsed = _parse_list(value)
    if not isinstance(parsed, list) or len(parsed) != 2:
        return None
    try:
        out = [float(x) for x in parsed]
    except Exception:
        return None
    if not all(np.isfinite(x) for x in out):
        return None
    return out


def _parse_binary_outcomes(value: Any) -> list[str] | None:
    parsed = _parse_list(value)
    if not isinstance(parsed, list) or len(parsed) != 2:
        return None
    out = [str(x).strip() for x in parsed]
    return out if {x.lower() for x in out} == {"yes", "no"} else None


def _resolved_outcome_from_prices(
    prices: list[float],
    outcomes: list[str],
) -> tuple[str, float] | tuple[None, None]:
    if len(prices) != 2 or len(outcomes) != 2:
        return None, None
    if abs(sum(prices) - 1.0) > 1e-3:
        return None, None
    winner_idx = int(np.argmax(prices))
    winner_prob = float(prices[winner_idx])
    if winner_prob < 0.99:
        return None, None
    return outcomes[winner_idx], winner_prob


def build_resolved_candidate_pool(markets_df: pd.DataFrame, *, min_resolved_volume: float) -> pd.DataFrame:
    work = markets_df.copy()
    work["closed_bool"] = _to_bool(work["closed"]) if "closed" in work.columns else False
    work["active_bool"] = _to_bool(work["active"]) if "active" in work.columns else False
    work["volume_num_norm"] = pd.to_numeric(
        work.get("volume_num", work.get("volume")),
        errors="coerce",
    ).fillna(0.0)
    work["liquidity_num_norm"] = pd.to_numeric(
        work.get("liquidity_num", work.get("liquidity")),
        errors="coerce",
    )
    work["created_at"] = pd.to_datetime(work.get("created_at"), utc=True, errors="coerce")
    work["end_date"] = pd.to_datetime(work.get("end_date"), utc=True, errors="coerce")
    work["parsed_prices"] = work.get("outcome_prices").map(_parse_binary_prices)
    work["parsed_outcomes"] = work.get("outcomes").map(_parse_binary_outcomes)

    final_outcomes: list[str | None] = []
    final_yes_probabilities: list[float | None] = []
    resolved_like: list[bool] = []
    for prices, outcomes in zip(work["parsed_prices"], work["parsed_outcomes"], strict=False):
        if prices is None or outcomes is None:
            final_outcomes.append(None)
            final_yes_probabilities.append(None)
            resolved_like.append(False)
            continue
        winner, _ = _resolved_outcome_from_prices(prices, outcomes)
        final_outcomes.append(winner)
        final_yes_probabilities.append(float(prices[outcomes.index("Yes")]) if winner is not None else None)
        resolved_like.append(winner is not None)

    work["final_outcome"] = final_outcomes
    work["final_yes_probability"] = final_yes_probabilities
    work["resolved_like"] = resolved_like
    work["has_clob_token_ids"] = work.get("clob_token_ids").notna() if "clob_token_ids" in work.columns else False

    mask = (
        work["closed_bool"]
        & work["resolved_like"]
        & work["has_clob_token_ids"]
        & (work["volume_num_norm"] >= float(min_resolved_volume))
    )

    cols = [
        "id",
        "condition_id",
        "slug",
        "question",
        "description",
        "resolution_source",
        "created_at",
        "end_date",
        "active_bool",
        "closed_bool",
        "archived",
        "volume_num_norm",
        "liquidity_num_norm",
        "final_outcome",
        "final_yes_probability",
    ]
    available_cols = [c for c in cols if c in work.columns]
    return work.loc[mask, available_cols].sort_values(
        ["volume_num_norm", "created_at"],
        ascending=[False, False],
    ).reset_index(drop=True)


def normalize_tag_labels(payload: Any) -> list[str]:
    if not isinstance(payload, list):
        return []
    out: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if label is None:
            continue
        clean = str(label).strip()
        if clean:
            out.append(clean)
    return sorted(set(out))


def classify_domains(
    tag_labels: Sequence[str],
    *,
    question: str | None = None,
    slug: str | None = None,
) -> tuple[list[str], list[str], str | None]:
    tag_set = {str(t).strip() for t in tag_labels if str(t).strip()}
    text_blob = " ".join(part for part in [question or "", slug or ""] if part).lower()

    tag_domains = {domain for domain in DOMAIN_PRIORITY if tag_set.intersection(DOMAIN_TAG_MAP[domain])}
    keyword_domains = {
        domain
        for domain, hints in DOMAIN_KEYWORD_HINTS.items()
        if any(hint in text_blob for hint in hints)
    }

    matched_domains = [domain for domain in DOMAIN_PRIORITY if domain in tag_domains or domain in keyword_domains]
    matched_tags = sorted(tag for tag in tag_set if any(tag in tags for tags in DOMAIN_TAG_MAP.values()))
    primary_domain = matched_domains[0] if matched_domains else None
    return matched_tags, matched_domains, primary_domain


def enrich_candidates_with_tags(candidate_df: pd.DataFrame, gamma_client: GammaClient) -> pd.DataFrame:
    total = len(candidate_df)
    rows: list[dict[str, Any]] = []
    matched = 0
    for idx, (_, row) in enumerate(candidate_df.iterrows(), start=1):
        try:
            tag_payload = gamma_client.get_market_tags(str(row["id"]))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "tag enrichment failed | market_id=%s slug=%s error=%s",
                row.get("id"),
                row.get("slug"),
                exc,
            )
            continue

        tag_labels = normalize_tag_labels(tag_payload)
        matched_tags, matched_domains, primary_domain = classify_domains(
            tag_labels,
            question=row.get("question"),
            slug=row.get("slug"),
        )
        enriched = row.to_dict()
        enriched["tag_labels"] = tag_labels
        enriched["matched_tags"] = matched_tags
        enriched["matched_domains"] = matched_domains
        enriched["primary_domain"] = primary_domain
        if primary_domain is not None:
            matched += 1
        rows.append(enriched)

        if idx % 50 == 0 or idx == total:
            logger.info(
                "tag enrichment progress | processed=%s/%s matched=%s",
                idx,
                total,
                matched,
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _json_text(value: Sequence[str] | None) -> str:
    if not value:
        return "[]"
    return json.dumps(list(value), ensure_ascii=True)


def _dt_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, pd.Timestamp):
        ts = value.tz_convert("UTC") if value.tzinfo is not None else value.tz_localize("UTC")
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, datetime):
        dt = value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    text = str(value).strip()
    return text or None


def upsert_markets(conn: sqlite3.Connection, category: str, candidate_df: pd.DataFrame) -> int:
    now_utc = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows_to_write = []
    for _, row in candidate_df.iterrows():
        if row.get("primary_domain") != category:
            continue
        rows_to_write.append(
            (
                str(row["id"]),
                str(row["condition_id"]),
                str(row["slug"]),
                str(row["question"]),
                row.get("description"),
                row.get("resolution_source"),
                _dt_text(row.get("created_at")),
                _dt_text(row.get("end_date")),
                int(bool(row.get("active_bool"))),
                int(bool(row.get("closed_bool"))),
                int(str(row.get("archived")).strip().lower() in {"true", "1", "yes"}),
                float(row.get("volume_num_norm")) if row.get("volume_num_norm") is not None else None,
                float(row.get("liquidity_num_norm")) if row.get("liquidity_num_norm") is not None else None,
                str(row["final_outcome"]),
                float(row["final_yes_probability"]),
                _json_text(row.get("tag_labels")),
                _json_text(row.get("matched_tags")),
                _json_text(row.get("matched_domains")),
                str(row["primary_domain"]),
                now_utc,
            )
        )

    if not rows_to_write:
        return 0

    conn.executemany(
        """
        INSERT INTO markets (
            market_id,
            condition_id,
            market_slug,
            question,
            description,
            resolution_source,
            created_at,
            end_date,
            active,
            closed,
            archived,
            volume_num,
            liquidity_num,
            final_outcome,
            final_yes_probability,
            tag_labels,
            matched_tags,
            matched_domains,
            primary_domain,
            synced_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(market_id) DO UPDATE SET
            condition_id = excluded.condition_id,
            market_slug = excluded.market_slug,
            question = excluded.question,
            description = excluded.description,
            resolution_source = excluded.resolution_source,
            created_at = excluded.created_at,
            end_date = excluded.end_date,
            active = excluded.active,
            closed = excluded.closed,
            archived = excluded.archived,
            volume_num = excluded.volume_num,
            liquidity_num = excluded.liquidity_num,
            final_outcome = excluded.final_outcome,
            final_yes_probability = excluded.final_yes_probability,
            tag_labels = excluded.tag_labels,
            matched_tags = excluded.matched_tags,
            matched_domains = excluded.matched_domains,
            primary_domain = excluded.primary_domain,
            synced_at_utc = excluded.synced_at_utc
        """,
        rows_to_write,
    )
    return len(rows_to_write)


def upsert_markets_for_categories(
    conn: sqlite3.Connection,
    candidate_df: pd.DataFrame,
    *,
    categories: Sequence[str] | None = None,
) -> dict[str, int]:
    category_list = list(categories) if categories is not None else list(DOMAIN_PRIORITY)
    out: dict[str, int] = {}
    for category in category_list:
        out[category] = upsert_markets(conn, category, candidate_df)
    return out


def load_markets_for_category(conn: sqlite3.Connection, category: str) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            market_id,
            condition_id,
            market_slug,
            question,
            description,
            resolution_source,
            created_at,
            end_date,
            active,
            closed,
            archived,
            volume_num,
            liquidity_num,
            final_outcome,
            final_yes_probability,
            tag_labels,
            matched_tags,
            matched_domains,
            primary_domain,
            synced_at_utc
        FROM markets
        WHERE primary_domain = ?
        ORDER BY volume_num DESC, created_at DESC
        """,
        conn,
        params=(category,),
    )


def build_pending_queue(conn: sqlite3.Connection, category: str) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            m.market_id,
            m.condition_id,
            m.market_slug,
            m.question,
            m.description,
            m.resolution_source,
            m.created_at,
            m.end_date,
            m.volume_num,
            m.final_outcome,
            m.final_yes_probability,
            m.tag_labels,
            m.matched_tags,
            m.matched_domains,
            m.primary_domain
        FROM markets AS m
        LEFT JOIN added_markets AS a
            ON a.market_id = m.market_id
        WHERE m.primary_domain = ?
          AND a.market_id IS NULL
        ORDER BY m.volume_num DESC, m.created_at DESC
        """,
        conn,
        params=(category,),
    )


def build_yes_probability_series_5m(trades_df: pd.DataFrame, market_id: str) -> pd.DataFrame:
    work = trades_df.copy()
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="coerce")
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["size"] = pd.to_numeric(work["size"], errors="coerce")
    work["outcome"] = work["outcome"].astype("string")

    outcome_norm = work["outcome"].str.strip().str.lower()
    work["yes_probability"] = np.where(
        outcome_norm.eq("yes"),
        work["price"],
        np.where(outcome_norm.eq("no"), 1.0 - work["price"], np.nan),
    )
    work = work.dropna(subset=["timestamp_utc", "yes_probability"]).sort_values("timestamp_utc").reset_index(drop=True)
    if work.empty:
        raise RuntimeError("No usable Yes/No trade history remained after normalization.")

    agg = (
        work.set_index("timestamp_utc")
        .groupby(pd.Grouper(freq="5min"))
        .agg(
            last_yes_probability=("yes_probability", "last"),
            trade_count=("transaction_hash", "size"),
            total_size=("size", "sum"),
            last_trade_price=("price", "last"),
        )
    )

    grid = pd.date_range(
        start=work["timestamp_utc"].min().floor("5min"),
        end=work["timestamp_utc"].max().ceil("5min"),
        freq="5min",
        tz="UTC",
    )

    panel = pd.DataFrame(index=grid).join(agg, how="left")
    panel.index.name = "timestamp_utc"
    panel["yes_probability"] = panel["last_yes_probability"].ffill()
    panel["trade_count"] = panel["trade_count"].fillna(0).astype(int)
    panel["total_size"] = panel["total_size"].fillna(0.0)
    panel["observed_trade"] = panel["last_yes_probability"].notna().astype(int)
    panel["market_id"] = str(market_id)
    panel = panel.reset_index()
    panel["timestamp_utc"] = pd.to_datetime(panel["timestamp_utc"], utc=True, errors="coerce").dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return panel[
        [
            "market_id",
            "timestamp_utc",
            "yes_probability",
            "observed_trade",
            "trade_count",
            "total_size",
            "last_trade_price",
        ]
    ]


def store_market_dataset(
    conn: sqlite3.Connection,
    *,
    market_row: pd.Series,
    probability_df: pd.DataFrame,
    trade_rows: int,
    storage_path: str,
) -> None:
    market_id = str(market_row["market_id"])
    now_utc = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    probability_rows = len(probability_df)
    prob_start = probability_df["timestamp_utc"].min() if probability_rows else None
    prob_end = probability_df["timestamp_utc"].max() if probability_rows else None

    with conn:
        conn.execute("DELETE FROM probabilities WHERE market_id = ?", (market_id,))
        probability_df.to_sql(
            "probabilities",
            conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1_000,
        )
        conn.execute(
            """
            INSERT INTO added_markets (
                market_id,
                condition_id,
                market_slug,
                primary_domain,
                added_at_utc,
                trade_rows,
                probability_rows,
                probability_start_utc,
                probability_end_utc,
                storage_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_id) DO UPDATE SET
                condition_id = excluded.condition_id,
                market_slug = excluded.market_slug,
                primary_domain = excluded.primary_domain,
                added_at_utc = excluded.added_at_utc,
                trade_rows = excluded.trade_rows,
                probability_rows = excluded.probability_rows,
                probability_start_utc = excluded.probability_start_utc,
                probability_end_utc = excluded.probability_end_utc,
                storage_path = excluded.storage_path
            """,
            (
                market_id,
                str(market_row["condition_id"]),
                str(market_row["market_slug"]),
                str(market_row["primary_domain"]),
                now_utc,
                int(trade_rows),
                int(probability_rows),
                prob_start,
                prob_end,
                storage_path,
            ),
        )


def refresh_market_registry(
    conn: sqlite3.Connection,
    *,
    gamma: GammaClient,
    category: str,
    min_created_at: str,
    min_resolved_volume: float,
    max_metadata_pages: int,
) -> pd.DataFrame:
    logger.info(
        "registry refresh started | category=%s min_created_at=%s max_metadata_pages=%s",
        category,
        min_created_at,
        max_metadata_pages,
    )
    logger.info(
        "registry refresh fetch mode | include_active=%s include_inactive=%s note=%s",
        True,
        False,
        "Gamma /markets currently returns overlapping universes for active=True and active=False; using a single pass",
    )
    markets_collector = MarketsCollector(gamma)
    report = markets_collector.download_market_meta(
        include_active=True,
        include_inactive=False,
        limit=200,
        max_pages=max_metadata_pages,
        min_created_at=min_created_at,
        show_progress=True,
        estimate_total=False,
        frame_type="pandas",
    )
    markets_df = report["markets"]
    raw_candidate_df = build_resolved_candidate_pool(markets_df, min_resolved_volume=min_resolved_volume)
    logger.info(
        "registry metadata ready | total_markets=%s resolved_candidates=%s",
        len(markets_df),
        len(raw_candidate_df),
    )
    enriched_df = enrich_candidates_with_tags(raw_candidate_df, gamma)
    if enriched_df.empty:
        logger.warning("registry refresh found no tag-enriched candidates | category=%s", category)
        return enriched_df

    selected_df = enriched_df.loc[enriched_df["primary_domain"] == category].reset_index(drop=True)
    upserted = upsert_markets(conn, category, selected_df)
    logger.info(
        "registry refresh finished | category=%s selected=%s upserted=%s",
        category,
        len(selected_df),
        upserted,
    )
    return selected_df


def refresh_market_registry_all_categories(
    conn: sqlite3.Connection,
    *,
    gamma: GammaClient,
    min_created_at: str,
    min_resolved_volume: float,
    max_metadata_pages: int,
    categories: Sequence[str] | None = None,
) -> pd.DataFrame:
    category_list = list(categories) if categories is not None else list(DOMAIN_PRIORITY)
    logger.info(
        "registry refresh started | categories=%s min_created_at=%s max_metadata_pages=%s",
        ",".join(category_list),
        min_created_at,
        max_metadata_pages,
    )
    logger.info(
        "registry refresh fetch mode | include_active=%s include_inactive=%s note=%s",
        True,
        False,
        "Gamma /markets currently returns overlapping universes for active=True and active=False; using a single pass",
    )
    markets_collector = MarketsCollector(gamma)
    report = markets_collector.download_market_meta(
        include_active=True,
        include_inactive=False,
        limit=200,
        max_pages=max_metadata_pages,
        min_created_at=min_created_at,
        show_progress=True,
        estimate_total=False,
        frame_type="pandas",
    )
    markets_df = report["markets"]
    raw_candidate_df = build_resolved_candidate_pool(markets_df, min_resolved_volume=min_resolved_volume)
    logger.info(
        "registry metadata ready | total_markets=%s resolved_candidates=%s",
        len(markets_df),
        len(raw_candidate_df),
    )
    enriched_df = enrich_candidates_with_tags(raw_candidate_df, gamma)
    if enriched_df.empty:
        logger.warning("registry refresh found no tag-enriched candidates")
        return enriched_df

    selected_df = enriched_df.loc[enriched_df["primary_domain"].isin(category_list)].reset_index(drop=True)
    upsert_counts = upsert_markets_for_categories(conn, selected_df, categories=category_list)
    selected_counts = {
        category: int((selected_df["primary_domain"] == category).sum())
        for category in category_list
    }
    logger.info(
        "registry refresh finished | selected_counts=%s upsert_counts=%s",
        json.dumps(selected_counts, ensure_ascii=True, sort_keys=True),
        json.dumps(upsert_counts, ensure_ascii=True, sort_keys=True),
    )
    return selected_df


def download_pending_markets(
    conn: sqlite3.Connection,
    *,
    gamma: GammaClient,
    category: str,
    db_path: str,
    max_markets: int | None,
) -> int:
    queue_df = build_pending_queue(conn, category)
    if max_markets is not None:
        queue_df = queue_df.head(int(max_markets)).reset_index(drop=True)

    logger.info(
        "download queue prepared | category=%s pending_markets=%s db=%s",
        category,
        len(queue_df),
        db_path,
    )
    if queue_df.empty:
        return 0

    trades_collector = TradesCollector(gamma)
    completed = 0

    for _, market_row in queue_df.iterrows():
        market_id = str(market_row["market_id"])
        condition_id = str(market_row["condition_id"])
        slug = str(market_row["market_slug"])
        logger.info(
            "dataset download started | category=%s market_id=%s slug=%s",
            category,
            market_id,
            slug,
        )
        try:
            trades_df = trades_collector.download_all_trades(
                condition_id,
                frame_type="pandas",
                show_progress=False,
                estimate_total=True,
            )
            trade_rows = int(len(trades_df))
            probability_df = build_yes_probability_series_5m(trades_df, market_id)
            store_market_dataset(
                conn,
                market_row=market_row,
                probability_df=probability_df,
                trade_rows=trade_rows,
                storage_path=db_path,
            )
            completed += 1
            logger.info(
                "dataset download finished | category=%s market_id=%s slug=%s trade_rows=%s probability_rows=%s saved_to=%s",
                category,
                market_id,
                slug,
                trade_rows,
                len(probability_df),
                db_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "dataset download failed | category=%s market_id=%s slug=%s error=%s",
                category,
                market_id,
                slug,
                exc,
            )

    return completed


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_path = log_dir / f"download_resolved_probability_dataset_{args.category}_{timestamp}.log"
    setup_logging(args.log_level, log_path=log_path)

    db_path = Path(args.db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("run logging initialized | log_path=%s", log_path)

    gamma = GammaClient()
    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)

        need_refresh = args.update
        if not table_exists(conn, "markets"):
            need_refresh = True
        elif load_markets_for_category(conn, args.category).empty:
            logger.info(
                "markets table has no rows for category=%s; forcing registry refresh",
                args.category,
            )
            need_refresh = True

        if need_refresh:
            refresh_market_registry(
                conn,
                gamma=gamma,
                category=args.category,
                min_created_at=args.min_created_at,
                min_resolved_volume=args.min_resolved_volume,
                max_metadata_pages=args.max_metadata_pages,
            )

        available_df = load_markets_for_category(conn, args.category)
        logger.info(
            "market registry loaded | category=%s available_markets=%s",
            args.category,
            len(available_df),
        )
        if available_df.empty:
            logger.warning("no markets available for category=%s", args.category)
            return 0

        already_added = conn.execute(
            "SELECT COUNT(*) FROM added_markets WHERE primary_domain = ?",
            (args.category,),
        ).fetchone()[0]
        pending_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM markets AS m
            LEFT JOIN added_markets AS a
                ON a.market_id = m.market_id
            WHERE m.primary_domain = ?
              AND a.market_id IS NULL
            """,
            (args.category,),
        ).fetchone()[0]
        logger.info(
            "category download summary | category=%s total_markets=%s already_downloaded=%s pending_download=%s",
            args.category,
            len(available_df),
            already_added,
            pending_count,
        )

        completed = download_pending_markets(
            conn,
            gamma=gamma,
            category=args.category,
            db_path=str(db_path),
            max_markets=args.max_markets,
        )
        logger.info(
            "run finished | category=%s completed_markets=%s db=%s",
            args.category,
            completed,
            db_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
