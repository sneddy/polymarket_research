"""SQLite upsert helpers for market metadata tables."""

from __future__ import annotations

import json
import ast
from datetime import UTC
from datetime import datetime
import sqlite3
from typing import Any
from typing import Sequence

import pandas as pd
import numpy as np

from configs.resolved_dataset_domain_config import DOMAIN_PRIORITY


def json_text(value: Sequence[str] | None) -> str:
    """Serialize a small list as stable JSON text."""
    if not value:
        return "[]"
    return json.dumps(list(value), ensure_ascii=True)


def dt_text(value: Any) -> str | None:
    """Normalize a timestamp-like value into UTC text."""
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


def extract_primary_event_fields(events_payload: Any) -> dict[str, str | None]:
    """Extract a compact event-level metadata view from a Gamma market payload."""
    default = {
        "event_id": None,
        "event_slug": None,
        "event_title": None,
        "event_series_slug": None,
        "event_description": None,
        "event_start_time": None,
        "event_score": None,
        "event_period": None,
        "event_series_id": None,
        "event_recurrence": None,
        "event_series_type": None,
    }
    if not isinstance(events_payload, list) or not events_payload:
        return default

    event = events_payload[0]
    if not isinstance(event, dict):
        return default

    series = event.get("series")
    if isinstance(series, list) and series and isinstance(series[0], dict):
        series_item = series[0]
    else:
        series_item = {}

    def text_value(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    return {
        "event_id": text_value(event.get("id")),
        "event_slug": text_value(event.get("slug")),
        "event_title": text_value(event.get("title")),
        "event_series_slug": text_value(event.get("seriesSlug") or event.get("series_slug")),
        "event_description": text_value(event.get("description")),
        "event_start_time": text_value(event.get("startTime") or event.get("start_time")),
        "event_score": text_value(event.get("score")),
        "event_period": text_value(event.get("period")),
        "event_series_id": text_value(series_item.get("id")),
        "event_recurrence": text_value(series_item.get("recurrence")),
        "event_series_type": text_value(series_item.get("seriesType") or series_item.get("series_type")),
    }


def parse_list(value: Any) -> list[Any] | None:
    """Parse a list-like payload stored as Python/JSON text."""
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


def parse_binary_prices(value: Any) -> list[float] | None:
    """Parse a binary-outcome price list."""
    parsed = parse_list(value)
    if not isinstance(parsed, list) or len(parsed) != 2:
        return None
    try:
        out = [float(x) for x in parsed]
    except Exception:
        return None
    if not all(np.isfinite(x) for x in out):
        return None
    return out


def parse_binary_outcomes(value: Any) -> list[str] | None:
    """Parse a binary-outcome label list."""
    parsed = parse_list(value)
    if not isinstance(parsed, list) or len(parsed) != 2:
        return None
    out = [str(x).strip() for x in parsed]
    return out if {x.lower() for x in out} == {"yes", "no"} else None


def resolved_outcome_from_prices(prices: list[float], outcomes: list[str]) -> str | None:
    """Infer a resolved binary outcome from near-extreme outcome prices."""
    if len(prices) != 2 or len(outcomes) != 2:
        return None
    if abs(sum(prices) - 1.0) > 1e-3:
        return None
    winner_idx = int(np.argmax(prices))
    winner_prob = float(prices[winner_idx])
    if winner_prob < 0.99:
        return None
    return outcomes[winner_idx]


def list_text(value: Any) -> str | None:
    """Normalize a list-like payload into compact JSON text."""
    parsed = parse_list(value)
    if isinstance(parsed, list):
        return json.dumps(parsed, ensure_ascii=True)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def upsert_market_universe(conn: sqlite3.Connection, markets_df: pd.DataFrame) -> int:
    """Persist the full fetched market universe separately from the filtered registry."""
    if markets_df.empty:
        return 0

    work = markets_df.copy()
    if "events" in work.columns:
        event_fields = work["events"].map(extract_primary_event_fields).apply(pd.Series)
        for column in (
            "event_id",
            "event_slug",
            "event_title",
            "event_series_slug",
            "event_description",
            "event_start_time",
            "event_score",
            "event_period",
            "event_series_id",
            "event_recurrence",
            "event_series_type",
        ):
            work[column] = event_fields[column]
    else:
        work["event_id"] = None
        work["event_slug"] = None
        work["event_title"] = None
        work["event_series_slug"] = None
        work["event_description"] = None
        work["event_start_time"] = None
        work["event_score"] = None
        work["event_period"] = None
        work["event_series_id"] = None
        work["event_recurrence"] = None
        work["event_series_type"] = None

    now_utc = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def bool_int(value: Any) -> int | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"true", "1", "yes"}:
            return 1
        if text in {"false", "0", "no", ""}:
            return 0
        return None

    rows_to_write = []
    for _, row in work.iterrows():
        rows_to_write.append(
            (
                str(row.get("id") or ""),
                row.get("condition_id"),
                row.get("slug"),
                row.get("event_id"),
                row.get("event_slug"),
                row.get("event_title"),
                row.get("event_series_slug"),
                row.get("event_description"),
                dt_text(row.get("event_start_time")),
                row.get("event_score"),
                row.get("event_period"),
                row.get("event_series_id"),
                row.get("event_recurrence"),
                row.get("event_series_type"),
                row.get("question"),
                row.get("description"),
                row.get("resolution_source"),
                dt_text(row.get("created_at")),
                dt_text(row.get("end_date")),
                bool_int(row.get("closed")),
                bool_int(row.get("archived")),
                pd.to_numeric(row.get("volume_num", row.get("volume")), errors="coerce"),
                pd.to_numeric(row.get("liquidity_num", row.get("liquidity")), errors="coerce"),
                list_text(row.get("outcomes")),
                list_text(row.get("outcome_prices")),
                list_text(row.get("clob_token_ids") or row.get("clobTokenIds")),
                dt_text(row.get("closed_time")),
                row.get("uma_resolution_status"),
                bool_int(row.get("neg_risk")),
                row.get("neg_risk_market_id"),
                row.get("group_item_title"),
                now_utc,
            )
        )

    conn.executemany(
        """
        INSERT INTO market_universe (
            market_id,
            condition_id,
            market_slug,
            event_id,
            event_slug,
            event_title,
            event_series_slug,
            event_description,
            event_start_time,
            event_score,
            event_period,
            event_series_id,
            event_recurrence,
            event_series_type,
            question,
            description,
            resolution_source,
            created_at,
            end_date,
            closed,
            archived,
            volume_num,
            liquidity_num,
            outcomes,
            outcome_prices,
            clob_token_ids,
            closed_time,
            uma_resolution_status,
            neg_risk,
            neg_risk_market_id,
            group_item_title,
            synced_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(market_id) DO UPDATE SET
            condition_id = excluded.condition_id,
            market_slug = excluded.market_slug,
            event_id = excluded.event_id,
            event_slug = excluded.event_slug,
            event_title = excluded.event_title,
            event_series_slug = excluded.event_series_slug,
            event_description = excluded.event_description,
            event_start_time = excluded.event_start_time,
            event_score = excluded.event_score,
            event_period = excluded.event_period,
            event_series_id = excluded.event_series_id,
            event_recurrence = excluded.event_recurrence,
            event_series_type = excluded.event_series_type,
            question = excluded.question,
            description = excluded.description,
            resolution_source = excluded.resolution_source,
            created_at = excluded.created_at,
            end_date = excluded.end_date,
            closed = excluded.closed,
            archived = excluded.archived,
            volume_num = excluded.volume_num,
            liquidity_num = excluded.liquidity_num,
            outcomes = excluded.outcomes,
            outcome_prices = excluded.outcome_prices,
            clob_token_ids = excluded.clob_token_ids,
            closed_time = excluded.closed_time,
            uma_resolution_status = excluded.uma_resolution_status,
            neg_risk = excluded.neg_risk,
            neg_risk_market_id = excluded.neg_risk_market_id,
            group_item_title = excluded.group_item_title,
            synced_at_utc = excluded.synced_at_utc
        """,
        rows_to_write,
    )
    return len(rows_to_write)


def upsert_markets(conn: sqlite3.Connection, category: str, candidate_df: pd.DataFrame) -> int:
    """Upsert filtered markets for a single research category."""
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
                row.get("event_id"),
                row.get("event_slug"),
                row.get("event_title"),
                row.get("event_series_slug"),
                row.get("event_description"),
                dt_text(row.get("event_start_time")),
                row.get("event_score"),
                row.get("event_period"),
                row.get("event_series_id"),
                row.get("event_recurrence"),
                row.get("event_series_type"),
                str(row["question"]),
                row.get("description"),
                row.get("resolution_source"),
                dt_text(row.get("created_at")),
                dt_text(row.get("end_date")),
                int(bool(row.get("active_bool"))),
                int(bool(row.get("closed_bool"))),
                int(str(row.get("archived")).strip().lower() in {"true", "1", "yes"}),
                float(row.get("volume_num_norm")) if row.get("volume_num_norm") is not None else None,
                float(row.get("liquidity_num_norm")) if row.get("liquidity_num_norm") is not None else None,
                list_text(row.get("outcomes")),
                list_text(row.get("outcome_prices")),
                list_text(row.get("clob_token_ids")),
                dt_text(row.get("closed_time")),
                row.get("uma_resolution_status"),
                row.get("neg_risk"),
                row.get("neg_risk_market_id"),
                row.get("group_item_title"),
                str(row["final_outcome"]),
                float(row["final_yes_probability"]),
                json_text(row.get("tag_labels")),
                json_text(row.get("matched_tags")),
                json_text(row.get("matched_domains")),
                str(row["primary_domain"]),
                now_utc,
            )
        )

    if not rows_to_write:
        return 0

    conn.executemany(
        """
        INSERT INTO selected_markets (
            market_id,
            condition_id,
            market_slug,
            event_id,
            event_slug,
            event_title,
            event_series_slug,
            event_description,
            event_start_time,
            event_score,
            event_period,
            event_series_id,
            event_recurrence,
            event_series_type,
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
            outcomes,
            outcome_prices,
            clob_token_ids,
            closed_time,
            uma_resolution_status,
            neg_risk,
            neg_risk_market_id,
            group_item_title,
            final_outcome,
            final_yes_probability,
            tag_labels,
            matched_tags,
            matched_domains,
            primary_domain,
            synced_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(market_id) DO UPDATE SET
            condition_id = excluded.condition_id,
            market_slug = excluded.market_slug,
            event_id = excluded.event_id,
            event_slug = excluded.event_slug,
            event_title = excluded.event_title,
            event_series_slug = excluded.event_series_slug,
            event_description = excluded.event_description,
            event_start_time = excluded.event_start_time,
            event_score = excluded.event_score,
            event_period = excluded.event_period,
            event_series_id = excluded.event_series_id,
            event_recurrence = excluded.event_recurrence,
            event_series_type = excluded.event_series_type,
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
            outcomes = excluded.outcomes,
            outcome_prices = excluded.outcome_prices,
            clob_token_ids = excluded.clob_token_ids,
            closed_time = excluded.closed_time,
            uma_resolution_status = excluded.uma_resolution_status,
            neg_risk = excluded.neg_risk,
            neg_risk_market_id = excluded.neg_risk_market_id,
            group_item_title = excluded.group_item_title,
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
    """Upsert filtered markets for multiple research categories."""
    category_list = list(categories) if categories is not None else list(DOMAIN_PRIORITY)
    out: dict[str, int] = {}
    for category in category_list:
        out[category] = upsert_markets(conn, category, candidate_df)
    return out


def replace_markets_for_categories(
    conn: sqlite3.Connection,
    candidate_df: pd.DataFrame,
    *,
    categories: Sequence[str] | None = None,
) -> dict[str, int]:
    """Replace filtered markets for the requested categories with a fresh selection."""
    category_list = list(categories) if categories is not None else list(DOMAIN_PRIORITY)
    if category_list:
        placeholders = ",".join("?" for _ in category_list)
        market_scope_sql = f"SELECT market_id FROM selected_markets WHERE primary_domain IN ({placeholders})"
        conn.execute(
            f"DELETE FROM probabilities WHERE market_id IN ({market_scope_sql})",
            tuple(category_list),
        )
        conn.execute(
            f"DELETE FROM added_markets WHERE market_id IN ({market_scope_sql})",
            tuple(category_list),
        )
        conn.execute(
            f"DELETE FROM selected_markets WHERE primary_domain IN ({placeholders})",
            tuple(category_list),
        )
    return upsert_markets_for_categories(conn, candidate_df, categories=category_list)


def load_markets_for_category(conn: sqlite3.Connection, category: str) -> pd.DataFrame:
    """Load the filtered market registry for a single research category."""
    return pd.read_sql_query(
        """
        SELECT
            market_id,
            condition_id,
            market_slug,
            event_id,
            event_slug,
            event_title,
            event_series_slug,
            event_description,
            event_start_time,
            event_score,
            event_period,
            event_series_id,
            event_recurrence,
            event_series_type,
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
            outcomes,
            outcome_prices,
            clob_token_ids,
            closed_time,
            uma_resolution_status,
            neg_risk,
            neg_risk_market_id,
            group_item_title,
            final_outcome,
            final_yes_probability,
            tag_labels,
            matched_tags,
            matched_domains,
            primary_domain,
            synced_at_utc
        FROM selected_markets
        WHERE primary_domain = ?
        ORDER BY volume_num DESC, created_at DESC
        """,
        conn,
        params=(category,),
    )


def load_all_markets(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load the full filtered market registry without category filtering."""
    return pd.read_sql_query(
        """
        SELECT
            market_id,
            condition_id,
            market_slug,
            event_id,
            event_slug,
            event_title,
            event_series_slug,
            event_description,
            event_start_time,
            event_score,
            event_period,
            event_series_id,
            event_recurrence,
            event_series_type,
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
            outcomes,
            outcome_prices,
            clob_token_ids,
            closed_time,
            uma_resolution_status,
            neg_risk,
            neg_risk_market_id,
            group_item_title,
            final_outcome,
            final_yes_probability,
            tag_labels,
            matched_tags,
            matched_domains,
            primary_domain,
            synced_at_utc
        FROM selected_markets
        ORDER BY volume_num DESC, created_at DESC
        """,
        conn,
    )
