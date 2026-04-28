"""Small helper functions for loading and persisting the basic market dataset."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Literal, Sequence

import pandas as pd


def resolve_repo_root(start: str | Path | None = None) -> Path:
    """Resolve the repository root by walking upward until current repo markers are found."""

    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / "polymarket_research").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    raise RuntimeError(f"Could not locate repository root from start={current}")


_SOURCE_NAME = Literal["polymarket", "kalshi"]


def default_db_path(
    repo_root: str | Path | None = None,
    *,
    source: _SOURCE_NAME = "polymarket",
) -> Path:
    """Return the default SQLite path for the requested source."""

    root = resolve_repo_root(repo_root)
    filename = {
        "polymarket": "resolved_probability_dataset.sqlite",
        "kalshi": "kalshi_probability_dataset.sqlite",
    }[source]
    return root / "db" / filename


def open_sqlite_dataset(
    db_path: str | Path | None = None,
    *,
    source: _SOURCE_NAME = "polymarket",
) -> sqlite3.Connection:
    """Open the local SQLite dataset that stores resolved market histories and probabilities."""

    conn = sqlite3.connect(str(db_path or default_db_path(source=source)))
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def infer_data_source(conn: sqlite3.Connection) -> _SOURCE_NAME:
    """Infer the dataset source from available SQLite tables/columns."""

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "raw_series" in tables or "selected_series" in tables:
        return "kalshi"

    selected_market_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(selected_markets)").fetchall()
    }
    if {"venue_market_id", "kalshi_category"} & selected_market_columns:
        return "kalshi"
    return "polymarket"


def load_selected_markets(
    conn: sqlite3.Connection,
    *,
    source: _SOURCE_NAME | Literal["auto"] = "auto",
) -> pd.DataFrame:
    """Load the selected-market registry plus export download metadata."""

    resolved_source: _SOURCE_NAME = infer_data_source(conn) if source == "auto" else source
    if resolved_source == "kalshi":
        frame = _load_selected_markets_kalshi(conn)
    else:
        frame = _load_selected_markets_polymarket(conn)
    return _normalize_market_frame(frame)


def _load_selected_markets_polymarket(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load the Polymarket selected-market registry plus export download metadata."""

    query = """
    SELECT
        m.market_id,
        m.condition_id,
        COALESCE(u.market_slug, m.market_slug) AS market_slug,
        COALESCE(u.event_id, m.event_id) AS event_id,
        COALESCE(u.event_slug, m.event_slug) AS event_slug,
        COALESCE(u.event_title, m.event_title) AS event_title,
        COALESCE(u.event_series_slug, m.event_series_slug) AS event_series_slug,
        COALESCE(u.question, m.question) AS question,
        COALESCE(u.description, m.description) AS description,
        COALESCE(u.resolution_source, m.resolution_source) AS resolution_source,
        COALESCE(m.active, 0) AS active,
        COALESCE(u.closed, m.closed, 0) AS closed,
        COALESCE(u.archived, m.archived, 0) AS archived,
        COALESCE(u.created_at, m.created_at) AS created_at,
        COALESCE(u.end_date, m.end_date) AS end_date,
        COALESCE(u.volume_num, m.volume_num) AS volume_num,
        COALESCE(u.liquidity_num, m.liquidity_num) AS liquidity_num,
        m.final_outcome,
        m.final_yes_probability,
        m.tag_labels,
        m.matched_tags,
        m.matched_domains,
        m.primary_domain AS research_category,
        NULL AS kalshi_category,
        COALESCE(u.synced_at_utc, m.synced_at_utc) AS synced_at_utc,
        a.added_at_utc,
        a.trade_rows,
        a.probability_rows,
        a.probability_start_utc,
        a.probability_end_utc,
        a.raw_trade_rows,
        a.raw_trade_start_utc,
        a.raw_trade_end_utc,
        a.raw_trades_saved,
        'polymarket' AS source
    FROM selected_markets AS m
    LEFT JOIN market_universe AS u
        ON u.market_id = m.market_id
    LEFT JOIN added_markets AS a
        ON a.market_id = m.market_id
    ORDER BY COALESCE(u.created_at, m.created_at) DESC, COALESCE(u.volume_num, m.volume_num, 0.0) DESC
    """
    return pd.read_sql_query(query, conn)


def _load_selected_markets_kalshi(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load the Kalshi selected-market registry into the shared selected-market contract."""

    query = """
    SELECT
        m.market_id,
        NULL AS condition_id,
        COALESCE(m.ticker, u.ticker, m.venue_market_id, a.venue_market_id, m.market_id) AS market_slug,
        COALESCE(m.event_id, u.event_id) AS event_id,
        COALESCE(m.event_ticker, u.event_ticker, m.venue_event_id) AS event_slug,
        COALESCE(m.event_title, u.event_title) AS event_title,
        COALESCE(m.series_ticker, u.series_ticker, a.series_ticker) AS event_series_slug,
        COALESCE(m.question, u.question, u.title) AS question,
        COALESCE(m.description, u.description) AS description,
        NULL AS resolution_source,
        COALESCE(m.is_active, u.is_active, 0) AS active,
        COALESCE(m.is_closed, u.is_closed, 0) AS closed,
        0 AS archived,
        COALESCE(m.created_at, u.created_at) AS created_at,
        COALESCE(m.end_date, u.end_date, m.settlement_ts, u.settlement_ts, m.close_time, u.close_time) AS end_date,
        COALESCE(m.volume_num, u.volume_num) AS volume_num,
        COALESCE(m.liquidity_dollars, u.liquidity_dollars) AS liquidity_num,
        m.final_outcome,
        m.final_yes_probability,
        NULL AS tag_labels,
        NULL AS matched_tags,
        NULL AS matched_domains,
        m.primary_domain AS research_category,
        m.kalshi_category,
        COALESCE(m.synced_at_utc, u.synced_at_utc) AS synced_at_utc,
        a.added_at_utc,
        COALESCE(a.raw_trade_rows, 0) AS trade_rows,
        a.probability_rows,
        a.probability_start_utc,
        a.probability_end_utc,
        a.raw_trade_rows,
        a.raw_trade_start_utc,
        a.raw_trade_end_utc,
        a.raw_trades_saved,
        COALESCE(m.is_resolved, u.is_resolved, 0) AS resolved,
        COALESCE(m.source, a.source, 'kalshi') AS source
    FROM selected_markets AS m
    LEFT JOIN market_universe AS u
        ON u.market_id = m.market_id
    LEFT JOIN added_markets AS a
        ON a.market_id = m.market_id
    ORDER BY COALESCE(m.created_at, u.created_at) DESC, COALESCE(m.volume_num, u.volume_num, 0.0) DESC
    """
    return pd.read_sql_query(query, conn)


def load_probabilities_for_market_frame(
    conn: sqlite3.Connection,
    markets: pd.DataFrame,
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Load probability history for all market ids contained in the provided dataframe."""

    return _load_probabilities_for_markets(
        conn,
        markets["market_id"].tolist(),
        show_progress=show_progress,
    )


def save_dataset_frames(
    *,
    directory: str | Path,
    markets: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """Persist the core market and probability tables as parquet files and return a manifest."""

    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        ("markets.parquet", markets),
        ("probabilities.parquet", probabilities),
    ]
    manifest_rows: list[dict[str, object]] = []
    for filename, frame in outputs:
        frame.to_parquet(target_dir / filename, index=False)
        manifest_rows.append({"file": filename, "rows": len(frame), "cols": frame.shape[1]})
    return pd.DataFrame(manifest_rows)


def load_saved_dataset_frames(directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load previously saved core dataset parquet files from a directory."""

    source_dir = Path(directory)
    markets = pd.read_parquet(source_dir / "markets.parquet")
    probabilities = pd.read_parquet(source_dir / "probabilities.parquet")
    return markets, probabilities


def _load_probabilities_for_markets(
    conn: sqlite3.Connection,
    market_ids: Sequence[str],
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Load probability trajectories for the requested market ids from SQLite."""

    market_ids = [str(market_id) for market_id in market_ids]
    if not market_ids:
        return pd.DataFrame(
            columns=[
                "market_id",
                "timestamp_utc",
                "yes_probability",
                "observed_trade",
                "trade_count",
                "total_size",
                "last_trade_price",
            ]
        )

    chunk_starts = range(0, len(market_ids), 250)
    if show_progress:
        from tqdm.auto import tqdm

        chunk_starts = tqdm(
            chunk_starts,
            total=((len(market_ids) + 249) // 250),
            desc="sqlite probabilities",
            unit="chunk",
        )

    frames: list[pd.DataFrame] = []
    for chunk_start in chunk_starts:
        chunk = market_ids[chunk_start : chunk_start + 250]
        placeholders = ",".join(["?"] * len(chunk))
        query = f"""
        SELECT
            market_id,
            timestamp_utc,
            yes_probability,
            observed_trade,
            trade_count,
            total_size,
            last_trade_price
        FROM probabilities
        WHERE market_id IN ({placeholders})
        ORDER BY market_id, timestamp_utc
        """
        frames.append(pd.read_sql_query(query, conn, params=tuple(chunk)))

    if show_progress:
        total_rows = int(sum(len(frame) for frame in frames))
        print(f"[sqlite probabilities] fetched chunks={len(frames)} raw_rows={total_rows}")
        print("[sqlite probabilities] concatenating chunks")
    out = pd.concat(frames, ignore_index=True)
    if show_progress:
        print(f"[sqlite probabilities] concatenated rows={len(out)}")
        print("[sqlite probabilities] normalizing dtypes")
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    for column in ("yes_probability", "trade_count", "total_size", "last_trade_price"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["observed_trade"] = pd.to_numeric(out["observed_trade"], errors="coerce").fillna(0).astype(int)
    if show_progress:
        print("[sqlite probabilities] sorting rows")
    return out.sort_values(["market_id", "timestamp_utc"], kind="stable").reset_index(drop=True)


def _normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize market metadata types after loading them from SQLite."""

    out = frame.copy()
    for column in (
        "created_at",
        "end_date",
        "probability_start_utc",
        "probability_end_utc",
        "raw_trade_start_utc",
        "raw_trade_end_utc",
        "synced_at_utc",
    ):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    for column in ("market_id", "event_id"):
        if column in out.columns:
            out[column] = out[column].astype("string")
    numeric_columns = (
        "volume_num",
        "final_yes_probability",
        "trade_rows",
        "probability_rows",
    )
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ("active", "closed", "archived"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
    if "resolved" in out.columns:
        out["resolved"] = pd.to_numeric(out["resolved"], errors="coerce").fillna(0).astype(int).astype(bool)
    elif {"end_date", "synced_at_utc"}.issubset(out.columns):
        out["resolved"] = (
            out["end_date"].notna()
            & out["synced_at_utc"].notna()
            & (out["end_date"] <= out["synced_at_utc"])
        )

    platform_category = _normalize_category_series(
        out["kalshi_category"] if "kalshi_category" in out.columns else pd.Series(pd.NA, index=out.index, dtype="string")
    )
    out["platform_category"] = platform_category
    out["research_category"] = _normalize_category_series(
        out["research_category"] if "research_category" in out.columns else pd.Series(pd.NA, index=out.index, dtype="string")
    ).fillna(platform_category)
    out["market_id"] = out["market_id"].astype(str)
    return out.reset_index(drop=True)


def _normalize_category_series(series: pd.Series) -> pd.Series:
    """Normalize category-like values, keeping only meaningful labels."""

    normalized = series.astype("string").str.strip()
    invalid = normalized.isna() | normalized.eq("") | normalized.str.lower().isin(
        {"unknown", "unassigned", "none", "<na>", "nan"}
    )
    return normalized.mask(invalid, pd.NA)
