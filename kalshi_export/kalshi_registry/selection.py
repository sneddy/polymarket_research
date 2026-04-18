from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from kalshi_registry.upsert import upsert_selected_markets


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = REPO_ROOT / "artefacts" / "kalshi_plots"

_SELECTED_MARKETS_COLUMNS = [
    "market_id",
    "source",
    "venue_market_id",
    "event_id",
    "venue_event_id",
    "series_ticker",
    "ticker",
    "event_ticker",
    "question",
    "description",
    "event_title",
    "kalshi_category",
    "primary_domain",
    "created_at",
    "open_time",
    "close_time",
    "settlement_ts",
    "end_date",
    "status",
    "market_type",
    "is_binary",
    "is_resolved",
    "is_active",
    "is_closed",
    "mutually_exclusive",
    "strike_type",
    "custom_strike_json",
    "volume_num",
    "volume_24h_num",
    "open_interest_num",
    "liquidity_dollars",
    "final_outcome",
    "final_yes_probability",
    "rules_primary",
    "rules_secondary",
    "selection_reason",
    "selection_version",
    "history_start_utc",
    "history_end_utc",
    "history_ready",
    "synced_at_utc",
]


def _save_plot(name: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / name
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    logger.info("Kalshi market_selection plot saved | path=%s", output_path)
    return output_path


def _first_non_empty_text(*values: object) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def plot_raw_market_volumes(raw_markets_df: pd.DataFrame) -> Path | None:
    if raw_markets_df.empty:
        logger.info("Kalshi market_selection volume plot skipped | reason=empty_raw_markets")
        return None

    volume = raw_markets_df["volume_num"].fillna(0)
    volume_buckets = pd.DataFrame(
        {
            "bucket": [
                "0",
                "0-10k",
                "10k-20k",
                "20k-50k",
                "50k-100k",
                "100k-500k",
                ">500k",
            ],
            "rows": [
                int(volume.le(0).sum()),
                int(((volume > 0) & (volume < 10_000)).sum()),
                int(((volume >= 10_000) & (volume < 20_000)).sum()),
                int(((volume >= 20_000) & (volume < 50_000)).sum()),
                int(((volume >= 50_000) & (volume < 100_000)).sum()),
                int(((volume >= 100_000) & (volume < 500_000)).sum()),
                int(volume.ge(500_000).sum()),
            ],
        }
    )
    total_rows = max(int(volume_buckets["rows"].sum()), 1)
    volume_buckets["pct_rows"] = 100.0 * volume_buckets["rows"] / total_rows

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    bars = axes[0].bar(volume_buckets["bucket"], volume_buckets["rows"], color="#4c956c")
    axes[0].set_title("Volume buckets")
    axes[0].set_ylabel("market rows")
    axes[0].tick_params(axis="x", rotation=30)

    for bar, pct in zip(bars, volume_buckets["pct_rows"]):
        height = bar.get_height()
        axes[0].annotate(
            f"{pct:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    lt_20k = int(((volume >= 0) & (volume < 20_000)).sum())
    gte_20k = int((volume >= 20_000).sum())
    tot = max(lt_20k + gte_20k, 1)
    simple_buckets = ["<20k", "≥20k"]
    simple_counts = [lt_20k, gte_20k]
    simple_pcts = [100.0 * n / tot for n in simple_counts]

    bars2 = axes[1].bar(simple_buckets, simple_counts, color=["#f4a259", "#588157"])
    for bar, pct in zip(bars2, simple_pcts):
        height = bar.get_height()
        axes[1].annotate(
            f"{pct:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    legend_labels = [f"{bucket}: {count:,}" for bucket, count in zip(simple_buckets, simple_counts)]
    axes[1].legend(bars2, legend_labels, title="Volume range (count)", loc="upper right")
    axes[1].set_title("Volume split (<20k vs ≥20k)")
    axes[1].set_ylabel("market rows")
    axes[1].set_ylim(0, max(simple_counts) * 1.2 if simple_counts else 1)
    axes[1].tick_params(axis="x", rotation=0)

    output_path = _save_plot("kalshi_raw_markets_volume_buckets.png")
    plt.close(fig)
    return output_path


def rebuild_selected_markets(
    conn: sqlite3.Connection,
    *,
    min_volume: float = 20_000.0,
    force_remove: bool = False,
    selection_version: str = "v2_binary_and_min_volume",
) -> dict[str, int]:
    logger.info(
        "Kalshi selected_markets rebuild started | min_volume=%s force_remove=%s selection_version=%s",
        min_volume,
        force_remove,
        selection_version,
    )
    raw_markets_df = pd.read_sql_query(
        """
        SELECT
            market_id,
            source,
            venue_market_id,
            event_id,
            venue_event_id,
            series_ticker,
            ticker,
            event_ticker,
            question,
            description,
            created_at,
            open_time,
            close_time,
            settlement_ts,
            end_date,
            status,
            market_type,
            is_binary,
            is_resolved,
            is_active,
            is_closed,
            strike_type,
            custom_strike_json,
            volume_num,
            volume_24h_num,
            open_interest_num,
            liquidity_dollars,
            final_outcome,
            final_yes_probability,
            rules_primary,
            rules_secondary
        FROM raw_markets
        """,
        conn,
    )
    plot_path = plot_raw_market_volumes(raw_markets_df)
    if raw_markets_df.empty:
        if force_remove:
            conn.execute("DELETE FROM selected_markets")
        logger.info("Kalshi selected_markets rebuild finished | selected_rows=0")
        return {"selected_rows": 0, "plot_path": str(plot_path) if plot_path is not None else None}

    market_type_norm = raw_markets_df["market_type"].fillna("").astype(str).str.strip().str.lower()
    source_df = raw_markets_df.loc[
        market_type_norm.eq("binary") & raw_markets_df["volume_num"].fillna(0).ge(float(min_volume))
    ].copy()
    if source_df.empty:
        if force_remove:
            conn.execute("DELETE FROM selected_markets")
        logger.info("Kalshi selected_markets rebuild finished | selected_rows=0")
        return {"selected_rows": 0, "plot_path": str(plot_path) if plot_path is not None else None}

    selected_df = source_df.copy()
    if "series_ticker" not in selected_df.columns:
        selected_df["series_ticker"] = None
    selected_df["event_title"] = None
    selected_df["kalshi_category"] = None
    selected_df["primary_domain"] = None
    selected_df["mutually_exclusive"] = None
    selected_df["selection_reason"] = f"market_type==binary and volume_num>={float(min_volume):.0f}"
    selected_df["selection_version"] = selection_version
    selected_df["history_start_utc"] = [
        _first_non_empty_text(open_time, created_at)
        for open_time, created_at in zip(selected_df["open_time"], selected_df["created_at"], strict=False)
    ]
    selected_df["history_end_utc"] = [
        _first_non_empty_text(settlement_ts, close_time, end_date)
        for settlement_ts, close_time, end_date in zip(
            selected_df["settlement_ts"],
            selected_df["close_time"],
            selected_df["end_date"],
            strict=False,
        )
    ]
    selected_df["history_ready"] = (
        selected_df["ticker"].fillna("").astype(str).str.strip().ne("")
        & selected_df["series_ticker"].fillna("").astype(str).str.strip().ne("")
        & selected_df["history_start_utc"].fillna("").astype(str).str.strip().ne("")
        & selected_df["history_end_utc"].fillna("").astype(str).str.strip().ne("")
    ).astype(int)
    selected_df["synced_at_utc"] = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%dT%H:%M:%SZ")
    for column in _SELECTED_MARKETS_COLUMNS:
        if column not in selected_df.columns:
            selected_df[column] = None
    selected_df = selected_df[_SELECTED_MARKETS_COLUMNS].copy()

    if force_remove:
        with conn:
            conn.execute("DELETE FROM selected_markets")
    written = upsert_selected_markets(conn, selected_df)
    logger.info("Kalshi selected_markets rebuild finished | selected_rows=%s", written)
    return {"selected_rows": int(written), "plot_path": str(plot_path) if plot_path is not None else None}
