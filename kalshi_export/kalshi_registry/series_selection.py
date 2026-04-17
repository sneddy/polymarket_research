from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd

from kalshi_registry.upsert import upsert_selected_series


logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = REPO_ROOT / "artefacts" / "kalshi_plots"

ALLOWED_CATEGORIES = {
    "Entertainment",
    "Elections",
    "Politics",
    "Economics",
    "Companies",
    "Financials",
    "Science and Technology",
    "World",
    "Health",
    "Social",
}

DENIED_FREQUENCIES = {
    "fifteen_min",
    "hourly",
    "daily",
}

DENIED_TITLE_PATTERNS = (
    r"\bdaily\b",
    r"\bprice range\b",
    r"\btemperature\b",
    r"\bwind\b",
    r"\brain\b",
    r"\bsnow\b",
    r"\bup or down\b",
    r"\bover/?under\b",
    r"\babove/?below\b",
    r"\bweekly range\b",
    r"\bweekly price\b",
    r"\bday after election\b",
    r"\bweekly yield\b",
    r"\bdaily case average\b",
    r"\bfavorability\b",
    r"\bapproval rating\b",
    r"\bconsumer confidence\b",
    r"\bbusiness confidence\b",
    r"\bprice peak\b",
    r"\brate high\b",
    r"\brate low\b",
    r"\byearly low\b",
    r"\byearly high\b",
    r"\byearly range\b",
    r"\byoy\b",
    r"\bmom\b",
    r"\bhousing starts\b",
    r"\brotten tomatoes score\b",
    r"\brt score\b",
    r"\bmetacritic score(?:s)?\b",
    r"\b#1 album\b",
    r"\b#1 song\b",
    r"\bbillboard peak\b",
    r"\btop song\b",
    r"\btop movie\b",
    r"\btv ranking\b",
    r"\bhours watched\b",
    r"\btotal stream count\b",
    r"\btotal album streams\b",
    r"\balbum streams\b",
    r"\btotal streams\b",
    r"\bstreams this week\b",
    r"\bdaily .*spotify\b",
    r"\bmost streamed\b",
    r"\btop ten spots\b",
    r"\bstraight weeks\b",
    r"\bactive subs\b",
    r"\bbillion streams\b",
    r"\bhow many streams\b",
    r"\bspotify usa chart\b",
    r"\bstockx\b",
    r"\baverage sales price\b",
    r"\btwitch followers\b",
)

_SELECTED_SERIES_COLUMNS = [
    "series_ticker",
    "title",
    "subtitle",
    "category",
    "tags_json",
    "frequency",
    "status",
    "selection_reason",
    "selection_version",
    "synced_at_utc",
]


def _matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _save_plot(name: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / name
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    logger.info("Kalshi series_selection plot saved | path=%s", output_path)
    return output_path


def plot_series_categories_before_after(series_df: pd.DataFrame, selected_df: pd.DataFrame) -> Path | None:
    if series_df.empty:
        logger.info("Kalshi series_selection category plot skipped | reason=empty_raw_series")
        return None

    raw_counts = (
        series_df.assign(category_norm=series_df["category"].fillna("(missing)").astype(str).str.strip().replace("", "(missing)"))
        .groupby("category_norm", dropna=False)
        .size()
        .reset_index(name="raw_series_rows")
    )
    selected_counts = (
        selected_df.assign(
            category_norm=selected_df["category"].fillna("(missing)").astype(str).str.strip().replace("", "(missing)")
        )
        .groupby("category_norm", dropna=False)
        .size()
        .reset_index(name="selected_series_rows")
        if not selected_df.empty
        else pd.DataFrame(columns=["category_norm", "selected_series_rows"])
    )
    category_compare = raw_counts.merge(selected_counts, on="category_norm", how="left")
    category_compare["selected_series_rows"] = category_compare["selected_series_rows"].fillna(0).astype(int)
    category_compare = category_compare.sort_values(["raw_series_rows", "category_norm"], ascending=[False, True], kind="stable")
    plot_categories = category_compare.head(15).copy()
    if plot_categories.empty:
        logger.info("Kalshi series_selection category plot skipped | reason=no_categories")
        return None

    x = list(range(len(plot_categories)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(15, 6))
    bars1 = ax.bar(
        [idx - width / 2 for idx in x],
        plot_categories["raw_series_rows"],
        width,
        label="before selection",
        color="#9aa5b1",
    )
    bars2 = ax.bar(
        [idx + width / 2 for idx in x],
        plot_categories["selected_series_rows"],
        width,
        label="after selection",
        color="#2f5d8a",
    )
    ax.set_title("Series categories before and after selection")
    ax.set_ylabel("series rows")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_categories["category_norm"], rotation=35, ha="right")
    ax.legend()

    for bars in (bars1, bars2):
        for bar in bars:
            height = int(bar.get_height())
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    output_path = _save_plot("kalshi_series_categories_before_after.png")
    plt.close(fig)
    return output_path


def rebuild_selected_series(
    conn: sqlite3.Connection,
    *,
    force_remove: bool = False,
    selection_version: str = "v2_drop_short_frequencies_then_allow_categories_and_deny_short_term",
) -> dict[str, int]:
    logger.info(
        "Kalshi selected_series rebuild started | force_remove=%s selection_version=%s",
        force_remove,
        selection_version,
    )
    series_df = pd.read_sql_query("SELECT * FROM raw_series", conn)
    if series_df.empty:
        if force_remove:
            conn.execute("DELETE FROM selected_series")
        logger.info("Kalshi selected_series rebuild finished | selected_rows=0")
        return {"selected_rows": 0}

    keep_rows: list[dict[str, object]] = []
    for row in series_df.to_dict(orient="records"):
        category = str(row.get("category") or "").strip()
        frequency = str(row.get("frequency") or "").strip().lower()
        title = str(row.get("title") or "")
        subtitle = str(row.get("subtitle") or "")
        text_blob = " ".join([title, subtitle])

        if frequency in DENIED_FREQUENCIES:
            continue
        if category not in ALLOWED_CATEGORIES:
            continue
        if _matches_any(DENIED_TITLE_PATTERNS, text_blob):
            continue

        keep_rows.append(
            {
                "series_ticker": row.get("series_ticker"),
                "title": row.get("title"),
                "subtitle": row.get("subtitle"),
                "category": row.get("category"),
                "tags_json": row.get("tags_json"),
                "frequency": row.get("frequency"),
                "status": row.get("status"),
                "selection_reason": json.dumps(
                    {
                        "frequency_filter": "passed",
                        "allowed_category": category,
                        "title_filter": "passed",
                    },
                    sort_keys=True,
                ),
                "selection_version": selection_version,
                "synced_at_utc": pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    selected_df = pd.DataFrame(keep_rows)
    for column in _SELECTED_SERIES_COLUMNS:
        if column not in selected_df.columns:
            selected_df[column] = None
    selected_df = selected_df[_SELECTED_SERIES_COLUMNS].copy()
    plot_path = plot_series_categories_before_after(series_df, selected_df)
    if force_remove:
        with conn:
            conn.execute("DELETE FROM selected_series")
    written = upsert_selected_series(conn, selected_df)
    logger.info("Kalshi selected_series rebuild finished | selected_rows=%s plot_path=%s", written, plot_path)
    return {"selected_rows": int(written), "plot_path": str(plot_path) if plot_path is not None else None}
