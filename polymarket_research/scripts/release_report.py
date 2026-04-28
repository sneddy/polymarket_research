"""Build source-level release reports for frozen benchmark artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from polymarket_research.benchmarks.io.paths import benchmark_release_dir, benchmark_release_report_dir
from polymarket_research.benchmarks.schemas.decisiveness import DecisivenessBenchmark
from polymarket_research.benchmarks.schemas.repricing import RepricingBenchmark
from polymarket_research.benchmarks.schemas.terminal import TerminalBenchmark
from polymarket_research.utils.data import open_sqlite_dataset


BenchmarkLike = TerminalBenchmark | DecisivenessBenchmark | RepricingBenchmark

_SHORT_HORIZON_UPDOWN_PATTERNS = ("-updown-5m-", "-updown-15m-", "-updown-4h-")
_KALSHI_ALLOWED_CATEGORIES = {
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
_KALSHI_DENIED_FREQUENCIES = {"fifteen_min", "hourly", "daily"}
_KALSHI_DENIED_TITLE_PATTERNS = (
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


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (str(table_name),),
    ).fetchone()
    return row is not None


def _count_table_rows(conn, table_name: str) -> int | None:
    if not _table_exists(conn, table_name):
        return None
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])


def _parse_list_text(value: Any) -> list[Any] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        try:
            import ast

            parsed = ast.literal_eval(text)
        except Exception:
            return None
        return parsed if isinstance(parsed, list) else None


def _parse_binary_prices(value: Any) -> list[float] | None:
    parsed = _parse_list_text(value)
    if not isinstance(parsed, list) or len(parsed) != 2:
        return None
    try:
        out = [float(item) for item in parsed]
    except Exception:
        return None
    return out


def _parse_binary_outcomes(value: Any) -> list[str] | None:
    parsed = _parse_list_text(value)
    if not isinstance(parsed, list) or len(parsed) != 2:
        return None
    outcomes = [str(item).strip() for item in parsed]
    if {item.lower() for item in outcomes} != {"yes", "no"}:
        return None
    return outcomes


def _resolved_outcome_from_prices(prices: list[float], outcomes: list[str]) -> str | None:
    if len(prices) != 2 or len(outcomes) != 2:
        return None
    if abs(sum(prices) - 1.0) > 1e-3:
        return None
    winner_idx = int(prices.index(max(prices)))
    if float(prices[winner_idx]) < 0.99:
        return None
    return outcomes[winner_idx]


def _is_short_horizon_updown(slug: Any, event_slug: Any) -> bool:
    slug_text = str(slug or "").lower()
    event_slug_text = str(event_slug or "").lower()
    return any(pattern in slug_text or pattern in event_slug_text for pattern in _SHORT_HORIZON_UPDOWN_PATTERNS)


def _polymarket_selection_funnel(conn) -> dict[str, Any]:
    if not _table_exists(conn, "market_universe"):
        return {
            "available": False,
            "stages": [],
            "notes": ["market_universe table not found; only downstream benchmark counts are available."],
        }

    universe = pd.read_sql_query(
        """
        SELECT
            market_id,
            market_slug,
            event_slug,
            outcomes,
            outcome_prices,
            clob_token_ids
        FROM market_universe
        """,
        conn,
    )
    resolved_mask: list[bool] = []
    clob_mask: list[bool] = []
    short_mask: list[bool] = []
    for row in universe.itertuples(index=False):
        outcomes = _parse_binary_outcomes(getattr(row, "outcomes", None))
        prices = _parse_binary_prices(getattr(row, "outcome_prices", None))
        resolved_mask.append(
            outcomes is not None
            and prices is not None
            and _resolved_outcome_from_prices(prices, outcomes) is not None
        )
        clob_value = getattr(row, "clob_token_ids", None)
        clob_mask.append(clob_value is not None and str(clob_value).strip() not in {"", "[]", "null", "None"})
        short_mask.append(_is_short_horizon_updown(getattr(row, "market_slug", None), getattr(row, "event_slug", None)))

    work = universe.copy()
    work["resolved_binary_candidate"] = resolved_mask
    work["has_clob_token_ids"] = clob_mask
    work["excluded_short_horizon_updown"] = short_mask
    core_candidate = work["resolved_binary_candidate"] & work["has_clob_token_ids"] & ~work["excluded_short_horizon_updown"]
    selected_rows = _count_table_rows(conn, "selected_markets")
    selected_selection_versions = None
    stages = [
        {"name": "market_universe", "rows": int(len(work))},
        {"name": "resolved_binary_candidates", "rows": int(work["resolved_binary_candidate"].sum())},
        {"name": "with_clob_token_ids", "rows": int((work["resolved_binary_candidate"] & work["has_clob_token_ids"]).sum())},
        {"name": "without_short_horizon_updown", "rows": int(core_candidate.sum())},
        {"name": "selected_markets_registry", "rows": selected_rows},
    ]
    return {
        "available": True,
        "stages": stages,
        "notes": [
            "The final delta from without_short_horizon_updown to selected_markets_registry includes remaining semantic/tag exclusions from the persisted Polymarket registry selection step."
        ],
        "selection_versions": selected_selection_versions,
    }


def _kalshi_selection_funnel(conn) -> dict[str, Any]:
    notes: list[str] = []
    out: dict[str, Any] = {"available": True, "series_stages": [], "market_stages": [], "notes": notes}

    if _table_exists(conn, "raw_series"):
        series_df = pd.read_sql_query(
            "SELECT series_ticker, title, subtitle, category, frequency FROM raw_series",
            conn,
        )
        text_blob = (
            series_df["title"].fillna("").astype(str)
            + " "
            + series_df["subtitle"].fillna("").astype(str)
        )
        frequency_allowed = ~series_df["frequency"].fillna("").astype(str).str.strip().str.lower().isin(_KALSHI_DENIED_FREQUENCIES)
        category_allowed = series_df["category"].fillna("").astype(str).isin(_KALSHI_ALLOWED_CATEGORIES)
        title_allowed = ~text_blob.map(
            lambda value: any(re.search(pattern, str(value), flags=re.IGNORECASE) for pattern in _KALSHI_DENIED_TITLE_PATTERNS)
        )
        out["series_stages"] = [
            {"name": "raw_series", "rows": int(len(series_df))},
            {"name": "allowed_frequency", "rows": int(frequency_allowed.sum())},
            {"name": "allowed_frequency_and_category", "rows": int((frequency_allowed & category_allowed).sum())},
            {"name": "allowed_frequency_category_and_title", "rows": int((frequency_allowed & category_allowed & title_allowed).sum())},
            {"name": "selected_series_registry", "rows": _count_table_rows(conn, "selected_series")},
        ]
        if _table_exists(conn, "selected_series"):
            version_rows = pd.read_sql_query(
                "SELECT DISTINCT selection_version FROM selected_series WHERE selection_version IS NOT NULL",
                conn,
            )
            out["selected_series_versions"] = sorted(version_rows["selection_version"].astype(str).tolist())
    else:
        notes.append("raw_series table not found; series-level funnel omitted.")

    if _table_exists(conn, "raw_markets"):
        markets_df = pd.read_sql_query(
            "SELECT market_type, volume_num FROM raw_markets",
            conn,
        )
        binary_mask = markets_df["market_type"].fillna("").astype(str).str.strip().str.lower().eq("binary")
        selected_markets_df = (
            pd.read_sql_query(
                """
                SELECT DISTINCT selection_reason, selection_version
                FROM selected_markets
                WHERE selection_reason IS NOT NULL OR selection_version IS NOT NULL
                """,
                conn,
            )
            if _table_exists(conn, "selected_markets")
            else pd.DataFrame(columns=["selection_reason", "selection_version"])
        )
        volume_threshold = None
        for reason in selected_markets_df["selection_reason"].dropna().astype(str):
            match = re.search(r"volume_num>=([0-9]+(?:\\.[0-9]+)?)", reason)
            if match:
                volume_threshold = float(match.group(1))
                break
        market_stages = [{"name": "raw_markets", "rows": int(len(markets_df))}, {"name": "binary_markets", "rows": int(binary_mask.sum())}]
        if volume_threshold is not None:
            volume_mask = markets_df["volume_num"].fillna(0).ge(float(volume_threshold))
            market_stages.append(
                {
                    "name": f"binary_and_volume_gte_{int(volume_threshold)}",
                    "rows": int((binary_mask & volume_mask).sum()),
                }
            )
        market_stages.append({"name": "selected_markets_registry", "rows": _count_table_rows(conn, "selected_markets")})
        out["market_stages"] = market_stages
        out["selected_markets_versions"] = sorted(
            selected_markets_df["selection_version"].dropna().astype(str).unique().tolist()
        )
    else:
        notes.append("raw_markets table not found; market-level funnel omitted.")

    out["available"] = bool(out["series_stages"] or out["market_stages"])
    return out


def _canonical_summary(canonical) -> dict[str, Any]:
    families = (
        canonical.markets["family_id"].dropna().astype(str).str.strip()
        if "family_id" in canonical.markets.columns
        else pd.Series(dtype="string")
    )
    families = families.loc[families != ""]
    categories = (
        canonical.markets["research_category"].dropna().astype(str).str.strip()
        if "research_category" in canonical.markets.columns
        else pd.Series(dtype="string")
    )
    categories = categories.loc[categories != ""]
    return {
        "markets": int(len(canonical.markets)),
        "probability_rows": int(len(canonical.probabilities)),
        "external_covariates_rows": 0 if canonical.external_covariates is None else int(len(canonical.external_covariates)),
        "download_status_rows": 0 if canonical.download_status is None else int(len(canonical.download_status)),
        "unique_families": int(families.nunique()) if not families.empty else 0,
        "research_category_counts": {
            str(key): int(value)
            for key, value in categories.value_counts(dropna=False).sort_index().to_dict().items()
        },
    }


def _benchmark_materialization_funnel(benchmark: BenchmarkLike) -> dict[str, Any]:
    return {
        "markets": int(benchmark.examples["market_id"].nunique()) if not benchmark.examples.empty else 0,
        "rows": int(len(benchmark.examples)),
        "market_timeseries_rows": int(len(benchmark.market_timeseries)),
    }


def _render_release_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {str(report['source']).title()} Benchmark Release Report",
        "",
        f"- Version: `{report['version']}`",
        f"- Generated at: `{report['generated_at_utc']}`",
    ]
    if report.get("db_path") is not None:
        lines.append(f"- SQLite: `{report['db_path']}`")
    lines.extend(
        [
            "",
            "## Canonical Summary",
            f"- Markets: {report['canonical_summary']['markets']}",
            f"- Probability rows: {report['canonical_summary']['probability_rows']}",
            f"- Unique families: {report['canonical_summary']['unique_families']}",
            "",
            "## Selection Funnel",
        ]
    )
    selection = report.get("selection_funnel", {})
    for key in ("stages", "series_stages", "market_stages"):
        for stage in selection.get(key, []) or []:
            lines.append(f"- {stage['name']}: {stage['rows']}")
    for note in selection.get("notes", []) or []:
        lines.append(f"- Note: {note}")
    lines.extend(["", "## Benchmarks"])
    for task, summary in report.get("benchmark_manifests", {}).items():
        lines.append(f"- {task}: rows={summary.get('rows', 0)} markets={summary.get('markets', 0)}")
    lines.append("")
    return "\n".join(lines)


def build_release_report(
    *,
    repo_root: str | Path,
    source: str,
    version: str,
    canonical,
    benchmarks: dict[str, BenchmarkLike],
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a source-level release report payload."""
    selection_funnel = {"available": False, "notes": ["No SQLite path provided; upstream selection funnel omitted."]}
    resolved_db_path = Path(db_path) if db_path is not None else None
    if resolved_db_path is not None and resolved_db_path.exists():
        with open_sqlite_dataset(resolved_db_path, source=str(source)) as conn:
            if str(source) == "kalshi":
                selection_funnel = _kalshi_selection_funnel(conn)
            else:
                selection_funnel = _polymarket_selection_funnel(conn)

    benchmark_manifests = {task: benchmark.manifest() for task, benchmark in benchmarks.items()}
    task_outputs = {
        task: {
            "bundle_dir": str(benchmark_release_dir(repo_root, source=source, task=task, version=version)),
            "manifest_file": str(benchmark_release_dir(repo_root, source=source, task=task, version=version) / "manifest.json"),
            "materialization": _benchmark_materialization_funnel(benchmark),
        }
        for task, benchmark in benchmarks.items()
    }
    return {
        "schema_version": 1,
        "source": str(source),
        "version": str(version),
        "generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "db_path": None if resolved_db_path is None else str(resolved_db_path),
        "canonical_summary": _canonical_summary(canonical),
        "selection_funnel": selection_funnel,
        "benchmark_manifests": benchmark_manifests,
        "task_outputs": task_outputs,
    }


def write_release_report(
    *,
    repo_root: str | Path,
    source: str,
    version: str,
    canonical,
    benchmarks: dict[str, BenchmarkLike],
    db_path: str | Path | None = None,
) -> dict[str, Path]:
    """Persist the source-level release report as JSON plus a compact Markdown summary."""
    report_dir = benchmark_release_report_dir(repo_root, source=source, version=version)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = build_release_report(
        repo_root=repo_root,
        source=source,
        version=version,
        canonical=canonical,
        benchmarks=benchmarks,
        db_path=db_path,
    )
    json_path = report_dir / "release_report.json"
    md_path = report_dir / "release_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_render_release_report_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
