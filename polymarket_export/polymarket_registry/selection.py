"""Selection helpers for turning market metadata into a filtered registry."""

from __future__ import annotations

import ast
import logging
from typing import Any

import numpy as np
import pandas as pd

from clients.gamma_client import GammaClient
from configs.resolved_dataset_domain_config import DOMAIN_KEYWORD_HINTS
from configs.resolved_dataset_domain_config import DOMAIN_PRIORITY
from configs.resolved_dataset_domain_config import DOMAIN_TAG_MAP
from polymarket_registry.block_filters import apply_default_block_filters
from polymarket_registry.filters import is_short_horizon_updown_series
from polymarket_registry.upsert import extract_primary_event_fields


logger = logging.getLogger(__name__)


def to_bool(series: pd.Series) -> pd.Series:
    """Normalize truthy string-like values into a boolean series."""
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


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


def resolved_outcome_from_prices(
    prices: list[float],
    outcomes: list[str],
) -> tuple[str, float] | tuple[None, None]:
    """Infer a resolved binary outcome from near-extreme outcome prices."""
    if len(prices) != 2 or len(outcomes) != 2:
        return None, None
    if abs(sum(prices) - 1.0) > 1e-3:
        return None, None
    winner_idx = int(np.argmax(prices))
    winner_prob = float(prices[winner_idx])
    if winner_prob < 0.99:
        return None, None
    return outcomes[winner_idx], winner_prob


def filter_low_volume(markets_df: pd.DataFrame, *, min_resolved_volume: float) -> pd.DataFrame:
    """Filter the market universe down to high-volume resolved-like candidates."""
    work = markets_df.copy()
    work["closed_bool"] = to_bool(work["closed"]) if "closed" in work.columns else False
    work["active_bool"] = to_bool(work["active"]) if "active" in work.columns else False
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
    work["parsed_prices"] = work.get("outcome_prices").map(parse_binary_prices)
    work["parsed_outcomes"] = work.get("outcomes").map(parse_binary_outcomes)

    final_outcomes: list[str | None] = []
    final_yes_probabilities: list[float | None] = []
    resolved_like: list[bool] = []
    for prices, outcomes in zip(work["parsed_prices"], work["parsed_outcomes"], strict=False):
        if prices is None or outcomes is None:
            final_outcomes.append(None)
            final_yes_probabilities.append(None)
            resolved_like.append(False)
            continue
        winner, _ = resolved_outcome_from_prices(prices, outcomes)
        final_outcomes.append(winner)
        final_yes_probabilities.append(float(prices[outcomes.index("Yes")]) if winner is not None else None)
        resolved_like.append(winner is not None)

    work["final_outcome"] = final_outcomes
    work["final_yes_probability"] = final_yes_probabilities
    work["resolved_like"] = resolved_like
    work["has_clob_token_ids"] = work.get("clob_token_ids").notna() if "clob_token_ids" in work.columns else False
    if "events" in work.columns:
        event_fields = work["events"].map(extract_primary_event_fields).apply(pd.Series)
        for column in ("event_id", "event_slug", "event_title", "event_series_slug"):
            work[column] = event_fields[column]
    else:
        work["event_id"] = None
        work["event_slug"] = None
        work["event_title"] = None
        work["event_series_slug"] = None

    work["excluded_short_horizon_updown"] = [
        is_short_horizon_updown_series(slug=slug, event_slug=event_slug)
        for slug, event_slug in zip(work.get("slug"), work.get("event_slug"), strict=False)
    ]

    mask = (
        work["resolved_like"]
        & work["has_clob_token_ids"]
        & (work["volume_num_norm"] >= float(min_resolved_volume))
        & ~work["excluded_short_horizon_updown"]
    )

    cols = [
        "id",
        "condition_id",
        "slug",
        "event_id",
        "event_slug",
        "event_title",
        "event_series_slug",
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


def filter_market_universe(
    markets_df: pd.DataFrame,
    *,
    min_resolved_volume: float,
    apply_exclusion_blocks: bool = True,
) -> pd.DataFrame:
    """Filter a locally stored market_universe table into selection candidates."""
    work = markets_df.copy()
    work["id"] = work.get("market_id")
    work["slug"] = work.get("market_slug")
    work["closed_bool"] = to_bool(work["closed"]) if "closed" in work.columns else False
    work["active_bool"] = False
    work["volume_num_norm"] = pd.to_numeric(work.get("volume_num"), errors="coerce").fillna(0.0)
    work["liquidity_num_norm"] = pd.to_numeric(work.get("liquidity_num"), errors="coerce")
    work["created_at"] = pd.to_datetime(work.get("created_at"), utc=True, errors="coerce")
    work["end_date"] = pd.to_datetime(work.get("end_date"), utc=True, errors="coerce")
    work["parsed_prices"] = work.get("outcome_prices").map(parse_binary_prices)
    work["parsed_outcomes"] = work.get("outcomes").map(parse_binary_outcomes)

    final_outcomes: list[str | None] = []
    final_yes_probabilities: list[float | None] = []
    for prices, outcomes in zip(work["parsed_prices"], work["parsed_outcomes"], strict=False):
        if prices is None or outcomes is None:
            final_outcomes.append(None)
            final_yes_probabilities.append(None)
            continue
        winner, _ = resolved_outcome_from_prices(prices, outcomes)
        final_outcomes.append(winner)
        final_yes_probabilities.append(float(prices[outcomes.index("Yes")]) if winner is not None else None)

    work["final_outcome"] = final_outcomes
    work["final_yes_probability"] = final_yes_probabilities
    work["excluded_short_horizon_updown"] = [
        is_short_horizon_updown_series(slug=slug, event_slug=event_slug)
        for slug, event_slug in zip(work.get("slug"), work.get("event_slug"), strict=False)
    ]
    if apply_exclusion_blocks:
        work = apply_default_block_filters(work, min_volume=float(min_resolved_volume))
        keep_semantic_mask = work["category"].isna()
    else:
        keep_semantic_mask = pd.Series(True, index=work.index)

    mask = (
        work["final_outcome"].notna()
        & (work["volume_num_norm"] >= float(min_resolved_volume))
        & ~work["excluded_short_horizon_updown"]
        & keep_semantic_mask
    )

    cols = [
        "id",
        "condition_id",
        "slug",
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
        "closed_time",
        "uma_resolution_status",
        "neg_risk",
        "neg_risk_market_id",
        "group_item_title",
        "clob_token_ids",
        "outcomes",
        "outcome_prices",
        "final_outcome",
        "final_yes_probability",
    ]
    available_cols = [c for c in cols if c in work.columns]
    return work.loc[mask, available_cols].sort_values(
        ["volume_num_norm", "created_at"],
        ascending=[False, False],
    ).reset_index(drop=True)


def normalize_tag_labels(payload: Any) -> list[str]:
    """Normalize a Gamma tag payload into a sorted label list."""
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
    tag_labels: list[str],
    *,
    question: str | None = None,
    slug: str | None = None,
) -> tuple[list[str], list[str], str | None]:
    """Map normalized market tags and text hints onto research domains."""
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
    """Attach tag-derived domain labels to candidate markets."""
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
