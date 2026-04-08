"""Pipeline entry points for building reusable research panels from local Polymarket data."""

from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd

from benchmarks.benchmark_utils import (
    add_time_features,
    build_multi_horizon_terminal_dataset,
    build_repricing_dataset,
)
from benchmarks.covariate_utils import asof_join_covariates, load_external_covariates, pivot_covariates_to_wide
from polymarket_research.data.bundle import DataBundle
from polymarket_research.data.config import DataPaths, ExternalShockConfig, MarketSelectionConfig, PanelBuildConfig
from benchmarks.legacy.repository import ResolvedMarketRepository


def _parse_listish(value: object) -> list[str]:
    """Parse stored tag metadata into a clean list of strings."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass

    if "|" in text:
        parts = text.split("|")
    elif "," in text:
        parts = text.split(",")
    else:
        parts = [text]
    return [part.strip() for part in parts if part.strip()]


def _normalize_text(value: object) -> str:
    """Normalize a text field so weak lexical grouping becomes more stable."""

    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_family_id(question: str, domain: str, tags: object) -> str:
    """Create a weak family identifier for semantically related markets."""

    norm_question = _normalize_text(question)
    norm_tags = [_normalize_text(tag) for tag in _parse_listish(tags)]
    tokens = [token for token in norm_question.split() if token not in {"will", "the", "a", "an", "be", "is", "are", "to", "of", "by", "in"}]
    question_key = " ".join(tokens[:6]) if tokens else norm_question[:48]
    tag_key = "|".join(sorted(norm_tags[:3]))
    return f"{domain}::{tag_key}::{question_key}".strip(":")


def _latest_context_snapshot(probabilities_df: pd.DataFrame, market_id: str, cutoff: pd.Timestamp) -> pd.Series | None:
    """Return the latest probability row for a market before the supplied cutoff time."""

    history = probabilities_df.loc[
        (probabilities_df["market_id"] == market_id) & (probabilities_df["timestamp_utc"] <= cutoff)
    ]
    if history.empty:
        return None
    return history.iloc[-1]


def _build_family_context_features(dataset: pd.DataFrame, probabilities_df: pd.DataFrame, market_meta: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sibling market probabilities into simple family-level context features."""

    family_map = market_meta.groupby("family_id")["market_id"].apply(list).to_dict()
    family_by_market = market_meta.set_index("market_id")["family_id"].to_dict()

    rows: list[dict[str, object]] = []
    for row in dataset[["market_id", "cutoff_timestamp_utc", "market_price_baseline"]].itertuples(index=False):
        family_id = family_by_market.get(row.market_id)
        related_ids = [market_id for market_id in family_map.get(family_id, []) if market_id != row.market_id]
        related_probs: list[float] = []
        for related_id in related_ids:
            snapshot = _latest_context_snapshot(probabilities_df, related_id, row.cutoff_timestamp_utc)
            if snapshot is None:
                continue
            related_probs.append(float(snapshot["yes_probability"]))

        rows.append(
            {
                "market_id": row.market_id,
                "cutoff_timestamp_utc": row.cutoff_timestamp_utc,
                "family_related_count": float(len(related_probs)),
                "family_prob_mean": float(np.mean(related_probs)) if related_probs else np.nan,
                "family_prob_gap": float(np.max(related_probs) - np.min(related_probs)) if related_probs else np.nan,
                "family_vs_market_gap": float(abs(np.mean(related_probs) - row.market_price_baseline)) if related_probs else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _build_shock_table(paths: DataPaths, config: ExternalShockConfig) -> pd.DataFrame:
    """Convert external covariates into returns, z-scores, and binary shock indicators."""

    covariates = load_external_covariates(paths.external_covariates_path)
    wide = pivot_covariates_to_wide(covariates, value_col="value").sort_values("timestamp_utc").reset_index(drop=True)

    out = wide[["timestamp_utc"]].copy()
    value_columns = [column for column in wide.columns if column != "timestamp_utc"]
    for column in value_columns:
        series = pd.to_numeric(wide[column], errors="coerce")
        returns = series.pct_change()
        sigma = returns.rolling(config.std_window, min_periods=max(24, config.std_window // 6)).std()
        z_score = returns / sigma.replace(0.0, np.nan)
        out[f"{column}_ret"] = returns
        out[f"{column}_z"] = z_score
        out[f"{column}_shock"] = (z_score.abs() >= config.z_threshold).astype(float)

    shock_columns = [column for column in out.columns if column.endswith("_shock")]
    out["any_external_shock"] = out[shock_columns].max(axis=1)
    return out


class PolymarketDatasetBuilder:
    """Build a reusable phase-one research bundle from the local Polymarket database."""

    def __init__(
        self,
        paths: DataPaths | None = None,
        selection: MarketSelectionConfig | None = None,
        panels: PanelBuildConfig | None = None,
        shocks: ExternalShockConfig | None = None,
    ) -> None:
        """Initialize the builder with explicit config objects for every pipeline stage."""

        self.paths = paths or DataPaths()
        self.selection = selection or MarketSelectionConfig()
        self.panels = panels or PanelBuildConfig()
        self.shocks = shocks or ExternalShockConfig()
        self.repository = ResolvedMarketRepository(self.paths, self.selection)

    def load_markets(self) -> pd.DataFrame:
        """Load eligible markets and attach weak family identifiers used by downstream context features."""

        markets = self.repository.load_markets().copy()
        markets["family_id"] = [
            _build_family_id(question, domain, tags)
            for question, domain, tags in zip(
                markets["question"],
                markets["primary_domain"],
                markets["tag_labels"],
                strict=False,
            )
        ]
        return markets

    def load_probabilities(self, markets: pd.DataFrame) -> pd.DataFrame:
        """Load probability history for the provided market table."""

        return self.repository.load_probabilities(markets["market_id"].tolist())

    def build_terminal_panel(self, markets: pd.DataFrame, probabilities: pd.DataFrame) -> pd.DataFrame:
        """Build the multi-horizon terminal dataset and enrich it with time and family context features."""

        terminal = build_multi_horizon_terminal_dataset(
            markets,
            probabilities,
            horizons_hours=self.panels.terminal_horizons_hours,
            max_snapshot_staleness_hours=self.panels.max_snapshot_staleness_hours,
        )
        terminal = add_time_features(terminal)
        terminal = terminal.merge(markets[["market_id", "primary_domain", "family_id"]], on="market_id", how="left")
        terminal = terminal.merge(
            _build_family_context_features(terminal, probabilities, markets[["market_id", "family_id"]]),
            on=["market_id", "cutoff_timestamp_utc"],
            how="left",
        )
        domain_dummies = pd.get_dummies(terminal["primary_domain"], prefix="domain", dtype=float)
        return pd.concat([terminal, domain_dummies], axis=1)

    def build_repricing_panel(self, markets: pd.DataFrame, probabilities: pd.DataFrame) -> pd.DataFrame:
        """Build the repricing dataset and join external BTC/ETH shock features."""

        repricing = build_repricing_dataset(
            markets,
            probabilities,
            future_horizon_hours=self.panels.repricing_future_horizon_hours,
            lookback_hours=self.panels.repricing_lookback_hours,
            sample_every_hours=self.panels.repricing_sample_every_hours,
            move_threshold=self.panels.repricing_move_threshold,
        )
        repricing = repricing.merge(
            markets[["market_id", "primary_domain", "question", "description", "tag_labels"]],
            on="market_id",
            how="left",
        )
        domain_dummies = pd.get_dummies(repricing["primary_domain"], prefix="domain", dtype=float)
        repricing = pd.concat([repricing, domain_dummies], axis=1)

        shock_table = _build_shock_table(self.paths, self.shocks)
        repricing = asof_join_covariates(
            repricing,
            shock_table,
            base_time_col="timestamp_utc",
            covariate_time_col="timestamp_utc",
            max_age=self.shocks.join_max_age,
        )
        repricing["btc_or_eth_shock"] = repricing[["btc_usd_shock", "eth_usd_shock"]].max(axis=1)
        return repricing

    def build_shock_table(self) -> pd.DataFrame:
        """Build the standalone external shock table without constructing full panels."""

        return _build_shock_table(self.paths, self.shocks)

    def build(self) -> DataBundle:
        """Build the complete phase-one research bundle in one call."""

        markets = self.load_markets()
        probabilities = self.load_probabilities(markets)
        terminal = self.build_terminal_panel(markets, probabilities)
        repricing = self.build_repricing_panel(markets, probabilities)
        shock_table = self.build_shock_table()
        return DataBundle(
            markets=markets,
            probabilities=probabilities,
            terminal=terminal,
            repricing=repricing,
            shock_table=shock_table,
        )
