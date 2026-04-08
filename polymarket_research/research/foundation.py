"""Reusable builders for the NeurIPS-style research notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from benchmarks.benchmark_utils import (
    add_time_features,
    build_multi_horizon_terminal_dataset,
    build_repricing_dataset,
)
from benchmarks.covariate_utils import (
    asof_join_covariates,
    load_external_covariates,
    pivot_covariates_to_wide,
)
from polymarket_research.data.raw.dataset import RawPolymarketDataset
from polymarket_research.utils.text import build_family_id, parse_listish


@dataclass(frozen=True)
class MarketPanelBuilder:
    """Build terminal and repricing panels from the base Polymarket dataset."""

    domains: tuple[str, ...]
    terminal_horizons: tuple[int, ...]
    min_probability_rows: int = 288
    max_snapshot_staleness_hours: float = 12.0
    repricing_future_hours: int = 24
    repricing_lookback_hours: int = 24
    repricing_sample_every_hours: int = 12
    repricing_move_threshold: float = 0.15

    def prepare_markets(self, markets_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize market metadata and add a weak family identifier."""
        out = markets_df.copy()
        out["market_id"] = out["market_id"].astype(str)
        out["created_at"] = pd.to_datetime(out["created_at"], utc=True, errors="coerce")
        out["end_date"] = pd.to_datetime(out["end_date"], utc=True, errors="coerce")
        out["primary_domain"] = out["primary_domain"].fillna("unknown").astype(str)
        out["domain"] = out["primary_domain"]
        out = out.dropna(subset=["market_id", "created_at", "end_date", "final_yes_probability"]).reset_index(drop=True)
        out = out.loc[out["domain"].isin(self.domains)].copy()
        out["family_id"] = [
            build_family_id(question, domain, tags)
            for question, domain, tags in zip(out["question"], out["primary_domain"], out["tag_labels"], strict=False)
        ]
        return out

    def prepare_probabilities(self, dataset: PolymarketDataset, markets_df: pd.DataFrame) -> pd.DataFrame:
        """Filter probability history down to the selected market universe."""
        probabilities = dataset.probabilities.copy()
        probabilities["market_id"] = probabilities["market_id"].astype(str)
        probabilities["timestamp_utc"] = pd.to_datetime(probabilities["timestamp_utc"], utc=True, errors="coerce")
        probabilities = probabilities.loc[probabilities["market_id"].isin(markets_df["market_id"])]
        return probabilities.sort_values(["market_id", "timestamp_utc"], kind="stable").reset_index(drop=True)

    def build_terminal_panel(self, markets_df: pd.DataFrame, probabilities_df: pd.DataFrame) -> pd.DataFrame:
        """Build the terminal forecasting panel and append family/domain context."""
        terminal = build_multi_horizon_terminal_dataset(
            markets_df,
            probabilities_df,
            horizons_hours=self.terminal_horizons,
            max_snapshot_staleness_hours=self.max_snapshot_staleness_hours,
        )
        terminal = add_time_features(terminal)
        terminal = terminal.merge(markets_df[["market_id", "primary_domain", "family_id"]], on="market_id", how="left")
        context = self.build_family_context_features(
            terminal,
            probabilities_df,
            markets_df[["market_id", "family_id"]],
            time_col="cutoff_timestamp_utc",
            prob_col="market_price_baseline",
        )
        terminal = terminal.merge(context, on=["market_id", "cutoff_timestamp_utc"], how="left")
        return self.add_domain_dummies(terminal)

    def build_repricing_panel(
        self,
        markets_df: pd.DataFrame,
        probabilities_df: pd.DataFrame,
        *,
        shock_table: pd.DataFrame | None = None,
        shock_max_age: str | pd.Timedelta | None = None,
    ) -> pd.DataFrame:
        """Build the repricing panel and optionally merge external shocks."""
        repricing = build_repricing_dataset(
            markets_df,
            probabilities_df,
            future_horizon_hours=self.repricing_future_hours,
            lookback_hours=self.repricing_lookback_hours,
            sample_every_hours=self.repricing_sample_every_hours,
            move_threshold=self.repricing_move_threshold,
        )
        repricing = repricing.merge(
            markets_df[["market_id", "primary_domain", "question", "tag_labels", "family_id"]],
            on="market_id",
            how="left",
        )
        if shock_table is not None:
            repricing = asof_join_covariates(repricing, shock_table, base_time_col="timestamp_utc", max_age=shock_max_age)
        repricing = self.add_domain_dummies(repricing)
        return repricing

    @staticmethod
    def add_domain_dummies(df: pd.DataFrame, source_col: str = "primary_domain") -> pd.DataFrame:
        """Append one-hot domain columns while keeping the original frame intact."""
        dummies = pd.get_dummies(df[source_col], prefix="domain", dtype=float)
        return pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    @staticmethod
    def latest_snapshot_before(probabilities_df: pd.DataFrame, market_id: str, cutoff: pd.Timestamp):
        """Fetch the latest market snapshot at or before a cutoff timestamp."""
        panel = probabilities_df.loc[
            (probabilities_df["market_id"] == market_id) & (probabilities_df["timestamp_utc"] <= cutoff)
        ]
        if panel.empty:
            return None
        return panel.iloc[-1]

    def build_family_context_features(
        self,
        panel_df: pd.DataFrame,
        probabilities_df: pd.DataFrame,
        market_meta: pd.DataFrame,
        *,
        time_col: str,
        prob_col: str,
    ) -> pd.DataFrame:
        """Aggregate contemporaneous same-family probabilities into compact features."""
        family_map = market_meta.groupby("family_id")["market_id"].apply(list).to_dict()
        market_to_family = market_meta.set_index("market_id")["family_id"].to_dict()
        rows: list[dict[str, object]] = []

        for row in panel_df[["market_id", time_col, prob_col]].itertuples(index=False):
            cutoff = getattr(row, time_col)
            base_prob = getattr(row, prob_col)
            family_id = market_to_family.get(row.market_id)
            related_ids = [market_id for market_id in family_map.get(family_id, []) if market_id != row.market_id]
            related_probs = []
            for related_id in related_ids:
                snapshot = self.latest_snapshot_before(probabilities_df, related_id, cutoff)
                if snapshot is None:
                    continue
                related_probs.append(float(snapshot["yes_probability"]))
            rows.append(
                {
                    "market_id": row.market_id,
                    time_col: cutoff,
                    "family_related_count": float(len(related_probs)),
                    "family_prob_mean": float(np.mean(related_probs)) if related_probs else np.nan,
                    "family_prob_gap": float(np.max(related_probs) - np.min(related_probs)) if related_probs else np.nan,
                    "family_vs_market_gap": float(abs(np.mean(related_probs) - base_prob)) if related_probs else np.nan,
                }
            )
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class ExternalShockBuilder:
    """Build BTC/ETH external shock tables from saved covariates."""

    z_threshold: float = 2.0
    std_window: int = 288

    def build(self, path: str | Path) -> pd.DataFrame:
        """Convert BTC/ETH external series into returns, z-scores, and shock flags."""
        covariates = load_external_covariates(path)
        wide = pivot_covariates_to_wide(covariates, value_col="value").sort_values("timestamp_utc").reset_index(drop=True)
        out = wide[["timestamp_utc"]].copy()
        for series_id in ("btc_usd", "eth_usd"):
            series = pd.to_numeric(wide.get(series_id), errors="coerce")
            returns = series.pct_change()
            sigma = returns.rolling(self.std_window, min_periods=max(24, self.std_window // 6)).std()
            z_score = returns / sigma.replace(0.0, np.nan)
            out[f"{series_id}_ret"] = returns
            out[f"{series_id}_z"] = z_score
            out[f"{series_id}_shock"] = (z_score.abs() >= self.z_threshold).astype(float)
        out["any_external_shock"] = out[["btc_usd_shock", "eth_usd_shock"]].max(axis=1)
        out["btc_or_eth_shock"] = out["any_external_shock"]
        return out


@dataclass(frozen=True)
class RetrievedContextBuilder:
    """Retrieve related markets and summarize their contemporaneous state."""

    min_probability_rows: int = 288
    max_markets_per_domain: int = 90
    top_k: int = 5

    def prepare_market_universe(self, markets_df: pd.DataFrame, domains: tuple[str, ...]) -> pd.DataFrame:
        """Select a bounded retrieval universe from the full market table."""
        frames = []
        for domain in domains:
            frame = markets_df.loc[markets_df["domain"] == domain].copy()
            frame = frame.loc[frame["probability_rows"].fillna(0) >= self.min_probability_rows]
            if self.max_markets_per_domain is not None:
                frame = frame.sort_values("volume_num", ascending=False).head(self.max_markets_per_domain)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def build_neighbor_map(self, retrieval_markets: pd.DataFrame) -> tuple[dict[str, list[str]], pd.DataFrame, np.ndarray]:
        """Build a text-and-tag-based nearest-neighbor map over markets."""
        work = retrieval_markets[
            ["market_id", "question", "description", "tag_labels", "primary_domain", "created_at", "end_date", "family_id"]
        ].copy()
        work["text_blob"] = (
            work["question"].fillna("")
            + " "
            + work["description"].fillna("")
            + " "
            + work["tag_labels"].fillna("")
        )
        vectorizer = TfidfVectorizer(min_df=2, max_features=5000, ngram_range=(1, 2), stop_words="english")
        tfidf = vectorizer.fit_transform(work["text_blob"])
        text_cos = cosine_similarity(tfidf)

        neighbor_map: dict[str, list[str]] = {}
        detail_rows: list[dict[str, object]] = []
        for i, row in work.reset_index(drop=True).iterrows():
            scores = []
            for j, row2 in work.reset_index(drop=True).iterrows():
                if i == j:
                    continue
                tag_overlap = len(set(parse_listish(row["tag_labels"])) & set(parse_listish(row2["tag_labels"])))
                score = 0.75 * float(text_cos[i, j]) + 0.25 * float(tag_overlap > 0)
                scores.append((score, row2["market_id"], row2["question"]))
            scores.sort(reverse=True)
            top_neighbors = scores[: self.top_k]
            neighbor_map[row["market_id"]] = [market_id for _, market_id, _ in top_neighbors]
            if i < 5:
                for rank, (score, neighbor_id, neighbor_question) in enumerate(top_neighbors, start=1):
                    detail_rows.append(
                        {
                            "query_market_id": row["market_id"],
                            "query_question": row["question"],
                            "rank": rank,
                            "neighbor_market_id": neighbor_id,
                            "score": score,
                            "neighbor_question": neighbor_question,
                        }
                    )
        return neighbor_map, pd.DataFrame(detail_rows), text_cos

    def build_context_features(
        self,
        panel_df: pd.DataFrame,
        probabilities_df: pd.DataFrame,
        neighbor_map: dict[str, list[str]],
        *,
        time_col: str = "timestamp_utc",
    ) -> pd.DataFrame:
        """Summarize contemporaneous neighbor state into a compact feature block."""
        rows: list[dict[str, object]] = []
        for row in panel_df[["market_id", time_col]].itertuples(index=False):
            cutoff = getattr(row, time_col)
            neighbors = neighbor_map.get(row.market_id, [])
            probs = []
            trades = []
            for neighbor_id in neighbors:
                snapshot = MarketPanelBuilder.latest_snapshot_before(probabilities_df, neighbor_id, cutoff)
                if snapshot is None:
                    continue
                probs.append(float(snapshot["yes_probability"]))
                trades.append(float(snapshot.get("observed_trade", 0.0)))
            rows.append(
                {
                    "market_id": row.market_id,
                    time_col: cutoff,
                    "retrieved_count": float(len(probs)),
                    "retrieved_prob_mean": float(np.mean(probs)) if probs else np.nan,
                    "retrieved_prob_std": float(np.std(probs)) if probs else np.nan,
                    "retrieved_prob_gap": float(np.max(probs) - np.min(probs)) if probs else np.nan,
                    "retrieved_trade_share": float(np.mean(trades)) if trades else np.nan,
                }
            )
        return pd.DataFrame(rows)
