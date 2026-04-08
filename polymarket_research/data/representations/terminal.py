"""Terminal-outcome market panels derived from canonical market state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from polymarket_research.data.canonical.dataset import CanonicalDataset
from polymarket_research.data.representations.common import (
    RepresentationFrame,
    add_time_progress_features,
    binary_log_loss,
    safe_float,
    safe_max,
    safe_mean,
    safe_std,
)
from polymarket_research.data.representations.context import FamilyContextBuilder


def extract_snapshot_features(
    market_panel: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    history_hours: Sequence[int] = (24, 168),
    max_snapshot_staleness_hours: float | None = None,
) -> dict[str, float] | None:
    """Extract local snapshot and lookback features before a cutoff time."""
    history = market_panel.loc[market_panel["timestamp_utc"] <= cutoff].copy()
    if history.empty:
        return None

    current = history.iloc[-1]
    current_prob = float(current["yes_probability"])
    current_ts = pd.Timestamp(current["timestamp_utc"])
    staleness_hours = float((cutoff - current_ts).total_seconds() / 3600.0)
    if max_snapshot_staleness_hours is not None and staleness_hours > float(max_snapshot_staleness_hours):
        return None

    features: dict[str, float] = {
        "cutoff_timestamp_utc": current_ts,
        "snapshot_staleness_hours": staleness_hours,
        "current_yes_probability": current_prob,
        "confidence_margin": abs(current_prob - 0.5),
        "observed_trade_now": float(current["observed_trade"]),
        "trade_count_now": safe_float(current["trade_count"]),
        "total_size_now": safe_float(current["total_size"]),
        "last_trade_price_now": safe_float(current["last_trade_price"]),
    }

    for hours in history_hours:
        window_start = cutoff - pd.Timedelta(hours=int(hours))
        window = history.loc[history["timestamp_utc"] > window_start]
        if window.empty:
            continue
        diff = window["yes_probability"].diff().dropna().to_numpy(dtype=float)
        prefix = f"lookback_{int(hours)}h"
        features[f"{prefix}_rows"] = float(len(window))
        features[f"{prefix}_observed_trade_share"] = float(window["observed_trade"].mean())
        features[f"{prefix}_trade_count_sum"] = float(window["trade_count"].sum())
        features[f"{prefix}_total_size_sum"] = float(window["total_size"].sum())
        features[f"{prefix}_yes_probability_change"] = float(window["yes_probability"].iloc[-1] - window["yes_probability"].iloc[0])
        features[f"{prefix}_volatility"] = safe_std(diff)
        features[f"{prefix}_abs_move_mean"] = safe_mean(np.abs(diff))
        features[f"{prefix}_abs_move_max"] = safe_max(np.abs(diff))

    return features


@dataclass
class TerminalPanelBuilder:
    """Build terminal-outcome snapshot panels from canonical market trajectories."""

    canonical: CanonicalDataset
    horizons_hours: tuple[int, ...] = (24, 72, 168)
    max_snapshot_staleness_hours: float | None = 12.0
    include_family_context: bool = True

    def build(self) -> RepresentationFrame:
        """Build terminal snapshots across configured horizons."""
        markets_df = self.canonical.markets
        probabilities_df = self.canonical.probabilities

        grouped = {
            market_id: frame.reset_index(drop=True)
            for market_id, frame in probabilities_df.groupby("market_id", sort=False)
        }

        rows: list[dict[str, object]] = []
        for market in markets_df.itertuples(index=False):
            market_panel = grouped.get(str(market.market_id))
            if market_panel is None or market_panel.empty:
                continue

            for horizon_hours in self.horizons_hours:
                cutoff = market.end_date - pd.Timedelta(hours=int(horizon_hours))
                if cutoff <= market.created_at:
                    continue

                features = extract_snapshot_features(
                    market_panel,
                    cutoff=cutoff,
                    history_hours=(24, 24 * 7),
                    max_snapshot_staleness_hours=self.max_snapshot_staleness_hours,
                )
                if features is None:
                    continue

                label = int(float(market.final_yes_probability) >= 0.5)
                base_prob = float(features["current_yes_probability"])
                rows.append(
                    {
                        "market_id": str(market.market_id),
                        "market_slug": market.market_slug,
                        "question": market.question,
                        "description": market.description,
                        "resolution_source": market.resolution_source,
                        "created_at": market.created_at,
                        "end_date": market.end_date,
                        "final_outcome": market.final_outcome,
                        "domain": market.domain,
                        "primary_domain": market.primary_domain,
                        "family_id": market.family_id,
                        "volume_num": market.volume_num,
                        "trade_rows": market.trade_rows,
                        "probability_rows": market.probability_rows,
                        "horizon_hours": int(horizon_hours),
                        "horizon_name": f"{float(horizon_hours) / 24.0:g}d",
                        "target": label,
                        "market_price_baseline": base_prob,
                        "market_abs_error": abs(label - base_prob),
                        "market_log_loss": binary_log_loss(np.array([label]), np.array([base_prob]))[0],
                        **features,
                    }
                )

        frame = pd.DataFrame(rows)
        if frame.empty:
            return RepresentationFrame(name="terminal_panel", frame=frame)

        frame = add_time_progress_features(frame, timestamp_col="cutoff_timestamp_utc")
        if self.include_family_context:
            frame = FamilyContextBuilder(
                market_meta=markets_df[["market_id", "family_id"]],
                probabilities=probabilities_df,
            ).attach(
                frame,
                timestamp_col="cutoff_timestamp_utc",
                market_price_col="market_price_baseline",
                prefix="family",
            )

        domain_dummies = pd.get_dummies(frame["primary_domain"], prefix="domain", dtype=float)
        frame = pd.concat([frame, domain_dummies], axis=1)
        frame = frame.sort_values(["end_date", "market_id", "horizon_hours"], kind="stable").reset_index(drop=True)
        return RepresentationFrame(name="terminal_panel", frame=frame)
