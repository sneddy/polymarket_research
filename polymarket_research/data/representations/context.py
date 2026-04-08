"""Context features derived from weak market family structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def latest_context_snapshot(probabilities_df: pd.DataFrame, market_id: str, cutoff: pd.Timestamp) -> pd.Series | None:
    """Return the latest probability row for a market before a cutoff timestamp."""
    history = probabilities_df.loc[
        (probabilities_df["market_id"] == market_id) & (probabilities_df["timestamp_utc"] <= cutoff)
    ]
    if history.empty:
        return None
    return history.iloc[-1]


@dataclass
class FamilyContextBuilder:
    """Build family-level snapshot aggregates for market-centered panels."""

    market_meta: pd.DataFrame
    probabilities: pd.DataFrame

    def attach(
        self,
        panel: pd.DataFrame,
        *,
        timestamp_col: str,
        market_price_col: str,
        prefix: str = "family",
    ) -> pd.DataFrame:
        """Attach weak family context features to a panel with market snapshots."""
        family_map = self.market_meta.groupby("family_id")["market_id"].apply(list).to_dict()
        family_by_market = self.market_meta.set_index("market_id")["family_id"].to_dict()

        rows: list[dict[str, object]] = []
        for row in panel[["market_id", timestamp_col, market_price_col]].itertuples(index=False):
            family_id = family_by_market.get(row.market_id)
            related_ids = [market_id for market_id in family_map.get(family_id, []) if market_id != row.market_id]
            related_probs: list[float] = []
            for related_id in related_ids:
                snapshot = latest_context_snapshot(self.probabilities, related_id, getattr(row, timestamp_col))
                if snapshot is None:
                    continue
                related_probs.append(float(snapshot["yes_probability"]))

            rows.append(
                {
                    "market_id": row.market_id,
                    timestamp_col: getattr(row, timestamp_col),
                    f"{prefix}_related_count": float(len(related_probs)),
                    f"{prefix}_prob_mean": float(np.mean(related_probs)) if related_probs else np.nan,
                    f"{prefix}_prob_gap": float(np.max(related_probs) - np.min(related_probs)) if related_probs else np.nan,
                    f"{prefix}_vs_market_gap": float(abs(np.mean(related_probs) - getattr(row, market_price_col)))
                    if related_probs
                    else np.nan,
                }
            )

        context = pd.DataFrame(rows)
        return panel.merge(context, on=["market_id", timestamp_col], how="left")
