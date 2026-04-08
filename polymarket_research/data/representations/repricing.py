"""Repricing panels derived from canonical market trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from polymarket_research.data.canonical.dataset import CanonicalDataset
from polymarket_research.data.representations.common import (
    RepresentationFrame,
    add_time_progress_features,
    safe_float,
    safe_max,
    safe_mean,
    safe_std,
    safe_sum,
)
from polymarket_research.data.representations.external import ShockPanelBuilder, asof_join_covariates


@dataclass
class RepricingPanelBuilder:
    """Build future-repricing panels from canonical market trajectories."""

    canonical: CanonicalDataset
    future_horizon_hours: int = 24
    lookback_hours: int = 24
    sample_every_hours: int = 12
    move_threshold: float = 0.15
    attach_external_shocks: bool = True
    shock_z_threshold: float = 2.0
    shock_std_window: int = 288
    shock_join_max_age: str | None = "2D"

    def build(self) -> RepresentationFrame:
        """Build the repricing panel and optionally join external shock features."""
        markets_df = self.canonical.markets
        probabilities_df = self.canonical.probabilities

        future_steps = int(self.future_horizon_hours * 12)
        history_steps = int(self.lookback_hours * 12)
        step_stride = max(1, int(self.sample_every_hours * 12))

        rows: list[dict[str, object]] = []
        meta_by_market = markets_df.set_index("market_id").to_dict("index")

        for market_id, frame in probabilities_df.groupby("market_id", sort=False):
            meta = meta_by_market.get(str(market_id))
            if meta is None:
                continue

            frame = frame.reset_index(drop=True)
            if len(frame) <= future_steps + history_steps:
                continue

            yes = frame["yes_probability"].to_numpy(dtype=float)
            ts = frame["timestamp_utc"].to_numpy()
            observed = frame["observed_trade"].to_numpy(dtype=float)
            trade_count = frame["trade_count"].to_numpy(dtype=float)
            total_size = frame["total_size"].to_numpy(dtype=float)

            for idx in range(history_steps, len(frame) - future_steps, step_stride):
                current_ts = pd.Timestamp(ts[idx])
                hours_to_resolution = (meta["end_date"] - current_ts).total_seconds() / 3600.0
                if hours_to_resolution < self.future_horizon_hours:
                    continue

                hist_slice = slice(idx - history_steps, idx + 1)
                yes_hist = yes[hist_slice]
                diff_hist = yes_hist[1:] - yes_hist[:-1]
                future_prob = float(yes[idx + future_steps])
                current_prob = float(yes[idx])
                future_move = future_prob - current_prob

                rows.append(
                    {
                        "market_id": str(market_id),
                        "timestamp_utc": current_ts,
                        "created_at": meta["created_at"],
                        "end_date": meta["end_date"],
                        "question": meta["question"],
                        "description": meta["description"],
                        "primary_domain": meta["primary_domain"],
                        "domain": meta["domain"],
                        "family_id": meta["family_id"],
                        "future_horizon_hours": int(self.future_horizon_hours),
                        "target": int(abs(future_move) >= self.move_threshold),
                        "future_move": future_move,
                        "current_yes_probability": current_prob,
                        "confidence_margin": abs(current_prob - 0.5),
                        "volume_num": safe_float(meta["volume_num"]),
                        "trade_rows": safe_float(meta["trade_rows"]),
                        "probability_rows": safe_float(meta["probability_rows"]),
                        "recent_abs_move_mean": safe_mean(abs(diff_hist)),
                        "recent_abs_move_max": safe_max(abs(diff_hist)),
                        "recent_volatility": safe_std(diff_hist),
                        "recent_directional_move": float(current_prob - yes_hist[0]),
                        "observed_trade_share": safe_mean(observed[hist_slice]),
                        "trade_count_sum": safe_sum(trade_count[hist_slice]),
                        "total_size_sum": safe_sum(total_size[hist_slice]),
                    }
                )

        frame = pd.DataFrame(rows)
        if frame.empty:
            return RepresentationFrame(name="repricing_panel", frame=frame)

        frame = add_time_progress_features(frame, timestamp_col="timestamp_utc")
        domain_dummies = pd.get_dummies(frame["primary_domain"], prefix="domain", dtype=float)
        frame = pd.concat([frame, domain_dummies], axis=1)

        if self.attach_external_shocks:
            shock_panel = ShockPanelBuilder(
                canonical=self.canonical,
                z_threshold=self.shock_z_threshold,
                std_window=self.shock_std_window,
            ).build().frame
            if not shock_panel.empty:
                frame = asof_join_covariates(
                    frame,
                    shock_panel,
                    base_time_col="timestamp_utc",
                    covariate_time_col="timestamp_utc",
                    max_age=self.shock_join_max_age,
                )
                shock_cols = [col for col in ("btc_usd_shock", "eth_usd_shock") if col in frame.columns]
                if shock_cols:
                    frame["btc_or_eth_shock"] = frame[shock_cols].max(axis=1)

        frame = frame.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
        return RepresentationFrame(name="repricing_panel", frame=frame)
