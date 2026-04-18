"""
Belief Updating Dataset Builder
===============================

Builds a shared manifest for two related protocols:

1. Legacy MVP outcome prediction:
   stale-local target state + fresh non-local context -> terminal outcome.
2. Main belief-update recovery:
   stale-local target state + fresh non-local context -> hidden current-state update.

Each example is anchored by a target market A, a context time t, and a stale time
t - Δ. The produced manifest keeps enough flat columns to support both protocols:

- stale local features for A at t - Δ;
- per-sibling context snapshots at t for set encoders;
- aggregated context summaries for tabular baselines;
- current-state labels and downstream labels stored in the examples frame.

Design note
-----------
We keep the data builder completely separate from the model so the same manifest
can feed sklearn baselines, PyTorch set encoders, and multiple protocol notebooks.
No torch import lives here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from polymarket_research.data.canonical.dataset import CanonicalDataset
from polymarket_research.data.representations.common import iter_with_progress, safe_float


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

#: Features extracted for each *family sibling* at the context time t.
#  Each sibling contributes one D_ctx-dimensional row in the context matrix.
CONTEXT_FEATURE_NAMES: list[str] = [
    "ctx_yes_probability",          # sibling's current probability at t
    "ctx_confidence_margin",        # |p - 0.5|: how decisive the sibling is
    "ctx_prob_vs_target_stale",     # sibling p − target stale p (key non-local signal)
    "ctx_lookback_24h_change",      # probability drift in last 24 h at t
    "ctx_lookback_24h_volatility",  # std of 5-min diffs in last 24 h at t
    "ctx_lookback_24h_log_trade",   # log(1 + trade count) last 24 h at t
    "ctx_lookback_24h_active_frac", # fraction of 5-min slots with an observed trade
    "ctx_life_progress",            # (context_time − created_at) / lifetime ∈ [0, 1]
]

#: Features for the *target* market A at the stale time t − Δ.
STALE_FEATURE_NAMES: list[str] = [
    "stale_yes_probability",        # target's probability at t − Δ
    "stale_confidence_margin",      # |p − 0.5| at t − Δ
    "stale_lookback_24h_change",    # probability drift 24 h before t − Δ
    "stale_lookback_24h_volatility",
    "stale_lookback_24h_log_trade",
    "stale_lookback_24h_active_frac",
    "stale_lookback_168h_change",   # longer-range drift
    "stale_lookback_168h_volatility",
    "stale_life_progress",          # (stale_time − created_at) / lifetime
    "stale_hours_to_end",           # (end_date − stale_time) in hours
    "delta_hours",                  # known staleness gap = context_time − stale_time
]

#: Aggregated context columns (mean + max over siblings) used for raw-context baseline.
AGG_CONTEXT_FEATURE_NAMES: list[str] = (
    [f"{name}_mean" for name in CONTEXT_FEATURE_NAMES]
    + [f"{name}_max" for name in CONTEXT_FEATURE_NAMES]
    + ["n_siblings"]
)

#: Flat global covariates aligned at the context time t.
GLOBAL_FEATURE_NAMES: list[str] = [
    "global_btc_return_1h",
    "global_btc_return_24h",
    "global_eth_return_1h",
    "global_eth_return_24h",
]


# ---------------------------------------------------------------------------
# Low-level feature extraction
# ---------------------------------------------------------------------------

def _extract_sibling_features(
    sibling_panel: pd.DataFrame,
    *,
    sibling_created_at: pd.Timestamp,
    sibling_end_date: pd.Timestamp,
    context_time: pd.Timestamp,
    target_stale_prob: float,
    max_staleness_hours: float = 12.0,
) -> dict[str, float] | None:
    """
    Extract context features for one sibling market at a given context time.

    Parameters
    ----------
    sibling_panel:
        Probability panel for the sibling, pre-filtered to rows <= context_time.
    sibling_created_at, sibling_end_date:
        Sibling market metadata (for life-progress calculation).
    context_time:
        The fresh context cutoff t.
    target_stale_prob:
        Target market's probability at t − Δ.  Used to compute the divergence
        signal ``ctx_prob_vs_target_stale``.
    max_staleness_hours:
        Reject siblings whose latest observation is more than this many hours
        before context_time (thin/inactive markets add noise, not signal).

    Returns
    -------
    dict[str, float] mapping CONTEXT_FEATURE_NAMES, or None if unusable.
    """
    history = sibling_panel.loc[sibling_panel["timestamp_utc"] <= context_time]
    if history.empty:
        return None

    latest = history.iloc[-1]
    latest_ts = pd.Timestamp(latest["timestamp_utc"])
    staleness_h = float((context_time - latest_ts).total_seconds() / 3600.0)
    if staleness_h > max_staleness_hours:
        return None  # sibling is stale; don't include in context

    prob = float(latest["yes_probability"])

    # 24-hour lookback window
    window_start = context_time - pd.Timedelta(hours=24)
    window = history.loc[history["timestamp_utc"] > window_start]
    diffs = window["yes_probability"].diff().dropna().to_numpy(dtype=float)

    change_24h = float(window["yes_probability"].iloc[-1] - window["yes_probability"].iloc[0]) if len(window) > 1 else 0.0
    vol_24h = float(np.std(diffs)) if len(diffs) > 0 else 0.0
    log_trade_24h = float(np.log1p(window["trade_count"].sum())) if not window.empty else 0.0
    active_frac_24h = float(window["observed_trade"].mean()) if not window.empty else 0.0

    # Life progress of the sibling at context_time
    lifetime = float((sibling_end_date - sibling_created_at).total_seconds())
    elapsed = float((context_time - sibling_created_at).total_seconds())
    life_progress = float(np.clip(elapsed / max(lifetime, 1.0), 0.0, 1.0))

    return {
        "ctx_yes_probability": prob,
        "ctx_confidence_margin": abs(prob - 0.5),
        "ctx_prob_vs_target_stale": prob - target_stale_prob,
        "ctx_lookback_24h_change": change_24h,
        "ctx_lookback_24h_volatility": vol_24h,
        "ctx_lookback_24h_log_trade": log_trade_24h,
        "ctx_lookback_24h_active_frac": active_frac_24h,
        "ctx_life_progress": life_progress,
    }


def _extract_stale_features(
    target_panel: pd.DataFrame,
    *,
    created_at: pd.Timestamp,
    end_date: pd.Timestamp,
    stale_time: pd.Timestamp,
    context_time: pd.Timestamp,
    max_staleness_hours: float = 12.0,
) -> dict[str, float] | None:
    """
    Extract stale-local features for the target market at t − Δ.

    Parameters
    ----------
    target_panel:
        Probability panel for the target market.
    created_at, end_date:
        Market lifetime bounds.
    stale_time:
        The stale cutoff t − Δ.
    context_time:
        The fresh context time t.  Used only to compute ``delta_hours``.
    max_staleness_hours:
        Reject if the latest observation before stale_time is more than this
        many hours stale (the market has no data at the desired cutoff).

    Returns
    -------
    dict[str, float] mapping STALE_FEATURE_NAMES, or None if unusable.
    """
    history = target_panel.loc[target_panel["timestamp_utc"] <= stale_time]
    if history.empty:
        return None

    latest = history.iloc[-1]
    latest_ts = pd.Timestamp(latest["timestamp_utc"])
    snapshot_staleness_h = float((stale_time - latest_ts).total_seconds() / 3600.0)
    if snapshot_staleness_h > max_staleness_hours:
        return None

    prob = float(latest["yes_probability"])

    # 24-hour lookback window relative to stale_time
    w24_start = stale_time - pd.Timedelta(hours=24)
    w24 = history.loc[history["timestamp_utc"] > w24_start]
    diff24 = w24["yes_probability"].diff().dropna().to_numpy(dtype=float)
    change_24h = float(w24["yes_probability"].iloc[-1] - w24["yes_probability"].iloc[0]) if len(w24) > 1 else 0.0
    vol_24h = float(np.std(diff24)) if len(diff24) > 0 else 0.0
    log_trade_24h = float(np.log1p(w24["trade_count"].sum())) if not w24.empty else 0.0
    active_24h = float(w24["observed_trade"].mean()) if not w24.empty else 0.0

    # 168-hour (one week) lookback
    w168_start = stale_time - pd.Timedelta(hours=168)
    w168 = history.loc[history["timestamp_utc"] > w168_start]
    diff168 = w168["yes_probability"].diff().dropna().to_numpy(dtype=float)
    change_168h = float(w168["yes_probability"].iloc[-1] - w168["yes_probability"].iloc[0]) if len(w168) > 1 else 0.0
    vol_168h = float(np.std(diff168)) if len(diff168) > 0 else 0.0

    # Life progress and time-to-end at stale_time
    lifetime = float((end_date - created_at).total_seconds())
    elapsed = float((stale_time - created_at).total_seconds())
    life_progress = float(np.clip(elapsed / max(lifetime, 1.0), 0.0, 1.0))
    hours_to_end = float((end_date - stale_time).total_seconds() / 3600.0)
    delta_hours = float((context_time - stale_time).total_seconds() / 3600.0)

    return {
        "stale_yes_probability": prob,
        "stale_confidence_margin": abs(prob - 0.5),
        "stale_lookback_24h_change": change_24h,
        "stale_lookback_24h_volatility": vol_24h,
        "stale_lookback_24h_log_trade": log_trade_24h,
        "stale_lookback_24h_active_frac": active_24h,
        "stale_lookback_168h_change": change_168h,
        "stale_lookback_168h_volatility": vol_168h,
        "stale_life_progress": life_progress,
        "stale_hours_to_end": hours_to_end,
        "delta_hours": delta_hours,
    }


def _safe_logit(probability: float, *, eps: float = 1e-6) -> float:
    """Convert probability to logit with clipping for numerical stability."""
    p = float(np.clip(probability, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def _safe_sigmoid(logit: float) -> float:
    """Convert a scalar logit back to probability."""
    return float(1.0 / (1.0 + np.exp(-float(logit))))


def _extract_current_target_state(
    target_panel: pd.DataFrame,
    *,
    context_time: pd.Timestamp,
    max_staleness_hours: float = 12.0,
) -> dict[str, float] | None:
    """Extract the hidden current target state at context time t for labels only."""
    history = target_panel.loc[target_panel["timestamp_utc"] <= context_time]
    if history.empty:
        return None

    latest = history.iloc[-1]
    latest_ts = pd.Timestamp(latest["timestamp_utc"])
    staleness_h = float((context_time - latest_ts).total_seconds() / 3600.0)
    if staleness_h > max_staleness_hours:
        return None

    prob = float(latest["yes_probability"])
    return {
        "target_current_probability": prob,
        "target_current_logit": _safe_logit(prob),
    }


def _extract_future_move_label(
    target_panel: pd.DataFrame,
    *,
    context_time: pd.Timestamp,
    future_horizon_hours: int = 24,
    move_threshold: float = 0.15,
    max_staleness_hours: float = 12.0,
) -> dict[str, float]:
    """Extract a simple future repricing label from the next admissible snapshot."""
    future_time = context_time + pd.Timedelta(hours=int(future_horizon_hours))
    history = target_panel.loc[target_panel["timestamp_utc"] <= future_time]
    if history.empty:
        return {
            "future_24h_probability": np.nan,
            "future_24h_move": np.nan,
            "future_24h_repricing_label": np.nan,
        }

    latest = history.iloc[-1]
    latest_ts = pd.Timestamp(latest["timestamp_utc"])
    staleness_h = float((future_time - latest_ts).total_seconds() / 3600.0)
    if staleness_h > max_staleness_hours:
        return {
            "future_24h_probability": np.nan,
            "future_24h_move": np.nan,
            "future_24h_repricing_label": np.nan,
        }

    future_prob = float(latest["yes_probability"])
    current_state = _extract_current_target_state(
        target_panel,
        context_time=context_time,
        max_staleness_hours=max_staleness_hours,
    )
    if current_state is None:
        return {
            "future_24h_probability": np.nan,
            "future_24h_move": np.nan,
            "future_24h_repricing_label": np.nan,
        }

    move = abs(future_prob - float(current_state["target_current_probability"]))
    return {
        "future_24h_probability": future_prob,
        "future_24h_move": move,
        "future_24h_repricing_label": float(move >= float(move_threshold)),
    }


def _build_external_lookup(external_covariates: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """Prepare the small set of global covariate series used by the main protocol."""
    if external_covariates is None or external_covariates.empty:
        return {}

    out: dict[str, pd.DataFrame] = {}
    for key, candidates in {
        "btc_usd": ("btc_usd", "BTCUSDT"),
        "eth_usd": ("eth_usd", "ETHUSDT"),
    }.items():
        frame = external_covariates.loc[
            external_covariates.get("series_id", pd.Series(dtype=object)).astype(str).isin(candidates)
            | external_covariates.get("provider_symbol", pd.Series(dtype=object)).astype(str).isin(candidates)
        ].copy()
        if frame.empty:
            continue
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
        value_col = "close" if "close" in frame.columns else "value"
        frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
        frame = frame.dropna(subset=["timestamp_utc", value_col]).sort_values("timestamp_utc", kind="stable")
        if frame.empty:
            continue
        out[key] = frame[["timestamp_utc", value_col]].rename(columns={value_col: "value"}).reset_index(drop=True)
    return out


def _lookup_latest_value(series: pd.DataFrame, timestamp: pd.Timestamp) -> float:
    """Return the latest series value at or before the timestamp, or NaN if absent."""
    history = series.loc[series["timestamp_utc"] <= timestamp]
    if history.empty:
        return np.nan
    return safe_float(history.iloc[-1]["value"])


def _extract_global_features(
    external_lookup: dict[str, pd.DataFrame],
    *,
    context_time: pd.Timestamp,
) -> dict[str, float]:
    """Extract BTC/ETH return covariates aligned to the context time."""
    features = {name: np.nan for name in GLOBAL_FEATURE_NAMES}
    if not external_lookup:
        return features

    for key, prefix in (("btc_usd", "global_btc"), ("eth_usd", "global_eth")):
        series = external_lookup.get(key)
        if series is None or series.empty:
            continue
        now = _lookup_latest_value(series, context_time)
        prev_1h = _lookup_latest_value(series, context_time - pd.Timedelta(hours=1))
        prev_24h = _lookup_latest_value(series, context_time - pd.Timedelta(hours=24))
        if np.isfinite(now) and np.isfinite(prev_1h) and prev_1h != 0:
            features[f"{prefix}_return_1h"] = float((now / prev_1h) - 1.0)
        if np.isfinite(now) and np.isfinite(prev_24h) and prev_24h != 0:
            features[f"{prefix}_return_24h"] = float((now / prev_24h) - 1.0)
    return features


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BeliefUpdatingSpec:
    """
    Configuration for the belief-updating dataset builder.

    Parameters
    ----------
    horizons_hours:
        Times-before-resolution at which we fix the *context* time t.
        These match the terminal benchmark horizons so results are comparable.
    delta_hours_options:
        Staleness gaps Δ to try for each horizon.  For each (horizon, delta)
        pair we create one training example per market.  Only pairs where
        stale_time > created_at are kept.
    max_snapshot_staleness_hours:
        Maximum tolerated gap between the requested cutoff and the latest
        observed 5-minute row.  Rows older than this are discarded.
    min_family_size:
        Minimum number of *sibling* markets (excluding the target) required
        to create an example.  Set to 1 to include singletons; set higher to
        restrict to well-connected families.
    include_global_context:
        When True, attach simple BTC/ETH return features aligned at context time.
    repricing_move_threshold:
        Threshold for the optional future-24h repricing label.
    """

    horizons_hours: tuple[int, ...] = (24, 72, 168)
    delta_hours_options: tuple[int, ...] = (24, 72, 168)
    max_snapshot_staleness_hours: float = 12.0
    min_family_size: int = 1
    include_global_context: bool = True
    repricing_move_threshold: float = 0.15


# ---------------------------------------------------------------------------
# Manifest container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BeliefUpdatingManifest:
    """
    Output of BeliefUpdatingDatasetBuilder.

    Attributes
    ----------
    examples:
        Flat DataFrame.  One row per (market_id, context_time, delta_hours).
        Contains:
          - metadata: market_id, question, family_id, research_category, end_date, …
          - stale features: STALE_FEATURE_NAMES columns
          - aggregated context: AGG_CONTEXT_FEATURE_NAMES columns (mean/max/count)
          - label: 0 or 1 (final binary outcome)

    context_snapshots:
        Per-market snapshots.  One row per (market_id, snapshot_time).
        Contains CONTEXT_FEATURE_NAMES columns (plus identifiers).
        Used by BeliefUpdatingTorchDataset to assemble context matrices.

    stale_feature_names:
        Ordered list of stale-feature column names (STALE_FEATURE_NAMES).
    context_feature_names:
        Ordered list of context-feature column names (CONTEXT_FEATURE_NAMES).
    agg_context_feature_names:
        Ordered list of aggregated context column names (AGG_CONTEXT_FEATURE_NAMES).
    global_feature_names:
        Ordered list of aligned global covariates stored in examples.
    spec:
        The BeliefUpdatingSpec that was used to build this manifest.
    """

    examples: pd.DataFrame
    context_snapshots: pd.DataFrame
    stale_feature_names: list[str]
    context_feature_names: list[str]
    agg_context_feature_names: list[str]
    global_feature_names: list[str]
    spec: BeliefUpdatingSpec

    def summary(self) -> pd.DataFrame:
        """Return a one-line summary of the manifest size and coverage."""
        n_markets = self.examples["market_id"].nunique() if not self.examples.empty else 0
        n_families = self.examples["family_id"].nunique() if not self.examples.empty else 0
        return pd.DataFrame([{
            "examples": len(self.examples),
            "unique_markets": n_markets,
            "unique_families": n_families,
            "context_snapshots": len(self.context_snapshots),
            "stale_features": len(self.stale_feature_names),
            "context_features_per_sibling": len(self.context_feature_names),
            "global_features": len(self.global_feature_names),
        }])

    def save(self, directory: str | Path) -> None:
        """Persist the manifest to parquet files in *directory*."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.examples.to_parquet(out / "examples.parquet", index=False)
        self.context_snapshots.to_parquet(out / "context_snapshots.parquet", index=False)

    @classmethod
    def from_parquet(cls, directory: str | Path, spec: BeliefUpdatingSpec) -> "BeliefUpdatingManifest":
        """Load a previously saved manifest from parquet files."""
        src = Path(directory)
        examples = pd.read_parquet(src / "examples.parquet")
        context_snapshots = pd.read_parquet(src / "context_snapshots.parquet")
        return cls(
            examples=examples,
            context_snapshots=context_snapshots,
            stale_feature_names=list(STALE_FEATURE_NAMES),
            context_feature_names=list(CONTEXT_FEATURE_NAMES),
            agg_context_feature_names=list(AGG_CONTEXT_FEATURE_NAMES),
            global_feature_names=list(GLOBAL_FEATURE_NAMES),
            spec=spec,
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

@dataclass
class BeliefUpdatingDatasetBuilder:
    """
    Build belief-updating examples from a CanonicalDataset.

    Usage
    -----
    ::

        from polymarket_research.data.canonical import CanonicalDataset
        from polymarket_research.belief_updating import (
            BeliefUpdatingDatasetBuilder, BeliefUpdatingSpec,
        )

        canonical = CanonicalDataset.from_parquet(CANONICAL_CACHE_DIR)
        spec = BeliefUpdatingSpec()                 # use defaults
        builder = BeliefUpdatingDatasetBuilder(canonical=canonical, spec=spec)
        manifest = builder.build(show_progress=True)

    The resulting ``manifest.examples`` DataFrame is ready for sklearn baselines.
    The ``manifest.context_snapshots`` table feeds the PyTorch set encoder.
    """

    canonical: CanonicalDataset
    spec: BeliefUpdatingSpec = field(default_factory=BeliefUpdatingSpec)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self, *, show_progress: bool = False) -> BeliefUpdatingManifest:
        """
        Build the complete manifest.

        Steps:
        1. Group probability panels by market_id (once, for speed).
        2. Build a family map: family_id → list of market_ids.
        3. For each market, for each (horizon, delta) combination:
           a. Compute context_time = end_date − horizon
           b. Compute stale_time   = context_time − delta
           c. Extract stale features for the target at stale_time.
           d. Find family siblings (other markets in the same family).
           e. Extract context features for each sibling at context_time.
           f. Aggregate (mean/max) context features for raw-context baseline.
        4. Store per-sibling snapshots in a separate lookup table.

        Returns
        -------
        BeliefUpdatingManifest
        """
        markets_df = self.canonical.markets
        probabilities_df = self.canonical.probabilities
        external_lookup = (
            _build_external_lookup(self.canonical.external_covariates)
            if self.spec.include_global_context
            else {}
        )

        # Pre-group probability panels by market_id (avoids repeated filtering)
        grouped_probs: dict[str, pd.DataFrame] = {
            mid: frame.sort_values("timestamp_utc").reset_index(drop=True)
            for mid, frame in probabilities_df.groupby("market_id", sort=False)
        }

        # family_id → [market_ids] (all markets in each family)
        family_map: dict[str, list[str]] = (
            markets_df.groupby("family_id")["market_id"].apply(list).to_dict()
        )
        # market_id → family_id
        family_by_market: dict[str, str] = (
            markets_df.set_index("market_id")["family_id"].to_dict()
        )
        # market_id → market metadata row (for sibling metadata in context step)
        meta_by_market: dict[str, object] = {
            row.market_id: row
            for row in markets_df.itertuples(index=False)
        }

        example_rows: list[dict] = []
        # sibling_key → context feature dict  (deduplicated across target markets)
        context_snapshot_map: dict[tuple, dict] = {}

        market_iter = iter_with_progress(
            markets_df.itertuples(index=False),
            enabled=show_progress,
            desc="belief-updating examples",
            total=len(markets_df),
        )

        for market in market_iter:
            target_id = str(market.market_id)
            target_panel = grouped_probs.get(target_id)
            if target_panel is None or target_panel.empty:
                continue

            # Final label: 1 if market resolved Yes, 0 if No
            try:
                label = int(float(market.final_yes_probability) >= 0.5)
            except (TypeError, ValueError):
                continue

            # Family siblings (excluding target itself)
            family_id = str(family_by_market.get(target_id, ""))
            sibling_ids = [
                mid for mid in family_map.get(family_id, [])
                if mid != target_id
            ]
            if len(sibling_ids) < self.spec.min_family_size:
                continue

            # Iterate over all (horizon, delta) combinations
            for horizon_h in self.spec.horizons_hours:
                context_time = market.end_date - pd.Timedelta(hours=int(horizon_h))
                if context_time <= market.created_at:
                    continue  # market too short for this horizon

                for delta_h in self.spec.delta_hours_options:
                    if delta_h >= horizon_h:
                        # stale_time would be before or at market creation
                        # (context_time - delta >= context_time - horizon = created_at approx)
                        # Allow it only if there's room
                        pass
                    stale_time = context_time - pd.Timedelta(hours=int(delta_h))
                    if stale_time <= market.created_at:
                        continue  # no data before market opened

                    # --- Stale local features ---
                    stale_feats = _extract_stale_features(
                        target_panel,
                        created_at=market.created_at,
                        end_date=market.end_date,
                        stale_time=stale_time,
                        context_time=context_time,
                        max_staleness_hours=self.spec.max_snapshot_staleness_hours,
                    )
                    if stale_feats is None:
                        continue  # target had no data at stale_time

                    target_stale_prob = stale_feats["stale_yes_probability"]
                    stale_logit = _safe_logit(target_stale_prob)
                    current_state = _extract_current_target_state(
                        target_panel,
                        context_time=context_time,
                        max_staleness_hours=self.spec.max_snapshot_staleness_hours,
                    )
                    if current_state is None:
                        continue

                    current_prob = float(current_state["target_current_probability"])
                    current_logit = float(current_state["target_current_logit"])
                    update_logit = current_logit - stale_logit
                    stale_error_abs = abs(current_prob - target_stale_prob)
                    future_move = _extract_future_move_label(
                        target_panel,
                        context_time=context_time,
                        future_horizon_hours=24,
                        move_threshold=self.spec.repricing_move_threshold,
                        max_staleness_hours=self.spec.max_snapshot_staleness_hours,
                    )
                    global_feats = _extract_global_features(
                        external_lookup,
                        context_time=context_time,
                    )

                    # --- Context features for each sibling ---
                    sibling_feats_list: list[dict[str, float]] = []
                    for sib_id in sibling_ids:
                        sib_panel = grouped_probs.get(sib_id)
                        if sib_panel is None or sib_panel.empty:
                            continue
                        sib_meta = meta_by_market.get(sib_id)
                        if sib_meta is None:
                            continue

                        # Check if we already computed this sibling snapshot
                        snapshot_key = (sib_id, context_time)
                        if snapshot_key in context_snapshot_map:
                            sib_feats_raw = context_snapshot_map[snapshot_key]
                            # Recompute target-relative feature
                            sib_feats = dict(sib_feats_raw)
                            sib_feats["ctx_prob_vs_target_stale"] = (
                                sib_feats["ctx_yes_probability"] - target_stale_prob
                            )
                        else:
                            sib_feats = _extract_sibling_features(
                                sib_panel,
                                sibling_created_at=sib_meta.created_at,
                                sibling_end_date=sib_meta.end_date,
                                context_time=context_time,
                                target_stale_prob=target_stale_prob,
                                max_staleness_hours=self.spec.max_snapshot_staleness_hours,
                            )
                            if sib_feats is None:
                                continue
                            # Store in map WITHOUT target-relative field (differs per target)
                            base_feats = {k: v for k, v in sib_feats.items()
                                          if k != "ctx_prob_vs_target_stale"}
                            context_snapshot_map[snapshot_key] = {
                                **base_feats,
                                "market_id": sib_id,
                                "family_id": family_id,
                                "snapshot_time": context_time,
                            }

                        sibling_feats_list.append(sib_feats)

                    if not sibling_feats_list:
                        continue  # no usable siblings at this context_time

                    # --- Aggregate context for raw-context baseline ---
                    agg_feats = _aggregate_context(sibling_feats_list)

                    # --- Assemble example row ---
                    row = {
                        # Identifiers / metadata
                        "market_id": target_id,
                        "family_id": family_id,
                        "question": getattr(market, "question", ""),
                        "platform_category": getattr(market, "platform_category", ""),
                        "research_category": getattr(market, "research_category", ""),
                        "end_date": market.end_date,
                        "created_at": market.created_at,
                        "horizon_hours": int(horizon_h),
                        "delta_hours_int": int(delta_h),
                        "context_time": context_time,
                        "stale_time": stale_time,
                        "split_group_id": target_id,
                        # Stale local features
                        **stale_feats,
                        # Optional global features at context time
                        **global_feats,
                        # Aggregated context features (for rung 2)
                        **agg_feats,
                        # Legacy outcome label and richer main-protocol labels
                        "label": label,
                        "label_terminal_outcome": label,
                        "target_stale_logit": stale_logit,
                        "target_current_probability": current_prob,
                        "target_current_logit": current_logit,
                        "label_update_logit": update_logit,
                        "label_stale_error_abs": stale_error_abs,
                        "label_stale_error_ge_015": float(stale_error_abs >= 0.15),
                        **future_move,
                        # Market-price baseline (stale probability as naive predictor)
                        "stale_prob_baseline": target_stale_prob,
                        "stale_update_baseline": 0.0,
                    }
                    example_rows.append(row)

        examples = pd.DataFrame(example_rows)
        if not examples.empty:
            examples = examples.sort_values(
                ["end_date", "market_id", "horizon_hours", "delta_hours_int"],
                kind="stable",
            ).reset_index(drop=True)

        # Build context snapshots table from deduplicated map
        ctx_rows = list(context_snapshot_map.values())
        context_snapshots = pd.DataFrame(ctx_rows) if ctx_rows else pd.DataFrame(
            columns=["market_id", "family_id", "snapshot_time"] + CONTEXT_FEATURE_NAMES
        )
        # Drop the target-relative field from snapshots table (it depends on the target)
        if "ctx_prob_vs_target_stale" in context_snapshots.columns:
            context_snapshots = context_snapshots.drop(columns=["ctx_prob_vs_target_stale"])

        if show_progress:
            print(
                f"[belief_updating] built {len(examples)} examples, "
                f"{len(context_snapshots)} context snapshots"
            )

        return BeliefUpdatingManifest(
            examples=examples,
            context_snapshots=context_snapshots,
            stale_feature_names=list(STALE_FEATURE_NAMES),
            context_feature_names=list(CONTEXT_FEATURE_NAMES),
            agg_context_feature_names=list(AGG_CONTEXT_FEATURE_NAMES),
            global_feature_names=list(GLOBAL_FEATURE_NAMES),
            spec=self.spec,
        )


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def _aggregate_context(sibling_feats_list: list[dict[str, float]]) -> dict[str, float]:
    """
    Aggregate per-sibling context features into mean and max summary statistics.

    These summaries are the "raw context" used by sklearn baselines (rung 2).
    They lose permutation structure but are compatible with tabular models.
    """
    feature_arrays: dict[str, list[float]] = {name: [] for name in CONTEXT_FEATURE_NAMES}
    for feats in sibling_feats_list:
        for name in CONTEXT_FEATURE_NAMES:
            val = feats.get(name, float("nan"))
            if not np.isnan(val):
                feature_arrays[name].append(val)

    agg: dict[str, float] = {}
    for name in CONTEXT_FEATURE_NAMES:
        vals = feature_arrays[name]
        agg[f"{name}_mean"] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{name}_max"] = float(np.max(vals)) if vals else float("nan")
    agg["n_siblings"] = float(len(sibling_feats_list))
    return agg
