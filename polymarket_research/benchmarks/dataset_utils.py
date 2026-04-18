from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "db" / "resolved_probability_dataset.sqlite"


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def load_probabilities_for_markets(
    conn: sqlite3.Connection,
    market_ids: Sequence[str],
) -> pd.DataFrame:
    market_ids = [str(x) for x in market_ids]
    if not market_ids:
        return pd.DataFrame(
            columns=[
                "market_id",
                "timestamp_utc",
                "yes_probability",
                "observed_trade",
                "trade_count",
                "total_size",
                "last_trade_price",
            ]
        )

    frames: list[pd.DataFrame] = []
    for chunk in _chunked(market_ids, size=250):
        placeholders = ",".join(["?"] * len(chunk))
        query = f"""
        SELECT
            market_id,
            timestamp_utc,
            yes_probability,
            observed_trade,
            trade_count,
            total_size,
            last_trade_price
        FROM probabilities
        WHERE market_id IN ({placeholders})
        ORDER BY market_id, timestamp_utc
        """
        frame = pd.read_sql_query(query, conn, params=tuple(chunk))
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    for col in ("yes_probability", "trade_count", "total_size", "last_trade_price"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["observed_trade"] = pd.to_numeric(out["observed_trade"], errors="coerce").fillna(0).astype(int)
    out = out.sort_values(["market_id", "timestamp_utc"], kind="stable").reset_index(drop=True)
    return out


def prepare_resolved_markets(markets_df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_market_frame(markets_df)
    if "research_category" not in out.columns:
        out["research_category"] = None
    if "platform_category" not in out.columns:
        out["platform_category"] = None
    return out


def build_multi_horizon_terminal_dataset(
    markets_df: pd.DataFrame,
    probabilities_df: pd.DataFrame,
    *,
    horizons_hours: Sequence[int] = (24, 72, 168),
    max_snapshot_staleness_hours: float | None = 12.0,
) -> pd.DataFrame:
    grouped = {
        market_id: frame.reset_index(drop=True)
        for market_id, frame in probabilities_df.groupby("market_id", sort=False)
    }

    rows: list[dict[str, object]] = []
    for market in prepare_resolved_markets(markets_df).itertuples(index=False):
        market_panel = grouped.get(str(market.market_id))
        if market_panel is None or market_panel.empty:
            continue

        for horizon_hours in horizons_hours:
            cutoff = market.end_date - pd.Timedelta(hours=int(horizon_hours))
            if cutoff <= market.created_at:
                continue

            features = extract_snapshot_features(
                market_panel,
                cutoff=cutoff,
                history_hours=(24, 24 * 7),
                max_snapshot_staleness_hours=max_snapshot_staleness_hours,
            )
            if features is None:
                continue

            horizon_days = float(horizon_hours) / 24.0
            label = int(float(market.final_yes_probability) >= 0.5)
            base_prob = float(features["current_yes_probability"])
            rows.append(
                {
                    "market_id": str(market.market_id),
                    "market_slug": market.market_slug,
                    "question": market.question,
                    "end_date": market.end_date,
                    "created_at": market.created_at,
                    "volume_num": market.volume_num,
                    "trade_rows": market.trade_rows,
                    "probability_rows": market.probability_rows,
                    "platform_category": getattr(market, "platform_category", None),
                    "research_category": getattr(market, "research_category", None),
                    "horizon_hours": int(horizon_hours),
                    "horizon_name": f"{horizon_days:g}d",
                    "target": label,
                    "market_price_baseline": base_prob,
                    "market_abs_error": abs(label - base_prob),
                    "market_log_loss": binary_log_loss(np.array([label]), np.array([base_prob]))[0],
                    **features,
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["end_date", "market_id", "horizon_hours"],
        kind="stable",
    ).reset_index(drop=True)


def load_snapshot_frame(
    probabilities_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    *,
    horizon_hours: int,
    max_snapshot_staleness_hours: float | None = 12.0,
) -> pd.DataFrame:
    return build_multi_horizon_terminal_dataset(
        markets_df,
        probabilities_df,
        horizons_hours=(int(horizon_hours),),
        max_snapshot_staleness_hours=max_snapshot_staleness_hours,
    )


def build_repricing_dataset(
    markets_df: pd.DataFrame,
    probabilities_df: pd.DataFrame,
    *,
    future_horizon_hours: int = 24,
    lookback_hours: int = 24,
    sample_every_hours: int = 12,
    move_threshold: float = 0.15,
) -> pd.DataFrame:
    future_steps = int(future_horizon_hours * 12)
    history_steps = int(lookback_hours * 12)
    step_stride = max(1, int(sample_every_hours * 12))

    rows: list[dict[str, object]] = []
    meta_by_market = prepare_resolved_markets(markets_df).set_index("market_id").to_dict("index")

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
            if hours_to_resolution < future_horizon_hours:
                continue

            hist_slice = slice(idx - history_steps, idx + 1)
            yes_hist = yes[hist_slice]
            diff_hist = np.diff(yes_hist)
            future_prob = float(yes[idx + future_steps])
            current_prob = float(yes[idx])
            future_move = future_prob - current_prob

            rows.append(
                {
                    "market_id": str(market_id),
                    "timestamp_utc": current_ts,
                    "end_date": meta["end_date"],
                    "future_horizon_hours": int(future_horizon_hours),
                    "target": int(abs(future_move) >= move_threshold),
                    "future_move": future_move,
                    "current_yes_probability": current_prob,
                    "confidence_margin": abs(current_prob - 0.5),
                    "hours_to_resolution": hours_to_resolution,
                    "life_progress": _life_progress(
                        current_ts,
                        meta["created_at"],
                        meta["end_date"],
                    ),
                    "volume_num": _safe_float(meta["volume_num"]),
                    "trade_rows": _safe_float(meta["trade_rows"]),
                    "probability_rows": _safe_float(meta["probability_rows"]),
                    "platform_category": meta.get("platform_category"),
                    "research_category": meta.get("research_category"),
                    "recent_abs_move_mean": _safe_mean(np.abs(diff_hist)),
                    "recent_abs_move_max": _safe_max(np.abs(diff_hist)),
                    "recent_volatility": _safe_std(diff_hist),
                    "recent_directional_move": float(current_prob - yes_hist[0]),
                    "observed_trade_share": _safe_mean(observed[hist_slice]),
                    "trade_count_sum": _safe_sum(trade_count[hist_slice]),
                    "total_size_sum": _safe_sum(total_size[hist_slice]),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("timestamp_utc", kind="stable").reset_index(drop=True)


def extract_snapshot_features(
    market_panel: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    history_hours: Sequence[int] = (24, 168),
    max_snapshot_staleness_hours: float | None = None,
) -> dict[str, float] | None:
    history = market_panel.loc[market_panel["timestamp_utc"] <= cutoff].copy()
    if history.empty:
        return None

    current = history.iloc[-1]
    current_prob = float(current["yes_probability"])
    current_ts = pd.Timestamp(current["timestamp_utc"])
    staleness_hours = float((cutoff - current_ts).total_seconds() / 3600.0)
    if max_snapshot_staleness_hours is not None and staleness_hours > float(max_snapshot_staleness_hours):
        return None

    current_features: dict[str, float] = {
        "cutoff_timestamp_utc": current_ts,
        "snapshot_staleness_hours": staleness_hours,
        "current_yes_probability": current_prob,
        "confidence_margin": abs(current_prob - 0.5),
        "observed_trade_now": float(current["observed_trade"]),
        "trade_count_now": _safe_float(current["trade_count"]),
        "total_size_now": _safe_float(current["total_size"]),
        "last_trade_price_now": _safe_float(current["last_trade_price"]),
    }

    for hours in history_hours:
        window_start = cutoff - pd.Timedelta(hours=int(hours))
        window = history.loc[history["timestamp_utc"] > window_start]
        if window.empty:
            continue

        diff = window["yes_probability"].diff().dropna().to_numpy(dtype=float)
        prefix = f"lookback_{int(hours)}h"
        current_features[f"{prefix}_rows"] = float(len(window))
        current_features[f"{prefix}_observed_trade_share"] = float(window["observed_trade"].mean())
        current_features[f"{prefix}_trade_count_sum"] = float(window["trade_count"].sum())
        current_features[f"{prefix}_total_size_sum"] = float(window["total_size"].sum())
        current_features[f"{prefix}_yes_probability_change"] = float(
            window["yes_probability"].iloc[-1] - window["yes_probability"].iloc[0]
        )
        current_features[f"{prefix}_volatility"] = _safe_std(diff)
        current_features[f"{prefix}_abs_move_mean"] = _safe_mean(np.abs(diff))
        current_features[f"{prefix}_abs_move_max"] = _safe_max(np.abs(diff))

    return current_features


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hours_to_resolution"] = (
        (out["end_date"] - out["cutoff_timestamp_utc"]).dt.total_seconds() / 3600.0
    )
    out["market_age_hours"] = (
        (out["cutoff_timestamp_utc"] - out["created_at"]).dt.total_seconds() / 3600.0
    )
    out["life_progress"] = [
        _life_progress(ts, created_at, end_date)
        for ts, created_at, end_date in zip(
            out["cutoff_timestamp_utc"],
            out["created_at"],
            out["end_date"],
            strict=False,
        )
    ]
    return out


def default_feature_columns(df: pd.DataFrame, *, exclude: Iterable[str] | None = None) -> list[str]:
    excluded = {
        "market_id",
        "market_slug",
        "question",
        "end_date",
        "created_at",
        "cutoff_timestamp_utc",
        "timestamp_utc",
        "target",
        "future_move",
        "market_price_baseline",
        "market_abs_error",
        "market_log_loss",
        "final_outcome",
        "resolution_source",
        "description",
        "tag_labels",
        "platform_category",
        "research_category",
    }
    if exclude is not None:
        excluded.update(exclude)

    return sorted(
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    )


def rolling_time_splits(
    df: pd.DataFrame,
    *,
    time_col: str,
    n_splits: int = 4,
    min_train_fraction: float = 0.5,
) -> list[tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]]:
    work = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    if work.empty or len(work) < 3:
        return []

    n = len(work)
    start_train = max(1, int(n * float(min_train_fraction)))
    if start_train >= n - 1:
        start_train = max(1, n - 2)

    remaining = n - start_train
    n_effective = max(1, min(int(n_splits), remaining))
    test_size = max(1, remaining // n_effective)

    splits: list[tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]] = []
    train_end = start_train
    for split_idx in range(n_effective):
        test_end = n if split_idx == n_effective - 1 else min(n, train_end + test_size)
        train_df = work.iloc[:train_end].copy()
        test_df = work.iloc[train_end:test_end].copy()
        if train_df.empty or test_df.empty:
            break
        meta = {
            "fold": split_idx,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_start": train_df[time_col].min(),
            "train_end": train_df[time_col].max(),
            "test_start": test_df[time_col].min(),
            "test_end": test_df[time_col].max(),
        }
        splits.append((train_df, test_df, meta))
        train_end = test_end
        if train_end >= n:
            break
    return splits


def summarize_metric_frame(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
) -> pd.DataFrame:
    grouped = df.groupby(list(group_cols), dropna=False)
    agg_spec: dict[str, list[str]] = {col: ["mean", "std", "min", "max"] for col in metric_cols}
    out = grouped.agg(agg_spec)
    out.columns = ["_".join(filter(None, parts)).strip() for parts in out.columns.to_flat_index()]
    return out.reset_index()


def binary_log_loss(y_true: np.ndarray, p_pred: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), eps, 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _normalize_market_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("created_at", "end_date", "probability_start_utc", "probability_end_utc"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    for col in ("volume_num", "trade_rows", "probability_rows", "final_yes_probability"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "market_id" in out.columns:
        out["market_id"] = out["market_id"].astype(str)
    return out.dropna(subset=["created_at", "end_date"]).reset_index(drop=True)


def _chunked(values: Sequence[str], *, size: int) -> Iterable[list[str]]:
    for idx in range(0, len(values), size):
        yield list(values[idx : idx + size])


def _life_progress(timestamp: pd.Timestamp, created_at: pd.Timestamp, end_date: pd.Timestamp) -> float:
    total = (end_date - created_at).total_seconds()
    elapsed = (timestamp - created_at).total_seconds()
    if total <= 0:
        return np.nan
    return float(np.clip(elapsed / total, 0.0, 1.0))


def _safe_float(value: object) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def _safe_mean(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.std(values))


def _safe_sum(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.sum(values))


def _safe_max(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.max(values))

