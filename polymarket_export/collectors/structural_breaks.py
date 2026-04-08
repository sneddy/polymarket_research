from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StructuralBreakConfig:
    """
    Configurable rules for volume+price structural breaks.

    A break is detected when either:
      1) volume surged and price moved meaningfully
      2) volume stayed substantial (not low) and price moved strongly

    Low-volume price moves are explicitly filtered out.
    """

    interval: str = "15min"
    baseline_window: int = 96
    baseline_min_periods: int = 24

    # Outcome handling:
    # - if outcome is set, detector uses only this outcome
    # - else if auto_select_outcome=True and `outcome` column exists, detector picks
    #   the most-liquid outcome (by total traded size) to avoid YES/NO mixing noise
    outcome: str | None = None
    auto_select_outcome: bool = True

    # Price move thresholds in percent.
    price_change_pct: float = 2.0
    strong_price_change_pct: float = 4.0

    # Volume-vs-baseline thresholds.
    volume_surge_ratio: float = 2.0
    volume_substantial_ratio: float = 0.8
    low_volume_ratio: float = 0.35

    # Optional absolute volume fallback for "substantial" volume.
    min_volume_absolute: float | None = None

    # Interval quality filters.
    min_trades_per_interval: int = 1

    # Minimum break score filter (applied before max_breaks capping).
    min_break_score: float | None = None

    # Optional cap on number of returned anomalies.
    # If set, detector picks top-scoring breaks up to this limit.
    max_breaks: int | None = None

    # Keep only one strongest break inside +/- break_radius window.
    # Example: "3h" means selected breaks cannot be closer than 3 hours.
    break_radius: str | None = "3h"

    # Collapse adjacent signaled intervals into one strongest break.
    merge_consecutive_breaks: bool = True
    merge_gap_intervals: int = 1

    # Plot controls.
    max_annotations: int = 40
    plot_price_smoothing_span: int = 32
    plot_volume_smoothing_span: int = 32
    plot_volume_clip_quantile: float = 0.995
    plot_show_raw_price: bool = False
    plot_raw_price_alpha: float = 0.25


@dataclass
class StructuralBreakResult:
    interval: str
    bars: pd.DataFrame
    breaks: pd.DataFrame
    candidates: pd.DataFrame
    score_threshold: float | None
    selected_outcome: str | None
    summary: dict[str, Any]


class StructuralBreakDetector:
    def __init__(self, config: StructuralBreakConfig | None = None) -> None:
        self.config = config or StructuralBreakConfig()

    def detect(self, trades: pd.DataFrame) -> StructuralBreakResult:
        bars, selected_outcome = self._build_bars(trades)

        price_abs = bars["abs_price_return_pct"]
        volume_ratio = bars["volume_ratio"]
        volume_signal = bars["volume_ratio_signal"]

        price_significant = price_abs >= float(self.config.price_change_pct)
        price_strong = price_abs >= float(self.config.strong_price_change_pct)

        volume_surge = volume_ratio >= float(self.config.volume_surge_ratio)
        volume_substantial = bars["volume_substantial"]
        volume_low = bars["volume_low"]

        has_trade = bars["n_trades"] >= int(self.config.min_trades_per_interval)

        cond_primary = has_trade & volume_surge & price_significant
        cond_secondary = has_trade & volume_substantial & price_strong

        is_break = (cond_primary | cond_secondary) & (~volume_low)

        reason = np.where(
            cond_primary,
            "volume_surge_and_price_shift",
            np.where(cond_secondary, "substantial_volume_and_strong_price_shift", "none"),
        )

        # Higher score => stronger combined price+volume signal.
        price_component = price_abs / max(float(self.config.strong_price_change_pct), 1e-9)
        volume_component = np.log1p(np.clip(volume_signal, 0.0, None))
        score = price_component * (1.0 + volume_component)

        bars["is_break"] = is_break
        bars["break_reason"] = reason
        bars["break_score"] = score

        candidates = bars[bars["is_break"]].copy()

        if self.config.merge_consecutive_breaks and not candidates.empty:
            candidates = self._merge_neighbor_breaks(bars, candidates)

        if self.config.min_break_score is not None and not candidates.empty:
            candidates = candidates[candidates["break_score"] >= float(self.config.min_break_score)].copy()

        if not candidates.empty:
            candidates = self._apply_break_radius_nms(candidates)

        selected_threshold: float | None = None
        breaks = candidates.copy()
        if self.config.max_breaks is not None and not breaks.empty:
            limit = max(1, int(self.config.max_breaks))
            ranked = breaks.sort_values(
                ["break_score", "abs_price_return_pct", "volume_ratio_signal"],
                ascending=[False, False, False],
                kind="stable",
            )
            if len(ranked) > limit:
                selected_threshold = float(ranked.iloc[limit - 1]["break_score"])
                breaks = ranked.head(limit).sort_index()
            else:
                selected_threshold = float(ranked["break_score"].min())
                breaks = ranked.sort_index()

        candidates_out = self._to_break_table(candidates)
        breaks_out = self._to_break_table(breaks)

        score_stats = self._score_distribution_stats(candidates["break_score"] if not candidates.empty else pd.Series(dtype=float))

        median_abs_return = bars["abs_price_return_pct"].median(skipna=True)
        median_vol_ratio = bars["volume_ratio"].median(skipna=True)

        summary = {
            "interval": self.config.interval,
            "n_intervals": int(len(bars)),
            "n_breaks": int(len(breaks_out)),
            "n_candidates_before_cap": int(len(candidates_out)),
            "selected_outcome": selected_outcome,
            "price_change_pct": float(self.config.price_change_pct),
            "strong_price_change_pct": float(self.config.strong_price_change_pct),
            "volume_surge_ratio": float(self.config.volume_surge_ratio),
            "volume_substantial_ratio": float(self.config.volume_substantial_ratio),
            "low_volume_ratio": float(self.config.low_volume_ratio),
            "min_break_score": None if self.config.min_break_score is None else float(self.config.min_break_score),
            "max_breaks": None if self.config.max_breaks is None else int(self.config.max_breaks),
            "break_radius": self.config.break_radius,
            "score_threshold_selected": selected_threshold,
            "median_abs_return_pct": float(median_abs_return) if pd.notna(median_abs_return) else float("nan"),
            "median_volume_ratio": float(median_vol_ratio) if pd.notna(median_vol_ratio) else float("nan"),
            **score_stats,
        }

        return StructuralBreakResult(
            interval=self.config.interval,
            bars=bars,
            breaks=breaks_out,
            candidates=candidates_out,
            score_threshold=selected_threshold,
            selected_outcome=selected_outcome,
            summary=summary,
        )

    def plot_break_score_distribution(
        self,
        result: StructuralBreakResult,
        *,
        bins: int = 60,
        figsize: tuple[int, int] = (10, 4),
    ) -> tuple[Any, Any]:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        cand = result.candidates

        if cand.empty:
            ax.text(0.5, 0.5, "No candidate breaks", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Break Score Distribution")
            fig.tight_layout()
            return fig, ax

        scores = pd.to_numeric(cand["score"], errors="coerce").dropna()
        if scores.empty:
            ax.text(0.5, 0.5, "No numeric scores", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Break Score Distribution")
            fig.tight_layout()
            return fig, ax

        upper = float(scores.quantile(0.995))
        if not np.isfinite(upper) or upper <= 0:
            upper = float(scores.max())
        plot_scores = scores.clip(upper=upper)
        tail_pct = float((scores > upper).mean() * 100.0)

        ax.hist(plot_scores, bins=int(bins), color="tab:purple", alpha=0.75, edgecolor="white")
        ax.set_yscale("log")
        ax.set_title(
            f"Break Score Distribution | candidates={len(scores)} | selected={len(result.breaks)}"
        )
        ax.set_xlabel("break_score (clipped)")
        ax.set_ylabel("Count (log scale)")

        q90 = float(scores.quantile(0.90))
        q95 = float(scores.quantile(0.95))
        q99 = float(scores.quantile(0.99))
        ax.axvline(q90, color="tab:blue", linestyle="--", linewidth=1.2, label=f"q90={q90:.2f}")
        ax.axvline(q95, color="tab:green", linestyle="--", linewidth=1.2, label=f"q95={q95:.2f}")
        ax.axvline(q99, color="tab:orange", linestyle="--", linewidth=1.2, label=f"q99={q99:.2f}")

        if result.score_threshold is not None:
            ax.axvline(
                float(result.score_threshold),
                color="tab:red",
                linestyle="-",
                linewidth=1.8,
                label=f"selected_threshold={float(result.score_threshold):.2f}",
            )

        ax.text(
            0.99,
            0.95,
            f">p99.5={tail_pct:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.7},
        )
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig, ax

    def plot_breaks_overview(
        self,
        result: StructuralBreakResult,
        *,
        annotate: bool = False,
        figsize: tuple[int, int] = (14, 6),
    ) -> tuple[Any, Any, Any]:
        import matplotlib.pyplot as plt

        bars = result.bars
        breaks = result.breaks

        fig, ax_price = plt.subplots(figsize=figsize)
        ax_volume = ax_price.twinx()

        x = bars.index
        raw_price = bars["price_plot"].astype(float)
        price_span = max(2, int(self.config.plot_price_smoothing_span))
        price_smoothed = raw_price.ewm(span=price_span, adjust=False, min_periods=1).mean()

        volume_raw = bars["volume"].fillna(0.0).astype(float)
        volume_span = max(2, int(self.config.plot_volume_smoothing_span))
        volume_smoothed = volume_raw.ewm(span=volume_span, adjust=False, min_periods=1).mean()

        baseline = bars["volume_baseline"].astype(float)
        baseline_smoothed = baseline.ewm(span=volume_span, adjust=False, min_periods=1).mean()

        q = float(self.config.plot_volume_clip_quantile)
        q = min(max(q, 0.90), 1.0)
        vol_upper = float(volume_smoothed.quantile(q))
        if not np.isfinite(vol_upper) or vol_upper <= 0:
            vol_upper = float(volume_smoothed.max()) if len(volume_smoothed) else 0.0
        volume_plot = volume_smoothed.clip(upper=vol_upper) if vol_upper > 0 else volume_smoothed
        baseline_plot = baseline_smoothed.clip(upper=vol_upper) if vol_upper > 0 else baseline_smoothed

        if self.config.plot_show_raw_price:
            ax_price.plot(
                x,
                raw_price,
                color="tab:green",
                linewidth=0.8,
                alpha=float(self.config.plot_raw_price_alpha),
                label="Price (raw)",
            )
        ax_price.plot(x, price_smoothed, color="tab:green", linewidth=1.9, label=f"Price (EWMA span={price_span})")

        ax_volume.fill_between(
            x,
            0.0,
            volume_plot,
            color="tab:blue",
            alpha=0.18,
            step="pre",
            label=f"Volume (EWMA span={volume_span})",
        )

        if baseline_plot.notna().any():
            ax_volume.plot(
                x,
                baseline_plot,
                color="tab:blue",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                label="Volume baseline (smoothed)",
            )

        if not breaks.empty:
            plot_breaks = breaks.copy().set_index("break_timestamp")
            colors = {
                "volume_surge_and_price_shift": "tab:red",
                "substantial_volume_and_strong_price_shift": "tab:orange",
            }
            labels = {
                "volume_surge_and_price_shift": "Surge + Price Shift",
                "substantial_volume_and_strong_price_shift": "Substantial Vol + Strong Price Shift",
            }

            for reason, color in colors.items():
                sub = plot_breaks[plot_breaks["reason"] == reason]
                if sub.empty:
                    continue
                marker_size = 30.0 + 18.0 * np.clip(sub["score"].astype(float), 0.0, 8.0)
                y_vals = price_smoothed.reindex(sub.index).ffill().bfill()
                ax_price.scatter(
                    sub.index,
                    y_vals,
                    s=marker_size,
                    color=color,
                    edgecolors="black",
                    linewidths=0.4,
                    alpha=0.95,
                    label=labels.get(reason, reason),
                    zorder=6,
                )

            if annotate:
                ann = plot_breaks.sort_values("score", ascending=False).head(int(self.config.max_annotations))
                for ts, row in ann.iterrows():
                    txt = f"{float(row['price_return_pct']):+.1f}% | x{float(row['volume_ratio']):.1f}"
                    y = float(price_smoothed.reindex([ts]).iloc[0]) if ts in price_smoothed.index else float(row["price_end"])
                    ax_price.annotate(
                        txt,
                        xy=(ts, y),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=30,
                    )

        ax_price.set_title(
            f"Structural Breaks ({result.interval}) | breaks={int(len(breaks))}"
        )
        ax_price.set_xlabel("Time (UTC)")
        ax_price.set_ylabel("Price")
        ax_volume.set_ylabel("Volume")

        if vol_upper > 0:
            ax_volume.set_ylim(0, vol_upper * 1.05)

        h1, l1 = ax_price.get_legend_handles_labels()
        h2, l2 = ax_volume.get_legend_handles_labels()
        ax_price.legend(h1 + h2, l1 + l2, loc="upper left")

        fig.tight_layout()
        return fig, ax_price, ax_volume

    @staticmethod
    def _to_break_table(rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty:
            return pd.DataFrame(
                columns=[
                    "break_timestamp",
                    "reason",
                    "score",
                    "price_end",
                    "price_return_pct",
                    "abs_price_return_pct",
                    "volume",
                    "volume_baseline",
                    "volume_ratio",
                    "n_trades",
                ]
            )

        out = rows[
            [
                "price_end",
                "price_return_pct",
                "abs_price_return_pct",
                "volume",
                "volume_baseline",
                "volume_ratio",
                "n_trades",
                "break_reason",
                "break_score",
            ]
        ].copy()
        out = out.rename(columns={"break_reason": "reason", "break_score": "score"})
        out.insert(0, "break_timestamp", out.index)
        out = out.sort_values("break_timestamp").reset_index(drop=True)
        return out

    @staticmethod
    def _score_distribution_stats(scores: pd.Series) -> dict[str, float]:
        if scores.empty:
            return {
                "score_min": float("nan"),
                "score_q50": float("nan"),
                "score_q75": float("nan"),
                "score_q90": float("nan"),
                "score_q95": float("nan"),
                "score_q99": float("nan"),
                "score_max": float("nan"),
            }
        s = pd.to_numeric(scores, errors="coerce").dropna()
        if s.empty:
            return {
                "score_min": float("nan"),
                "score_q50": float("nan"),
                "score_q75": float("nan"),
                "score_q90": float("nan"),
                "score_q95": float("nan"),
                "score_q99": float("nan"),
                "score_max": float("nan"),
            }
        return {
            "score_min": float(s.min()),
            "score_q50": float(s.quantile(0.50)),
            "score_q75": float(s.quantile(0.75)),
            "score_q90": float(s.quantile(0.90)),
            "score_q95": float(s.quantile(0.95)),
            "score_q99": float(s.quantile(0.99)),
            "score_max": float(s.max()),
        }

    def _build_bars(self, trades: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
        clean, selected_outcome = self._clean_trades(trades)
        work = clean[["timestamp_utc", "price", "size"]].copy()
        work["notional"] = work["price"] * work["size"]
        work = work.set_index("timestamp_utc").sort_index()

        grp = work.resample(self.config.interval)

        bars = pd.DataFrame(index=grp["price"].size().index)
        bars["n_trades"] = grp["price"].size().astype("int64")
        bars["volume"] = grp["size"].sum().fillna(0.0)
        bars["notional"] = grp["notional"].sum().fillna(0.0)
        bars["price_start"] = grp["price"].first()
        bars["price_end"] = grp["price"].last()

        # For plotting a continuous price line.
        bars["price_plot"] = bars["price_end"].ffill()

        bars["price_return_pct"] = np.where(
            bars["price_start"].notna() & bars["price_end"].notna() & (bars["price_start"] != 0),
            (bars["price_end"] / bars["price_start"] - 1.0) * 100.0,
            np.nan,
        )
        bars["abs_price_return_pct"] = np.abs(bars["price_return_pct"])

        # Baseline uses historical data only (shift by 1 interval).
        shifted_vol = bars["volume"].shift(1)
        bars["volume_baseline"] = shifted_vol.rolling(
            window=int(self.config.baseline_window),
            min_periods=int(self.config.baseline_min_periods),
        ).median()

        bars["volume_q50"] = shifted_vol.rolling(
            window=int(self.config.baseline_window),
            min_periods=int(self.config.baseline_min_periods),
        ).quantile(0.50)

        bars["volume_q75"] = shifted_vol.rolling(
            window=int(self.config.baseline_window),
            min_periods=int(self.config.baseline_min_periods),
        ).quantile(0.75)

        bars["volume_ratio"] = bars["volume"] / bars["volume_baseline"]
        bars["volume_ratio"] = bars["volume_ratio"].replace([np.inf, -np.inf], np.nan)

        # Fallback ratio for periods where baseline is unavailable.
        fallback_ratio = np.where(
            bars["volume_q50"].to_numpy(dtype=float) > 0,
            bars["volume"].to_numpy(dtype=float) / bars["volume_q50"].to_numpy(dtype=float),
            0.0,
        )
        bars["volume_ratio_signal"] = np.where(
            bars["volume_ratio"].notna(),
            bars["volume_ratio"],
            fallback_ratio,
        )

        vol_substantial = (
            (bars["volume_ratio_signal"] >= float(self.config.volume_substantial_ratio))
            | (bars["volume"] >= bars["volume_q75"].fillna(np.inf))
        )
        if self.config.min_volume_absolute is not None:
            vol_substantial = vol_substantial | (bars["volume"] >= float(self.config.min_volume_absolute))

        bars["volume_substantial"] = vol_substantial.fillna(False)

        bars["volume_low"] = (
            (bars["volume"] <= 0)
            | (
                (bars["volume_ratio_signal"] < float(self.config.low_volume_ratio))
                & (bars["volume"] < bars["volume_q50"].fillna(np.inf))
            )
        ).fillna(True)

        return bars, selected_outcome

    def _merge_neighbor_breaks(self, bars: pd.DataFrame, breaks: pd.DataFrame) -> pd.DataFrame:
        all_positions = np.flatnonzero(bars["is_break"].to_numpy(dtype=bool))
        if len(all_positions) <= 1:
            return breaks

        keep_positions: list[int] = []
        current_cluster: list[int] = [int(all_positions[0])]
        max_gap = max(1, int(self.config.merge_gap_intervals))

        for pos in all_positions[1:]:
            pos_i = int(pos)
            if pos_i - current_cluster[-1] <= max_gap:
                current_cluster.append(pos_i)
            else:
                best = max(current_cluster, key=lambda p: float(bars.iloc[p]["break_score"]))
                keep_positions.append(best)
                current_cluster = [pos_i]

        best = max(current_cluster, key=lambda p: float(bars.iloc[p]["break_score"]))
        keep_positions.append(best)

        keep_index = bars.iloc[keep_positions].index
        return breaks.loc[keep_index].copy()

    def _apply_break_radius_nms(self, breaks: pd.DataFrame) -> pd.DataFrame:
        radius = self._parse_break_radius()
        if radius is None or breaks.empty:
            return breaks

        ranked = breaks.sort_values(
            ["break_score", "abs_price_return_pct", "volume_ratio_signal"],
            ascending=[False, False, False],
            kind="stable",
        )

        selected_index: list[pd.Timestamp] = []
        selected_times: list[pd.Timestamp] = []
        for ts in ranked.index:
            if all(abs(ts - kept_ts) > radius for kept_ts in selected_times):
                selected_index.append(ts)
                selected_times.append(ts)

        return breaks.loc[selected_index].sort_index().copy()

    def _parse_break_radius(self) -> pd.Timedelta | None:
        value = self.config.break_radius
        if value is None:
            return None
        try:
            td = pd.to_timedelta(value)
        except Exception:
            return None
        if pd.isna(td) or td <= pd.Timedelta(0):
            return None
        return td

    def _clean_trades(self, trades: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
        required = {"timestamp_utc", "price", "size"}
        missing = required.difference(trades.columns)
        if missing:
            raise ValueError(f"Missing required trade columns: {sorted(missing)}")

        out = trades.copy()
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
        out["size"] = pd.to_numeric(out["size"], errors="coerce")

        selected_outcome: str | None = None
        if "outcome" in out.columns:
            out["outcome"] = out["outcome"].astype("string").str.strip()
            out["outcome"] = out["outcome"].replace({"": pd.NA})

            if self.config.outcome is not None:
                wanted = str(self.config.outcome).strip().lower()
                matched = out[out["outcome"].astype("string").str.lower() == wanted]
                if matched.empty:
                    available = sorted({str(v) for v in out["outcome"].dropna().unique().tolist()})
                    raise ValueError(
                        f"Configured outcome={self.config.outcome!r} not found. Available outcomes: {available}"
                    )
                out = matched
                selected_outcome = str(out["outcome"].dropna().iloc[0])
            elif self.config.auto_select_outcome:
                with_outcome = out.dropna(subset=["outcome"]).copy()
                if not with_outcome.empty:
                    by_size = (
                        with_outcome.groupby("outcome", dropna=True)["size"].sum().sort_values(ascending=False)
                    )
                    if not by_size.empty:
                        selected_outcome = str(by_size.index[0])
                        out = with_outcome[with_outcome["outcome"] == selected_outcome]

        out = out.dropna(subset=["timestamp_utc", "price", "size"])
        out = out.sort_values("timestamp_utc").reset_index(drop=True)

        if out.empty:
            raise ValueError("No valid trades after cleaning.")
        return out, selected_outcome
