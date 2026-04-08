from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_utils import (
    DEFAULT_DB_PATH,
    add_time_features,
    build_multi_horizon_terminal_dataset,
    build_repricing_dataset,
    connect,
    default_feature_columns,
    load_eligible_markets,
    load_probabilities_for_markets,
    rolling_time_splits,
    summarize_metric_frame,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Polymarket benchmark suite and export reproducible results.")
    p.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    p.add_argument("--domain", default="geopolitics")
    p.add_argument("--terminal-max-markets", type=int, default=5000)
    p.add_argument("--repricing-max-markets", type=int, default=5000)
    p.add_argument("--output-dir", default="benchmarks/results")
    p.add_argument("--terminal-staleness-hours", type=float, default=12.0)
    p.add_argument("--terminal-horizons-hours", nargs="+", type=int, default=[24, 72, 168])
    p.add_argument("--trust-horizon-hours", type=int, default=24)
    p.add_argument("--repricing-future-hours", type=int, default=24)
    p.add_argument("--repricing-lookback-hours", type=int, default=24)
    p.add_argument("--repricing-sample-every-hours", type=int, default=12)
    p.add_argument("--repricing-move-threshold", type=float, default=0.15)
    return p.parse_args()


def safe_auc(y_true: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, pred))


def clipped(pred: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(pred, dtype=float), eps, 1.0 - eps)


def horizon_prior_pred(frame: pd.DataFrame, train_df: pd.DataFrame) -> np.ndarray:
    prior = train_df.groupby("horizon_name")["target"].mean().to_dict()
    return clipped(frame["horizon_name"].map(prior).fillna(train_df["target"].mean()).to_numpy(dtype=float))


def shrinkage_pred(frame: pd.DataFrame, train_df: pd.DataFrame, *, alpha: float) -> np.ndarray:
    prior = horizon_prior_pred(frame, train_df)
    activity = frame["lookback_24h_trade_count_sum"].fillna(0.0).to_numpy(dtype=float)
    weight = activity / (activity + float(alpha))
    pred = weight * frame["market_price_baseline"].to_numpy(dtype=float) + (1.0 - weight) * prior
    return clipped(pred)


def volatility_baseline(frame: pd.DataFrame) -> np.ndarray:
    score = 2.0 * frame["recent_abs_move_mean"].to_numpy(dtype=float) + 2.0 * frame["recent_volatility"].to_numpy(dtype=float)
    return clipped(np.clip(score, 0.0, 1.0))


def repricing_heuristic(frame: pd.DataFrame, *, beta: float) -> np.ndarray:
    vol = frame["recent_volatility"].to_numpy(dtype=float)
    abs_move = frame["recent_abs_move_mean"].to_numpy(dtype=float)
    uncertainty = np.clip(1.0 - 2.0 * frame["confidence_margin"].to_numpy(dtype=float), 0.0, 1.0)
    score = vol + abs_move + float(beta) * uncertainty * vol
    return clipped(np.clip(score, 0.0, 1.0))


def terminal_model_suite(feature_cols: list[str]) -> dict[str, Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], np.ndarray]]:
    logistic = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    hgb = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, random_state=42)),
        ]
    )

    def _fit_predict_sklearn(model: Pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        model.fit(train_df[feature_cols], train_df["target"])
        return clipped(model.predict_proba(test_df[feature_cols])[:, 1])

    def _market_price(_train_df: pd.DataFrame, test_df: pd.DataFrame, _all_train: pd.DataFrame) -> np.ndarray:
        return clipped(test_df["market_price_baseline"].to_numpy(dtype=float))

    def _horizon_prior(train_df: pd.DataFrame, test_df: pd.DataFrame, _all_train: pd.DataFrame) -> np.ndarray:
        return horizon_prior_pred(test_df, train_df)

    def _shrinkage(train_df: pd.DataFrame, test_df: pd.DataFrame, _all_train: pd.DataFrame) -> np.ndarray:
        alpha_grid = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
        y = train_df["target"].to_numpy(dtype=float)
        best_alpha = min(alpha_grid, key=lambda alpha: log_loss(y, shrinkage_pred(train_df, train_df, alpha=alpha), labels=[0, 1]))
        return shrinkage_pred(test_df, train_df, alpha=best_alpha)

    return {
        "market_price": _market_price,
        "horizon_prior": _horizon_prior,
        "liquidity_aware_shrinkage": _shrinkage,
        "logistic_regression": lambda train_df, test_df, _all_train: _fit_predict_sklearn(logistic, train_df, test_df),
        "hist_gradient_boosting": lambda train_df, test_df, _all_train: _fit_predict_sklearn(hgb, train_df, test_df),
    }


def evaluate_terminal(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = default_feature_columns(dataset)
    models = terminal_model_suite(feature_cols)
    rows: list[dict[str, object]] = []

    for train_df, test_df, meta in rolling_time_splits(dataset, time_col="end_date", n_splits=4, min_train_fraction=0.5):
        y_test = test_df["target"].to_numpy(dtype=float)
        for model_name, predictor in models.items():
            pred = predictor(train_df, test_df, train_df)
            base_row = {
                "benchmark": "terminal_forecasting",
                "fold": meta["fold"],
                "model": model_name,
                "rows": len(test_df),
                "log_loss": float(log_loss(y_test, pred, labels=[0, 1])),
                "brier": float(brier_score_loss(y_test, pred)),
                "roc_auc": safe_auc(y_test, pred),
            }
            rows.append(base_row)
            for horizon_name, frame in test_df.assign(_pred=pred).groupby("horizon_name"):
                y_h = frame["target"].to_numpy(dtype=float)
                p_h = frame["_pred"].to_numpy(dtype=float)
                rows.append(
                    {
                        "benchmark": "terminal_forecasting",
                        "fold": meta["fold"],
                        "model": model_name,
                        "horizon_name": horizon_name,
                        "rows": len(frame),
                        "log_loss": float(log_loss(y_h, p_h, labels=[0, 1])),
                        "brier": float(brier_score_loss(y_h, p_h)),
                        "roc_auc": safe_auc(y_h, p_h),
                    }
                )

    metrics = pd.DataFrame(rows)
    summary = summarize_metric_frame(
        metrics,
        group_cols=["benchmark", "model", "horizon_name"] if "horizon_name" in metrics.columns else ["benchmark", "model"],
        metric_cols=["log_loss", "brier", "roc_auc"],
    )
    return metrics, summary


def trust_policy_scores(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, np.ndarray]:
    regressor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, random_state=42)),
        ]
    )
    regressor.fit(train_df[feature_cols], train_df["market_abs_error"])
    predicted_error = regressor.predict(test_df[feature_cols])

    margin_score = test_df["confidence_margin"].to_numpy(dtype=float)
    margin_x_trade_share = margin_score * (1.0 + test_df["lookback_24h_observed_trade_share"].fillna(0.0).to_numpy(dtype=float))
    low_vol_high_activity = margin_score * (1.0 + test_df["lookback_24h_observed_trade_share"].fillna(0.0).to_numpy(dtype=float)) / (
        1.0 + 25.0 * test_df["lookback_24h_volatility"].fillna(0.0).to_numpy(dtype=float)
    )

    return {
        "learned_trust": -np.asarray(predicted_error, dtype=float),
        "confidence_margin": margin_score,
        "margin_x_trade_share": margin_x_trade_share,
        "low_vol_high_activity": low_vol_high_activity,
        "oracle": -test_df["market_abs_error"].to_numpy(dtype=float),
    }


def selective_metrics(frame: pd.DataFrame, *, score: np.ndarray, coverage: float) -> dict[str, float]:
    ordered = frame.assign(_score=score).sort_values("_score", ascending=False).reset_index(drop=True)
    n = max(1, int(len(ordered) * float(coverage)))
    subset = ordered.iloc[:n]
    return {
        "coverage": float(coverage),
        "kept_rows": int(n),
        "mean_abs_error": float(subset["market_abs_error"].mean()),
        "mean_log_loss": float(subset["market_log_loss"].mean()),
    }


def evaluate_trust(dataset: pd.DataFrame, *, trust_horizon_hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = dataset.loc[dataset["horizon_hours"] == int(trust_horizon_hours)].reset_index(drop=True)
    feature_cols = default_feature_columns(work)
    rows: list[dict[str, object]] = []
    for train_df, test_df, meta in rolling_time_splits(work, time_col="end_date", n_splits=4, min_train_fraction=0.5):
        scores = trust_policy_scores(train_df, test_df, feature_cols)
        for policy_name, score in scores.items():
            for coverage in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
                metrics = selective_metrics(test_df, score=score, coverage=coverage)
                rows.append(
                    {
                        "benchmark": "trustworthiness",
                        "fold": meta["fold"],
                        "policy": policy_name,
                        **metrics,
                    }
                )
    metrics = pd.DataFrame(rows)
    summary = summarize_metric_frame(
        metrics,
        group_cols=["benchmark", "policy", "coverage"],
        metric_cols=["mean_abs_error", "mean_log_loss"],
    )
    return metrics, summary


def repricing_model_suite(feature_cols: list[str]) -> dict[str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]]:
    logistic = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    hgb = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, random_state=42)),
        ]
    )

    def _fit_predict(model: Pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        model.fit(train_df[feature_cols], train_df["target"])
        return clipped(model.predict_proba(test_df[feature_cols])[:, 1])

    def _heuristic(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        beta_grid = [0.0, 0.5, 1.0, 2.0, 4.0]
        best_beta = max(
            beta_grid,
            key=lambda beta: average_precision_score(train_df["target"].to_numpy(), repricing_heuristic(train_df, beta=beta)),
        )
        return repricing_heuristic(test_df, beta=best_beta)

    return {
        "volatility_baseline": lambda _train_df, test_df: volatility_baseline(test_df),
        "volatility_uncertainty_heuristic": _heuristic,
        "logistic_regression": lambda train_df, test_df: _fit_predict(logistic, train_df, test_df),
        "hist_gradient_boosting": lambda train_df, test_df: _fit_predict(hgb, train_df, test_df),
    }


def evaluate_repricing(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = default_feature_columns(dataset, exclude=["future_horizon_hours"])
    models = repricing_model_suite(feature_cols)
    rows: list[dict[str, object]] = []
    for train_df, test_df, meta in rolling_time_splits(dataset, time_col="timestamp_utc", n_splits=4, min_train_fraction=0.5):
        y_test = test_df["target"].to_numpy(dtype=float)
        for model_name, predictor in models.items():
            pred = predictor(train_df, test_df)
            rows.append(
                {
                    "benchmark": "repricing",
                    "fold": meta["fold"],
                    "model": model_name,
                    "rows": len(test_df),
                    "roc_auc": safe_auc(y_test, pred),
                    "average_precision": float(average_precision_score(y_test, pred)),
                    "log_loss": float(log_loss(y_test, pred, labels=[0, 1])),
                    "brier": float(brier_score_loss(y_test, pred)),
                }
            )
    metrics = pd.DataFrame(rows)
    summary = summarize_metric_frame(
        metrics,
        group_cols=["benchmark", "model"],
        metric_cols=["roc_auc", "average_precision", "log_loss", "brier"],
    )
    return metrics, summary


def write_report(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    market_counts: dict[str, int],
    terminal_summary: pd.DataFrame,
    trust_summary: pd.DataFrame,
    repricing_summary: pd.DataFrame,
) -> None:
    terminal_overall = terminal_summary.loc[terminal_summary["horizon_name"].isna()].sort_values("log_loss_mean")
    terminal_by_horizon = terminal_summary.loc[terminal_summary["horizon_name"].notna()].sort_values(["horizon_name", "log_loss_mean"])
    trust_best = trust_summary.sort_values(["coverage", "mean_log_loss_mean"])
    repricing_best = repricing_summary.sort_values("average_precision_mean", ascending=False)

    def _markdown_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for row in df.itertuples(index=False):
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append(f"{value:.6f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines = [
        "# Benchmark Report",
        "",
        "## Configuration",
        f"- Domain: `{args.domain}`",
        f"- Database: `{Path(args.db_path).resolve()}`",
        f"- Terminal horizons: `{args.terminal_horizons_hours}` hours",
        f"- Trust horizon: `{args.trust_horizon_hours}` hours",
        f"- Repricing horizon: `{args.repricing_future_hours}` hours",
        f"- Repricing move threshold: `{args.repricing_move_threshold}`",
        "",
        "## Dataset Sizes",
        f"- Eligible terminal markets: `{market_counts['terminal_markets']}`",
        f"- Eligible repricing markets: `{market_counts['repricing_markets']}`",
        f"- Terminal benchmark rows: `{market_counts['terminal_rows']}`",
        f"- Trust benchmark rows: `{market_counts['trust_rows']}`",
        f"- Repricing benchmark rows: `{market_counts['repricing_rows']}`",
        "",
        "## Terminal Forecasting",
        _markdown_table(terminal_overall[["model", "log_loss_mean", "brier_mean", "roc_auc_mean"]].head(5)),
        "",
        "### Terminal Forecasting By Horizon",
        _markdown_table(terminal_by_horizon[["horizon_name", "model", "log_loss_mean", "brier_mean", "roc_auc_mean"]]),
        "",
        "## Trustworthiness",
        _markdown_table(trust_best[["policy", "coverage", "mean_abs_error_mean", "mean_log_loss_mean"]]),
        "",
        "## Repricing",
        _markdown_table(repricing_best[["model", "average_precision_mean", "roc_auc_mean", "log_loss_mean", "brier_mean"]]),
        "",
    ]
    (out_dir / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve() / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = connect(args.db_path)

    terminal_markets = load_eligible_markets(
        conn,
        domain=args.domain,
        max_markets=args.terminal_max_markets,
        min_probability_rows=24 * 12,
    )
    terminal_probabilities = load_probabilities_for_markets(conn, terminal_markets["market_id"].tolist())
    terminal_dataset = build_multi_horizon_terminal_dataset(
        terminal_markets,
        terminal_probabilities,
        horizons_hours=args.terminal_horizons_hours,
        max_snapshot_staleness_hours=args.terminal_staleness_hours,
    )
    terminal_dataset = add_time_features(terminal_dataset)

    repricing_markets = load_eligible_markets(
        conn,
        domain=args.domain,
        max_markets=args.repricing_max_markets,
        min_probability_rows=14 * 24 * 12,
    )
    repricing_probabilities = load_probabilities_for_markets(conn, repricing_markets["market_id"].tolist())
    repricing_dataset = build_repricing_dataset(
        repricing_markets,
        repricing_probabilities,
        future_horizon_hours=args.repricing_future_hours,
        lookback_hours=args.repricing_lookback_hours,
        sample_every_hours=args.repricing_sample_every_hours,
        move_threshold=args.repricing_move_threshold,
    )

    terminal_metrics, terminal_summary = evaluate_terminal(terminal_dataset)
    trust_metrics, trust_summary = evaluate_trust(terminal_dataset, trust_horizon_hours=args.trust_horizon_hours)
    repricing_metrics, repricing_summary = evaluate_repricing(repricing_dataset)

    terminal_dataset.to_csv(out_dir / "terminal_dataset.csv", index=False)
    repricing_dataset.to_csv(out_dir / "repricing_dataset.csv", index=False)
    terminal_metrics.to_csv(out_dir / "terminal_metrics.csv", index=False)
    terminal_summary.to_csv(out_dir / "terminal_summary.csv", index=False)
    trust_metrics.to_csv(out_dir / "trust_metrics.csv", index=False)
    trust_summary.to_csv(out_dir / "trust_summary.csv", index=False)
    repricing_metrics.to_csv(out_dir / "repricing_metrics.csv", index=False)
    repricing_summary.to_csv(out_dir / "repricing_summary.csv", index=False)

    config_payload = {
        "db_path": str(Path(args.db_path).resolve()),
        "domain": args.domain,
        "terminal_max_markets": args.terminal_max_markets,
        "repricing_max_markets": args.repricing_max_markets,
        "terminal_horizons_hours": args.terminal_horizons_hours,
        "trust_horizon_hours": args.trust_horizon_hours,
        "repricing_future_hours": args.repricing_future_hours,
        "repricing_lookback_hours": args.repricing_lookback_hours,
        "repricing_sample_every_hours": args.repricing_sample_every_hours,
        "repricing_move_threshold": args.repricing_move_threshold,
        "terminal_staleness_hours": args.terminal_staleness_hours,
    }
    (out_dir / "benchmark_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    market_counts = {
        "terminal_markets": len(terminal_markets),
        "repricing_markets": len(repricing_markets),
        "terminal_rows": len(terminal_dataset),
        "trust_rows": len(terminal_dataset.loc[terminal_dataset["horizon_hours"] == int(args.trust_horizon_hours)]),
        "repricing_rows": len(repricing_dataset),
    }
    write_report(
        out_dir,
        args=args,
        market_counts=market_counts,
        terminal_summary=terminal_summary,
        trust_summary=trust_summary,
        repricing_summary=repricing_summary,
    )

    summary_payload = {
        "market_counts": market_counts,
        "terminal_best_model": terminal_summary.loc[terminal_summary["horizon_name"].isna()].sort_values("log_loss_mean").iloc[0].to_dict(),
        "repricing_best_model": repricing_summary.sort_values("average_precision_mean", ascending=False).iloc[0].to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")

    print(f"Benchmark results written to {out_dir}")


if __name__ == "__main__":
    main()
