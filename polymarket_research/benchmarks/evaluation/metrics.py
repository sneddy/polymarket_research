"""Lightweight metrics and evaluator primitives for frozen benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def binary_log_loss_values(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute row-level binary log loss with clipped probabilities."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-6) -> float:
    """Compute aggregate binary log loss."""
    return float(np.mean(binary_log_loss_values(y_true, y_prob, eps=eps)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute aggregate Brier score."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute ROC AUC without depending on sklearn."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for pos_score in pos:
        wins += float(np.sum(pos_score > neg))
        wins += 0.5 * float(np.sum(pos_score == neg))
    return float(wins / (len(pos) * len(neg)))


def prediction_frame(
    predictions: pd.DataFrame | pd.Series,
    *,
    id_columns: tuple[str, ...] = ("example_id",),
    value_col: str = "pred_prob",
) -> pd.DataFrame:
    """Normalize user predictions into an id-columns-plus-value frame."""
    if isinstance(predictions, pd.Series):
        if len(id_columns) != 1:
            raise ValueError("Series predictions only support benchmarks with a single id column.")
        pred_frame = predictions.rename(value_col).reset_index().rename(columns={"index": id_columns[0]})
    else:
        pred_frame = predictions.copy()

    expected_columns = [*id_columns, value_col]
    missing_columns = [column for column in expected_columns if column not in pred_frame.columns]
    if missing_columns:
        missing_text = ", ".join(repr(column) for column in missing_columns)
        raise ValueError(f"predictions must provide columns {missing_text}.")

    out = pred_frame.loc[:, expected_columns].copy()
    for column in id_columns:
        if "timestamp" in column:
            out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    return out


def evaluate_binary_predictions(
    gold: pd.DataFrame,
    predictions: pd.DataFrame | pd.Series,
    *,
    split: str,
    group_col: str | None = None,
    id_columns: tuple[str, ...] = ("example_id",),
    reference_prob_col: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate binary probabilistic predictions against a gold table."""
    pred_frame = prediction_frame(predictions, id_columns=id_columns, value_col="pred_prob")
    scored = gold.merge(pred_frame, on=list(id_columns), how="inner")
    if len(scored) != len(gold):
        if len(id_columns) == 1:
            key = id_columns[0]
            missing = sorted(set(gold[key]) - set(scored[key]))
        else:
            missing = sorted(
                set(gold.loc[:, id_columns].itertuples(index=False, name=None))
                - set(scored.loc[:, id_columns].itertuples(index=False, name=None))
            )
        raise ValueError(f"Missing predictions for {len(missing)} examples.")

    def metric_row(frame: pd.DataFrame) -> dict[str, float]:
        y_true = frame["label"].to_numpy(dtype=float)
        y_prob = frame["pred_prob"].to_numpy(dtype=float)
        row = {
            "rows": int(len(frame)),
            "log_loss": binary_log_loss(y_true, y_prob),
        }
        if reference_prob_col is not None:
            if reference_prob_col not in frame.columns:
                raise ValueError(f"gold table must contain reference probability column {reference_prob_col!r}.")
            reference_prob = frame[reference_prob_col].to_numpy(dtype=float)
            model_losses = binary_log_loss_values(y_true, y_prob)
            reference_losses = binary_log_loss_values(y_true, reference_prob)
            reference_loss_sum = float(np.sum(reference_losses))
            row["delta_log_loss_vs_market"] = float(np.mean(reference_losses - model_losses))
            row["market_skill_log_loss"] = (
                float("nan")
                if reference_loss_sum <= 0.0
                else float(1.0 - (float(np.sum(model_losses)) / reference_loss_sum))
            )
        row["brier_score"] = brier_score(y_true, y_prob)
        row["roc_auc"] = roc_auc(y_true, y_prob)
        return row

    overall_row = {"split": split}
    overall_row.update(metric_row(scored))
    overall = pd.DataFrame([overall_row])

    results = {"overall": overall}
    if group_col is None:
        return results

    grouped_rows: list[dict[str, Any]] = []
    for group_value, frame in scored.groupby(group_col, sort=True):
        row = {
            "split": split,
            group_col: int(group_value) if pd.notna(group_value) else group_value,
        }
        row.update(metric_row(frame))
        grouped_rows.append(row)
    results[f"by_{group_col.removesuffix('_hours')}"] = pd.DataFrame(grouped_rows)
    return results


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute simple classification accuracy."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if len(y_true_arr) == 0:
        return float("nan")
    return float(np.mean(y_true_arr == y_pred_arr))


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro F1 without sklearn."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = sorted(set(y_true_arr.tolist()) | set(y_pred_arr.tolist()))
    if not labels:
        return float("nan")

    f1_values: list[float] = []
    for label in labels:
        true_positive = float(np.sum((y_true_arr == label) & (y_pred_arr == label)))
        false_positive = float(np.sum((y_true_arr != label) & (y_pred_arr == label)))
        false_negative = float(np.sum((y_true_arr == label) & (y_pred_arr != label)))
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        if precision == 0.0 and recall == 0.0:
            f1_values.append(0.0)
        else:
            f1_values.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1_values))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute lightweight regression diagnostics."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    error = y_pred_arr - y_true_arr
    return {
        "mae": float(np.mean(np.abs(error))) if len(error) else float("nan"),
        "rmse": float(np.sqrt(np.mean(error**2))) if len(error) else float("nan"),
        "bias": float(np.mean(error)) if len(error) else float("nan"),
    }


def evaluate_multiclass_predictions(
    gold: pd.DataFrame,
    predictions: pd.DataFrame | pd.Series,
    *,
    split: str,
    group_col: str | None = None,
    id_columns: tuple[str, ...] = ("example_id",),
) -> dict[str, pd.DataFrame]:
    """Evaluate discrete class predictions against a gold table."""
    pred_frame = prediction_frame(predictions, id_columns=id_columns, value_col="pred_label")
    scored = gold.merge(pred_frame, on=list(id_columns), how="inner")
    if len(scored) != len(gold):
        if len(id_columns) == 1:
            key = id_columns[0]
            missing = sorted(set(gold[key]) - set(scored[key]))
        else:
            missing = sorted(
                set(gold.loc[:, id_columns].itertuples(index=False, name=None))
                - set(scored.loc[:, id_columns].itertuples(index=False, name=None))
            )
        raise ValueError(f"Missing predictions for {len(missing)} examples.")

    y_true = scored["label"].to_numpy()
    y_pred = scored["pred_label"].to_numpy()
    overall = pd.DataFrame(
        [
            {
                "split": split,
                "rows": int(len(scored)),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": macro_f1_score(y_true, y_pred),
                "ordinal_mae": float(np.mean(np.abs(y_true.astype(float) - y_pred.astype(float)))),
            }
        ]
    )

    results = {"overall": overall}
    if group_col is None:
        return results

    grouped_rows: list[dict[str, Any]] = []
    for group_value, frame in scored.groupby(group_col, sort=True):
        y_group = frame["label"].to_numpy()
        pred_group = frame["pred_label"].to_numpy()
        grouped_rows.append(
            {
                "split": split,
                group_col: group_value,
                "rows": int(len(frame)),
                "accuracy": accuracy_score(y_group, pred_group),
                "macro_f1": macro_f1_score(y_group, pred_group),
                "ordinal_mae": float(np.mean(np.abs(y_group.astype(float) - pred_group.astype(float)))),
            }
        )
    results[f"by_{group_col.removesuffix('_hours')}"] = pd.DataFrame(grouped_rows)
    return results


def evaluate_regression_predictions(
    gold: pd.DataFrame,
    predictions: pd.DataFrame | pd.Series,
    *,
    split: str,
    value_col: str = "target",
    pred_col: str = "pred_target",
    group_col: str | None = None,
    id_columns: tuple[str, ...] = ("example_id",),
) -> dict[str, pd.DataFrame]:
    """Evaluate continuous predictions against a gold table."""
    pred_frame = prediction_frame(predictions, id_columns=id_columns, value_col=pred_col)
    scored = gold.merge(pred_frame, on=list(id_columns), how="inner")
    if len(scored) != len(gold):
        if len(id_columns) == 1:
            key = id_columns[0]
            missing = sorted(set(gold[key]) - set(scored[key]))
        else:
            missing = sorted(
                set(gold.loc[:, id_columns].itertuples(index=False, name=None))
                - set(scored.loc[:, id_columns].itertuples(index=False, name=None))
            )
        raise ValueError(f"Missing predictions for {len(missing)} examples.")

    overall_row = {"split": split, "rows": int(len(scored))}
    overall_row.update(regression_metrics(scored[value_col].to_numpy(dtype=float), scored[pred_col].to_numpy(dtype=float)))
    results = {"overall": pd.DataFrame([overall_row])}
    if group_col is None:
        return results

    grouped_rows: list[dict[str, Any]] = []
    for group_value, frame in scored.groupby(group_col, sort=True):
        row = {"split": split, group_col: group_value, "rows": int(len(frame))}
        row.update(regression_metrics(frame[value_col].to_numpy(dtype=float), frame[pred_col].to_numpy(dtype=float)))
        grouped_rows.append(row)
    results[f"by_{group_col.removesuffix('_hours')}"] = pd.DataFrame(grouped_rows)
    return results
