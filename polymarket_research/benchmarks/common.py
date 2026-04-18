"""Shared helpers for protocol and tabular benchmark layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_json_ready(value: Any) -> Any:
    """Convert benchmark config values into JSON-safe objects."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_ready(v) for v in value]
    return value


def normalize_utc_timestamp(timestamp: pd.Timestamp | None) -> pd.Timestamp | None:
    """Normalize an optional timestamp to UTC."""
    if timestamp is None:
        return None
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def assign_time_splits(
    examples: pd.DataFrame,
    *,
    split_on: str,
    valid_columns: set[str],
    split_timestamp_utc: pd.Timestamp | None,
    train_fraction: float,
) -> pd.Series:
    """Assign train/test splits using a single time threshold."""
    if examples.empty:
        return pd.Series(dtype="string")
    if split_on not in valid_columns:
        valid = ", ".join(sorted(repr(column) for column in valid_columns))
        raise ValueError(f"split_on must be one of: {valid}.")

    split_time = (
        pd.Timestamp(split_timestamp_utc)
        if split_timestamp_utc is not None
        else pd.to_datetime(examples[split_on], utc=True).quantile(float(train_fraction))
    )
    split_source = pd.to_datetime(examples[split_on], utc=True)
    return pd.Series(
        np.where(split_source < split_time, "train", "test"),
        index=examples.index,
        dtype="string",
    )


def build_example_id(frame: pd.DataFrame, *, prefix_parts: list[str]) -> pd.Series:
    """Build a stable example id from selected columns."""
    id_parts: list[pd.Series] = []
    for column in prefix_parts:
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            normalized = pd.to_datetime(series, utc=True, errors="coerce").dt.strftime("%Y%m%dT%H%M%SZ")
        else:
            normalized = series.astype(str)
        id_parts.append(normalized)

    if not id_parts:
        return pd.RangeIndex(start=0, stop=len(frame)).astype(str)

    example_id = id_parts[0]
    for series in id_parts[1:]:
        example_id = example_id + "__" + series
    return example_id


def format_terminal_example_ids(frame: pd.DataFrame) -> pd.Series:
    """Return the canonical terminal example ids for a frame."""
    return build_example_id(
        frame.assign(horizon_tag="h" + frame["horizon_hours"].astype(int).astype(str)),
        prefix_parts=["market_id", "horizon_tag", "cutoff_timestamp_utc"],
    )


def format_decisiveness_example_ids(frame: pd.DataFrame) -> pd.Series:
    """Return the canonical decisiveness example ids for a frame."""
    return build_example_id(
        frame.assign(task_tag="decisive"),
        prefix_parts=["market_id", "task_tag", "cutoff_timestamp_utc"],
    )


def format_repricing_example_ids(frame: pd.DataFrame) -> pd.Series:
    """Return the canonical repricing example ids for a frame."""
    return build_example_id(
        frame.assign(horizon_tag="repricing_h" + frame["future_horizon_hours"].astype(int).astype(str)),
        prefix_parts=["market_id", "horizon_tag", "timestamp_utc"],
    )


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-6) -> float:
    """Compute aggregate binary log loss."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


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
    """Normalize user predictions into an id-columns-plus-pred_prob frame."""
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
) -> dict[str, pd.DataFrame]:
    """Evaluate binary probabilistic predictions against a gold table."""
    pred_frame = prediction_frame(predictions, id_columns=id_columns, value_col="pred_prob")
    scored = gold.merge(pred_frame, on=list(id_columns), how="inner")
    if len(scored) != len(gold):
        if len(id_columns) == 1:
            key = id_columns[0]
            missing = sorted(set(gold[key]) - set(scored[key]))
        else:
            missing = sorted(set(gold.loc[:, id_columns].itertuples(index=False, name=None)) - set(scored.loc[:, id_columns].itertuples(index=False, name=None)))
        raise ValueError(f"Missing predictions for {len(missing)} examples.")

    y_true = scored["label"].to_numpy(dtype=float)
    y_prob = scored["pred_prob"].to_numpy(dtype=float)
    overall = pd.DataFrame(
        [
            {
                "split": split,
                "rows": int(len(scored)),
                "log_loss": binary_log_loss(y_true, y_prob),
                "brier_score": brier_score(y_true, y_prob),
                "roc_auc": roc_auc(y_true, y_prob),
            }
        ]
    )

    results = {"overall": overall}
    if group_col is None:
        return results

    grouped_rows: list[dict[str, Any]] = []
    for group_value, frame in scored.groupby(group_col, sort=True):
        y_group = frame["label"].to_numpy(dtype=float)
        p_group = frame["pred_prob"].to_numpy(dtype=float)
        grouped_rows.append(
            {
                "split": split,
                group_col: int(group_value) if pd.notna(group_value) else group_value,
                "rows": int(len(frame)),
                "log_loss": binary_log_loss(y_group, p_group),
                "brier_score": brier_score(y_group, p_group),
                "roc_auc": roc_auc(y_group, p_group),
            }
        )
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


@dataclass(frozen=True)
class SplitFrame:
    """Simple train/test split views for tabular access."""

    frame: pd.DataFrame
    target_col: str
    feature_columns: list[str]

    @property
    def X(self) -> pd.DataFrame:
        return self.frame.loc[:, self.feature_columns].copy()

    @property
    def y(self) -> pd.Series:
        return self.frame.loc[:, self.target_col].copy()
