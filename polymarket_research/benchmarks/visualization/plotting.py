"""Optional plotting helpers for frozen benchmark artifacts.

These functions are convenience utilities for notebooks. They operate on loaded
benchmark objects, prediction frames, and evaluator reports; they do not build
benchmarks and do not access raw or canonical data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from polymarket_research.benchmarks.evaluation.metrics import prediction_frame
from polymarket_research.benchmarks.utils.splits import select_split_rows


def _get_ax(ax=None, *, figsize: tuple[float, float] = (6.0, 4.0)):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    _, created_ax = plt.subplots(figsize=figsize)
    return created_ax


def _title_prefix(benchmark: Any) -> str:
    source = getattr(benchmark, "source", None)
    if source is None:
        return "Benchmark"
    return str(source).capitalize()


def _infer_prediction_id_columns(targets: pd.DataFrame, predictions: pd.DataFrame | pd.Series) -> tuple[str, ...]:
    if isinstance(predictions, pd.Series):
        return ("example_id",)

    candidate_id_columns = (
        ("market_id", "horizon_hours"),
        ("market_id", "timestamp_utc"),
        ("market_id", "cutoff_timestamp_utc"),
        ("example_id",),
    )
    target_columns = set(targets.columns)
    prediction_columns = set(predictions.columns)
    for id_columns in candidate_id_columns:
        if set(id_columns).issubset(target_columns) and set(id_columns).issubset(prediction_columns):
            return id_columns
    raise ValueError("Could not infer prediction id columns from benchmark targets and predictions.")


def _scored_frame(
    benchmark: Any,
    predictions: pd.DataFrame | pd.Series,
    *,
    split: str,
    pred_col: str,
) -> pd.DataFrame:
    targets = select_split_rows(benchmark.targets_frame, split).reset_index(drop=True)
    id_columns = _infer_prediction_id_columns(targets, predictions)
    pred_frame = prediction_frame(predictions, id_columns=id_columns, value_col=pred_col)
    scored = targets.merge(pred_frame, on=list(id_columns), how="inner")
    if len(scored) != len(targets):
        raise ValueError(f"Missing predictions for {len(targets) - len(scored)} examples.")
    return scored


def _terminal_visible_history_frame(
    benchmark: Any,
    *,
    split: str | None = "train",
) -> pd.DataFrame:
    """Return terminal examples with display-only visible history fields."""
    examples = select_split_rows(benchmark.examples, split).copy()
    required_columns = {"market_id", "horizon_hours", "cutoff_timestamp_utc", "current_yes_probability"}
    missing_columns = sorted(required_columns - set(examples.columns))
    if missing_columns:
        raise ValueError(f"benchmark examples are missing required columns: {missing_columns}")
    if examples.empty:
        return examples.assign(
            visible_history_days=pd.Series(dtype=float),
        )

    examples["cutoff_timestamp_utc"] = pd.to_datetime(
        examples["cutoff_timestamp_utc"],
        utc=True,
        errors="coerce",
    )
    if "cutoff_age_hours" in examples.columns:
        examples["visible_history_days"] = pd.to_numeric(examples["cutoff_age_hours"], errors="coerce") / 24.0
    elif "created_at" in examples.columns:
        created_at = pd.to_datetime(examples["created_at"], utc=True, errors="coerce")
        examples["visible_history_days"] = (
            examples["cutoff_timestamp_utc"] - created_at
        ).dt.total_seconds() / (24.0 * 3600.0)
    else:
        examples["visible_history_days"] = np.nan
    return examples


def select_terminal_history_prefix_examples(
    benchmark: Any,
    *,
    split: str | None = "train",
    examples_per_horizon: int = 1,
) -> pd.DataFrame:
    """Select representative terminal examples for visible-history prefix plots."""
    frame = _terminal_visible_history_frame(benchmark, split=split)
    frame = frame.dropna(subset=["current_yes_probability"]).copy()
    if frame.empty:
        return frame.reset_index(drop=True)

    rows: list[pd.DataFrame] = []
    for _, horizon_frame in frame.groupby("horizon_hours", sort=True):
        median_probability = horizon_frame["current_yes_probability"].median()
        selected = (
            horizon_frame.assign(
                _distance_to_median_probability=(
                    horizon_frame["current_yes_probability"] - median_probability
                ).abs()
            )
            .sort_values(
                ["_distance_to_median_probability", "cutoff_timestamp_utc", "market_id"],
                kind="stable",
            )
            .head(int(examples_per_horizon))
            .drop(columns=["_distance_to_median_probability"])
        )
        rows.append(selected)

    return pd.concat(rows, ignore_index=True).sort_values("horizon_hours", kind="stable").reset_index(drop=True)


def plot_terminal_visible_history_diagnostics(
    benchmark: Any,
    *,
    split: str | None = "train",
    frame: pd.DataFrame | None = None,
    bins: int = 40,
    title_prefix: str | None = None,
    axes=None,
):
    """Plot terminal visible-history duration and current cutoff probability."""
    if frame is None:
        frame = _terminal_visible_history_frame(benchmark, split=split)
    if frame.empty:
        raise ValueError("Cannot plot visible-history diagnostics for an empty frame.")
    if "visible_history_days" not in frame.columns:
        frame = frame.copy()
        frame["visible_history_days"] = pd.to_numeric(frame["cutoff_age_hours"], errors="coerce") / 24.0

    if axes is None:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))
    axes_list = np.ravel(np.asarray(axes, dtype=object)).tolist()
    if len(axes_list) < 2:
        raise ValueError("axes must contain at least two matplotlib axes.")

    prefix = title_prefix or _title_prefix(benchmark)
    split_suffix = "" if split is None else f" {split}"

    duration_ax = axes_list[0]
    frame["visible_history_days"].plot(
        kind="hist",
        bins=int(bins),
        ax=duration_ax,
        color="tab:blue",
        alpha=0.85,
    )
    duration_ax.set_title(f"{prefix}{split_suffix} visible history window")
    duration_ax.set_xlabel("days from market creation to cutoff")
    duration_ax.set_ylabel("examples")
    duration_ax.grid(True, axis="y", alpha=0.25)

    probability_ax = axes_list[1]
    probability_bins = min(int(bins), 30)
    for horizon, horizon_frame in frame.groupby("horizon_hours", sort=True):
        horizon_frame["current_yes_probability"].plot(
            kind="hist",
            bins=probability_bins,
            ax=probability_ax,
            alpha=0.45,
            label=f"{int(horizon)}h",
        )
    probability_ax.set_title(f"{prefix}{split_suffix} current cutoff probability")
    probability_ax.set_xlabel("yes_probability at cutoff")
    probability_ax.set_ylabel("examples")
    probability_ax.legend(title="horizon")
    probability_ax.grid(True, axis="y", alpha=0.25)

    return tuple(axes_list[:2])


def _plot_terminal_history_prefix_row(benchmark: Any, example_row: pd.Series, *, ax) -> None:
    market_id = str(example_row["market_id"])
    cutoff = pd.Timestamp(example_row["cutoff_timestamp_utc"])
    history = benchmark.market_history(market_id).copy()
    history["timestamp_utc"] = pd.to_datetime(history["timestamp_utc"], utc=True, errors="coerce")
    history["yes_probability"] = pd.to_numeric(history["yes_probability"], errors="coerce")
    history = history.dropna(subset=["timestamp_utc", "yes_probability"]).sort_values("timestamp_utc", kind="stable")

    visible = history.loc[history["timestamp_utc"].le(cutoff)]
    after_cutoff = history.loc[history["timestamp_utc"].gt(cutoff)]

    ax.plot(
        visible["timestamp_utc"],
        visible["yes_probability"],
        color="tab:blue",
        linewidth=1.8,
        label="visible prefix",
    )
    if not after_cutoff.empty:
        ax.plot(
            after_cutoff["timestamp_utc"],
            after_cutoff["yes_probability"],
            color="0.75",
            linewidth=1.0,
            label="after cutoff",
        )
    ax.axvline(cutoff, color="tab:red", linestyle="--", linewidth=1.4, label="cutoff")
    ax.set_ylim(-0.02, 1.02)

    title_parts = [f"{int(example_row['horizon_hours'])}h horizon"]
    if "label" in example_row and pd.notna(example_row["label"]):
        title_parts.append(f"label={int(example_row['label'])}")
    if "current_yes_probability" in example_row and pd.notna(example_row["current_yes_probability"]):
        title_parts.append(f"p_cutoff={float(example_row['current_yes_probability']):.3f}")
    ax.set_title(", ".join(title_parts))
    ax.set_xlabel("timestamp_utc")
    ax.set_ylabel("yes_probability")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")


def plot_terminal_history_prefix_examples(
    benchmark: Any,
    examples: pd.DataFrame | None = None,
    *,
    split: str | None = "train",
    examples_per_horizon: int = 1,
    axes=None,
):
    """Plot representative terminal histories with visible prefix and cutoff."""
    if examples is None:
        examples = select_terminal_history_prefix_examples(
            benchmark,
            split=split,
            examples_per_horizon=examples_per_horizon,
        )
    else:
        examples = examples.copy().reset_index(drop=True)

    if examples.empty:
        raise ValueError("Cannot plot terminal history prefixes for an empty examples frame.")
    if "current_yes_probability" not in examples.columns:
        cutoff_probabilities = benchmark.market_cutoff_probabilities(split=None)
        examples = examples.merge(
            cutoff_probabilities,
            on=["market_id", "horizon_hours"],
            how="left",
        ).rename(columns={"market_pred_prob": "current_yes_probability"})

    if axes is None:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(len(examples), 1, figsize=(10.0, 3.2 * len(examples)))
    axes_list = np.ravel(np.asarray(axes, dtype=object)).tolist()
    if len(axes_list) < len(examples):
        raise ValueError("axes must contain at least one matplotlib axis per example.")

    for ax, (_, row) in zip(axes_list, examples.iterrows(), strict=False):
        _plot_terminal_history_prefix_row(benchmark, row, ax=ax)
    return axes_list[: len(examples)]


def plot_metric_by_horizon(
    report: dict[str, pd.DataFrame],
    *,
    metric: str = "log_loss",
    horizon_col: str | None = None,
    title: str | None = None,
    ax=None,
):
    """Plot a horizon-level metric from an evaluator report."""
    by_horizon = report.get("by_horizon")
    if by_horizon is None or by_horizon.empty:
        raise ValueError("report must contain a non-empty 'by_horizon' table.")
    if metric not in by_horizon.columns:
        raise ValueError(f"by_horizon table does not contain metric column {metric!r}.")

    if horizon_col is None:
        horizon_candidates = [column for column in by_horizon.columns if str(column).endswith("_hours")]
        if not horizon_candidates:
            raise ValueError("Could not infer horizon column from by_horizon table.")
        horizon_col = horizon_candidates[0]

    plot_frame = by_horizon.sort_values(horizon_col, kind="stable")
    ax = _get_ax(ax)
    ax.plot(plot_frame[horizon_col], plot_frame[metric], marker="o", linewidth=2)
    ax.set_xlabel(horizon_col)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by horizon")
    ax.grid(True, alpha=0.25)
    return ax


def plot_label_distribution(
    benchmark: Any,
    *,
    split: str | None = None,
    label_col: str | None = None,
    normalize: bool = False,
    title: str | None = None,
    ax=None,
):
    """Plot target label counts or proportions from a loaded benchmark."""
    frame = benchmark.targets_frame.copy()
    if split is not None:
        frame = select_split_rows(frame, split)
    if label_col is None:
        label_col = "label_name" if "label_name" in frame.columns else "label"
    if label_col not in frame.columns:
        raise ValueError(f"targets_frame does not contain label column {label_col!r}.")

    counts = frame[label_col].value_counts(normalize=normalize, dropna=False).sort_index()
    ax = _get_ax(ax)
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel(label_col)
    ax.set_ylabel("proportion" if normalize else "rows")
    split_suffix = "" if split is None else f" ({split})"
    ax.set_title(title or f"{_title_prefix(benchmark)} label distribution{split_suffix}")
    ax.grid(True, axis="y", alpha=0.25)
    return ax


def plot_binary_label_rate_by_split(
    benchmark: Any,
    *,
    group_col: str | None = None,
    title: str | None = None,
    ax=None,
):
    """Plot empirical positive label rates by split, optionally grouped by a task column."""
    frame = benchmark.targets_frame.copy()
    if "split" not in frame.columns or "label" not in frame.columns:
        raise ValueError("targets_frame must contain 'split' and 'label' columns.")
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame.dropna(subset=["split", "label"])

    if group_col is None:
        plot_frame = frame.groupby("split", sort=True)["label"].mean().rename("positive_rate").reset_index()
        ax = _get_ax(ax)
        ax.bar(plot_frame["split"].astype(str), plot_frame["positive_rate"])
        ax.set_xlabel("split")
    else:
        if group_col not in frame.columns:
            raise ValueError(f"targets_frame does not contain group column {group_col!r}.")
        plot_frame = (
            frame.groupby(["split", group_col], sort=True)["label"]
            .mean()
            .rename("positive_rate")
            .reset_index()
        )
        pivot = plot_frame.pivot(index=group_col, columns="split", values="positive_rate").sort_index()
        ax = _get_ax(ax)
        pivot.plot(kind="bar", ax=ax)
        ax.set_xlabel(group_col)
        ax.legend(title="split")

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("positive label rate")
    ax.set_title(title or f"{_title_prefix(benchmark)} label rate by split")
    ax.grid(True, axis="y", alpha=0.25)
    return ax


def plot_numeric_distribution(
    benchmark: Any,
    *,
    column: str,
    split: str | None = None,
    bins: int = 30,
    title: str | None = None,
    ax=None,
):
    """Plot a numeric target distribution from a loaded benchmark."""
    frame = benchmark.targets_frame.copy()
    if split is not None:
        frame = select_split_rows(frame, split)
    if column not in frame.columns:
        raise ValueError(f"targets_frame does not contain numeric column {column!r}.")

    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    ax = _get_ax(ax)
    ax.hist(values, bins=int(bins), edgecolor="white")
    ax.set_xlabel(column)
    ax.set_ylabel("rows")
    split_suffix = "" if split is None else f" ({split})"
    ax.set_title(title or f"{_title_prefix(benchmark)} {column} distribution{split_suffix}")
    ax.grid(True, axis="y", alpha=0.25)
    return ax


def plot_binary_calibration(
    benchmark: Any,
    predictions: pd.DataFrame | pd.Series,
    *,
    split: str = "test",
    n_bins: int = 10,
    title: str | None = None,
    ax=None,
):
    """Plot predicted probability against empirical binary label rate."""
    scored = _scored_frame(benchmark, predictions, split=split, pred_col="pred_prob")
    scored["pred_prob"] = pd.to_numeric(scored["pred_prob"], errors="coerce").clip(0.0, 1.0)
    scored["label"] = pd.to_numeric(scored["label"], errors="coerce")
    scored = scored.dropna(subset=["pred_prob", "label"])

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    scored["prob_bin"] = pd.cut(scored["pred_prob"], bins=bins, include_lowest=True)
    calibration = (
        scored.groupby("prob_bin", observed=False)
        .agg(predicted=("pred_prob", "mean"), observed=("label", "mean"), rows=("label", "size"))
        .dropna(subset=["predicted", "observed"])
        .reset_index(drop=True)
    )

    ax = _get_ax(ax)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.6", linewidth=1)
    if not calibration.empty:
        sizes = np.sqrt(calibration["rows"].to_numpy(dtype=float)) * 20.0
        ax.scatter(calibration["predicted"], calibration["observed"], s=sizes, alpha=0.8)
        ax.plot(calibration["predicted"], calibration["observed"], alpha=0.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("empirical positive rate")
    ax.set_title(title or f"{_title_prefix(benchmark)} calibration ({split})")
    ax.grid(True, alpha=0.25)
    return ax


def plot_precision_recall(
    benchmark: Any,
    predictions: pd.DataFrame | pd.Series,
    *,
    split: str = "test",
    title: str | None = None,
    ax=None,
):
    """Plot a lightweight precision-recall curve for binary predictions."""
    scored = _scored_frame(benchmark, predictions, split=split, pred_col="pred_prob")
    scored["pred_prob"] = pd.to_numeric(scored["pred_prob"], errors="coerce").clip(0.0, 1.0)
    scored["label"] = pd.to_numeric(scored["label"], errors="coerce")
    scored = scored.dropna(subset=["pred_prob", "label"]).sort_values("pred_prob", ascending=False, kind="stable")
    positives = scored["label"].to_numpy(dtype=float)
    total_positive = float(positives.sum())

    ax = _get_ax(ax)
    if len(scored) and total_positive > 0.0:
        true_positive = np.cumsum(positives)
        rank = np.arange(1, len(scored) + 1, dtype=float)
        precision = true_positive / rank
        recall = true_positive / total_positive
        ax.step(recall, precision, where="post", linewidth=2)
        ax.axhline(total_positive / len(scored), linestyle="--", color="0.6", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(title or f"{_title_prefix(benchmark)} precision-recall ({split})")
    ax.grid(True, alpha=0.25)
    return ax


def plot_confusion_matrix(
    benchmark: Any,
    predictions: pd.DataFrame | pd.Series,
    *,
    split: str = "test",
    normalize: bool = False,
    title: str | None = None,
    ax=None,
):
    """Plot a confusion matrix for discrete label predictions."""
    scored = _scored_frame(benchmark, predictions, split=split, pred_col="pred_label")
    scored["label"] = pd.to_numeric(scored["label"], errors="coerce")
    scored["pred_label"] = pd.to_numeric(scored["pred_label"], errors="coerce")
    scored = scored.dropna(subset=["label", "pred_label"])
    labels = sorted(set(scored["label"].astype(int)) | set(scored["pred_label"].astype(int)))
    matrix = pd.crosstab(scored["label"].astype(int), scored["pred_label"].astype(int)).reindex(
        index=labels,
        columns=labels,
        fill_value=0,
    )
    values = matrix.to_numpy(dtype=float)
    if normalize:
        row_totals = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_totals, out=np.zeros_like(values), where=row_totals != 0)

    ax = _get_ax(ax)
    image = ax.imshow(values, cmap="Blues")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("predicted label")
    ax.set_ylabel("true label")
    ax.set_title(title or f"{_title_prefix(benchmark)} confusion matrix ({split})")

    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(col_index, row_index, text, ha="center", va="center", color="black")
    return ax


def plot_market_history(
    benchmark: Any,
    *,
    market_id: str | None = None,
    cutoff: pd.Timestamp | str | None = None,
    future_until: pd.Timestamp | str | None = None,
    title: str | None = None,
    ax=None,
):
    """Plot one market probability trajectory from a loaded benchmark."""
    if market_id is None:
        if benchmark.examples.empty:
            raise ValueError("Cannot infer market_id from an empty benchmark.")
        market_id = str(benchmark.examples.iloc[0]["market_id"])

    history = benchmark.market_history(str(market_id)).copy()
    history["timestamp_utc"] = pd.to_datetime(history["timestamp_utc"], utc=True, errors="coerce")
    history["yes_probability"] = pd.to_numeric(history["yes_probability"], errors="coerce")
    history = history.dropna(subset=["timestamp_utc", "yes_probability"]).sort_values("timestamp_utc", kind="stable")

    ax = _get_ax(ax, figsize=(8.0, 4.0))
    ax.plot(history["timestamp_utc"], history["yes_probability"], linewidth=1.8)
    if cutoff is not None:
        cutoff_ts = pd.Timestamp(cutoff)
        ax.axvline(cutoff_ts, color="tab:red", linestyle="--", linewidth=1.5, label="cutoff")
    if future_until is not None:
        future_ts = pd.Timestamp(future_until)
        ax.axvline(future_ts, color="tab:orange", linestyle=":", linewidth=1.5, label="future horizon")
    if cutoff is not None or future_until is not None:
        ax.legend()
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("timestamp_utc")
    ax.set_ylabel("yes_probability")
    ax.set_title(title or f"{_title_prefix(benchmark)} market history: {market_id}")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    return ax
