"""Optional visualization helpers for benchmark demos."""

from polymarket_research.benchmarks.visualization.plotting import (
    plot_binary_calibration,
    plot_binary_label_rate_by_split,
    plot_confusion_matrix,
    plot_label_distribution,
    plot_market_history,
    plot_metric_by_horizon,
    plot_numeric_distribution,
    plot_precision_recall,
    plot_terminal_history_prefix_examples,
    plot_terminal_visible_history_diagnostics,
    select_terminal_history_prefix_examples,
)

__all__ = [
    "select_terminal_history_prefix_examples",
    "plot_metric_by_horizon",
    "plot_label_distribution",
    "plot_binary_label_rate_by_split",
    "plot_numeric_distribution",
    "plot_binary_calibration",
    "plot_precision_recall",
    "plot_confusion_matrix",
    "plot_market_history",
    "plot_terminal_visible_history_diagnostics",
    "plot_terminal_history_prefix_examples",
]
