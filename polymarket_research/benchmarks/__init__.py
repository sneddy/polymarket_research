"""Clean benchmark and evaluation layer built on top of data representations and tasks."""

from polymarket_research.benchmarks.covariate_utils import (
    add_lagged_covariate_features,
    asof_join_covariates,
    load_external_covariates,
    pivot_covariates_to_wide,
)
from polymarket_research.benchmarks.dataset_utils import (
    add_time_features,
    build_multi_horizon_terminal_dataset,
    build_repricing_dataset,
    connect,
    default_feature_columns,
    extract_snapshot_features,
    load_probabilities_for_markets,
    load_snapshot_frame,
    prepare_resolved_markets,
    rolling_time_splits,
    summarize_metric_frame,
)
from polymarket_research.benchmarks.repricing import (
    RepricingBenchmark,
    RepricingBenchmarkConfig,
)
from polymarket_research.benchmarks.tabular import TabularBenchmark
from polymarket_research.benchmarks.terminal import (
    TerminalBenchmark,
    TerminalBenchmarkConfig,
)
from polymarket_research.benchmarks.decisiveness import (
    DecisivenessBenchmark,
    DecisivenessBenchmarkConfig,
)

__all__ = [
    "add_lagged_covariate_features",
    "add_time_features",
    "asof_join_covariates",
    "build_multi_horizon_terminal_dataset",
    "build_repricing_dataset",
    "connect",
    "DecisivenessBenchmark",
    "DecisivenessBenchmarkConfig",
    "default_feature_columns",
    "extract_snapshot_features",
    "load_external_covariates",
    "load_probabilities_for_markets",
    "load_snapshot_frame",
    "pivot_covariates_to_wide",
    "prepare_resolved_markets",
    "RepricingBenchmark",
    "RepricingBenchmarkConfig",
    "rolling_time_splits",
    "summarize_metric_frame",
    "TabularBenchmark",
    "TerminalBenchmark",
    "TerminalBenchmarkConfig",
]
