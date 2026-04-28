"""Reference baselines for frozen benchmark artifacts."""

from polymarket_research.benchmarks.baselines.decisiveness import (
    DecisivenessMajorityBaseline,
    fit_decisiveness_majority_baseline,
)
from polymarket_research.benchmarks.baselines.repricing import (
    RepricingTrainRateBaseline,
    fit_repricing_train_rate_baseline,
)
from polymarket_research.benchmarks.baselines.terminal import (
    TerminalLastProbabilityBaseline,
    TerminalTrainRateBaseline,
    fit_terminal_last_probability_baseline,
    fit_terminal_train_rate_baseline,
)

__all__ = [
    "TerminalTrainRateBaseline",
    "TerminalLastProbabilityBaseline",
    "RepricingTrainRateBaseline",
    "DecisivenessMajorityBaseline",
    "fit_terminal_train_rate_baseline",
    "fit_terminal_last_probability_baseline",
    "fit_repricing_train_rate_baseline",
    "fit_decisiveness_majority_baseline",
]
