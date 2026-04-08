"""Configuration objects for Polymarket data loading and panel construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataPaths:
    """Filesystem locations used by the data loading pipeline."""

    repo_root: Path = PACKAGE_ROOT
    db_path: Path = field(default_factory=lambda: PACKAGE_ROOT / "db" / "resolved_probability_dataset.sqlite")
    external_covariates_path: Path = field(default_factory=lambda: PACKAGE_ROOT / "cached_data" / "external_covariates")

    def with_artefact_dir(self, artefact_dir: str | Path) -> Path:
        """Return an absolute artefact directory inside or outside the repository."""

        candidate = Path(artefact_dir)
        if candidate.is_absolute():
            return candidate
        return self.repo_root / candidate


@dataclass(frozen=True)
class MarketSelectionConfig:
    """Controls which Polymarket markets enter the raw research bundle."""

    domains: tuple[str, ...] = ("politics", "geopolitics", "technology", "finance_economy")
    max_markets_per_domain: int = 120
    min_probability_rows: int = 288


@dataclass(frozen=True)
class PanelBuildConfig:
    """Controls how terminal and repricing panels are constructed."""

    terminal_horizons_hours: tuple[int, ...] = (24, 72, 168)
    max_snapshot_staleness_hours: float = 12.0
    repricing_future_horizon_hours: int = 24
    repricing_lookback_hours: int = 24
    repricing_sample_every_hours: int = 12
    repricing_move_threshold: float = 0.15


@dataclass(frozen=True)
class ExternalShockConfig:
    """Controls how external covariates are converted into shock features."""

    z_threshold: float = 2.0
    std_window: int = 288
    join_max_age: str = "2D"
