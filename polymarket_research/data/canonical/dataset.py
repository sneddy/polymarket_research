"""Canonicalized representations of markets, probabilities, and external covariates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from polymarket_research.data.raw.dataset import RawExternalCovariates, RawPolymarketDataset
from polymarket_research.utils.text import build_family_id


@dataclass(frozen=True)
class CanonicalDataset:
    """Hold canonicalized market, probability, and external covariate tables."""

    markets: pd.DataFrame
    probabilities: pd.DataFrame
    external_covariates: pd.DataFrame | None = None

    def summary(self) -> pd.DataFrame:
        """Return a compact summary of the canonical tables."""
        return pd.DataFrame(
            [
                {"name": "markets", "rows": len(self.markets), "cols": self.markets.shape[1]},
                {"name": "probabilities", "rows": len(self.probabilities), "cols": self.probabilities.shape[1]},
                {
                    "name": "external_covariates",
                    "rows": 0 if self.external_covariates is None else len(self.external_covariates),
                    "cols": 0 if self.external_covariates is None else self.external_covariates.shape[1],
                },
            ]
        )

    def save(self, directory: str | Path) -> pd.DataFrame:
        """Persist canonical tables as parquet files and return a save manifest."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        outputs: list[tuple[str, pd.DataFrame | None]] = [
            ("markets.parquet", self.markets),
            ("probabilities.parquet", self.probabilities),
            ("external_covariates.parquet", self.external_covariates),
        ]

        manifest_rows: list[dict[str, object]] = []
        for filename, frame in outputs:
            if frame is None:
                continue
            frame.to_parquet(target_dir / filename, index=False)
            manifest_rows.append({"file": filename, "rows": len(frame), "cols": frame.shape[1]})
        return pd.DataFrame(manifest_rows)

    @classmethod
    def from_parquet(cls, directory: str | Path) -> "CanonicalDataset":
        """Instantiate the canonical dataset from saved parquet files."""
        source_dir = Path(directory)
        markets = pd.read_parquet(source_dir / "markets.parquet")
        probabilities = pd.read_parquet(source_dir / "probabilities.parquet")
        external_path = source_dir / "external_covariates.parquet"
        external_covariates = pd.read_parquet(external_path) if external_path.exists() else None
        return cls(
            markets=markets,
            probabilities=probabilities,
            external_covariates=external_covariates,
        )


@dataclass
class CanonicalDatasetBuilder:
    """Build canonicalized tables from raw Polymarket sources."""

    raw_dataset: RawPolymarketDataset
    raw_external: RawExternalCovariates | None = None
    resolved_only: bool = True

    def build(self) -> CanonicalDataset:
        """Build the canonical dataset from the configured raw sources."""
        if not self.raw_dataset.is_loaded:
            self.raw_dataset.load()

        raw_markets = self.raw_dataset.markets
        if self.resolved_only and raw_markets is not None and "resolved" in raw_markets.columns:
            raw_markets = raw_markets.loc[raw_markets["resolved"].fillna(False)].reset_index(drop=True)

        markets = self.canonicalize_markets(raw_markets)
        probabilities = self.canonicalize_probabilities(self.raw_dataset.probabilities, allowed_market_ids=markets["market_id"])
        external_covariates = None
        if self.raw_external is not None:
            if not self.raw_external.is_loaded:
                self.raw_external.load()
            external_covariates = self.canonicalize_external_covariates(self.raw_external.covariates)
        return CanonicalDataset(
            markets=markets,
            probabilities=probabilities,
            external_covariates=external_covariates,
        )

    @staticmethod
    def canonicalize_markets(markets: pd.DataFrame | None) -> pd.DataFrame:
        """Canonicalize market metadata into a stable schema for downstream layers."""
        if markets is None:
            return pd.DataFrame()

        out = markets.copy()
        out["market_id"] = out["market_id"].astype(str)
        if "event_id" in out.columns:
            out["event_id"] = out["event_id"].astype("string")
        for column in ("created_at", "end_date", "probability_start_utc", "probability_end_utc"):
            if column in out.columns:
                out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")

        numeric_columns = (
            "volume_num",
            "final_yes_probability",
            "trade_rows",
            "probability_rows",
        )
        for column in numeric_columns:
            if column in out.columns:
                out[column] = pd.to_numeric(out[column], errors="coerce")

        if "primary_domain" in out.columns:
            out["primary_domain"] = out["primary_domain"].fillna("unknown").astype(str)
            out["domain"] = out["primary_domain"]
        else:
            out["primary_domain"] = "unknown"
            out["domain"] = "unknown"

        out["family_id"] = [
            build_family_id(question, domain, tags)
            for question, domain, tags in zip(
                out.get("question", pd.Series(index=out.index, dtype=object)),
                out["primary_domain"],
                out.get("tag_labels", pd.Series(index=out.index, dtype=object)),
                strict=False,
            )
        ]
        return out.reset_index(drop=True)

    @staticmethod
    def canonicalize_probabilities(
        probabilities: pd.DataFrame | None,
        *,
        allowed_market_ids: pd.Series | pd.Index | None = None,
    ) -> pd.DataFrame:
        """Canonicalize probability history into typed, filtered trajectories."""
        if probabilities is None:
            return pd.DataFrame()

        out = probabilities.copy()
        out["market_id"] = out["market_id"].astype(str)
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
        for column in ("yes_probability", "trade_count", "total_size", "last_trade_price"):
            if column in out.columns:
                out[column] = pd.to_numeric(out[column], errors="coerce")
        if "observed_trade" in out.columns:
            out["observed_trade"] = pd.to_numeric(out["observed_trade"], errors="coerce").fillna(0).astype(int)
        if allowed_market_ids is not None:
            out = out.loc[out["market_id"].isin(pd.Index(allowed_market_ids).astype(str))]
        return out.sort_values(["market_id", "timestamp_utc"], kind="stable").reset_index(drop=True)

    @staticmethod
    def canonicalize_external_covariates(external_covariates: pd.DataFrame | None) -> pd.DataFrame:
        """Canonicalize external covariate rows into a stable time-series schema."""
        if external_covariates is None:
            return pd.DataFrame()

        out = external_covariates.copy()
        if "timestamp_utc" in out.columns:
            out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
        if "series_id" in out.columns:
            out["series_id"] = out["series_id"].astype(str)
        if "value" in out.columns:
            out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return out.sort_values(["series_id", "timestamp_utc"], kind="stable").reset_index(drop=True)
