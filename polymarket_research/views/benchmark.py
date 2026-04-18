"""Frozen benchmark views built on top of canonical and representation layers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import pandas as pd

from polymarket_research.benchmarks.common import (
    format_decisiveness_example_ids,
    format_repricing_example_ids,
    format_terminal_example_ids,
)
from polymarket_research.data.canonical.dataset import CanonicalDataset
from polymarket_research.data.representations.repricing import RepricingPanelBuilder
from polymarket_research.data.representations.terminal import TerminalPanelBuilder
from polymarket_research.tasks.base import TaskFrame


def _confidence_slice(probability: float) -> str:
    """Bucket confidence by distance from 0.5 for paper-facing summaries."""
    p = float(probability)
    confidence = max(p, 1.0 - p)
    if confidence < 0.60:
        return "50-60"
    if confidence < 0.75:
        return "60-75"
    if confidence < 0.90:
        return "75-90"
    return "90-100"


def _json_ready(value: Any) -> Any:
    """Convert nested Python/pandas objects into JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (str, bytes, dict, list, tuple, set)) else False:
        return None
    return value


@dataclass(frozen=True)
class BenchmarkView:
    """Benchmark-ready frozen view with task metadata and export helpers."""

    name: str
    frame: pd.DataFrame
    target_col: str
    time_col: str
    metadata: dict[str, Any] = field(default_factory=dict)
    entity_id_col: str = "market_id"

    @property
    def task(self) -> TaskFrame:
        """Expose the view as a task-ready frame."""
        return TaskFrame(
            name=self.name,
            frame=self.frame,
            target_col=self.target_col,
            time_col=self.time_col,
            entity_id_col=self.entity_id_col,
        )

    def summary(self) -> pd.DataFrame:
        """Return a compact benchmark-facing summary table."""
        frame = self.frame
        if frame.empty:
            return pd.DataFrame(
                [
                    {
                        "name": self.name,
                        "rows": 0,
                        "cols": 0,
                        "unique_markets": 0,
                        "categories": 0,
                        "target_col": self.target_col,
                        "positive_rate": None,
                    }
                ]
            )

        positive_rate = None
        if self.target_col in frame.columns and pd.api.types.is_numeric_dtype(frame[self.target_col]):
            target_values = frame[self.target_col].dropna().unique().tolist()
            if set(target_values).issubset({0, 1}):
                positive_rate = float(frame[self.target_col].mean())

        summary: dict[str, Any] = {
            "name": self.name,
            "rows": int(len(frame)),
            "cols": int(frame.shape[1]),
            "unique_markets": int(frame[self.entity_id_col].nunique()) if self.entity_id_col in frame.columns else 0,
            "categories": int(_category_series(frame).nunique()) if not frame.empty else 0,
            "target_col": self.target_col,
            "positive_rate": positive_rate,
            "feature_cols": int(len(self.task.feature_columns)),
        }
        if self.time_col in frame.columns:
            summary["time_min"] = pd.to_datetime(frame[self.time_col], utc=True, errors="coerce").min()
            summary["time_max"] = pd.to_datetime(frame[self.time_col], utc=True, errors="coerce").max()
        if "horizon_hours" in frame.columns:
            summary["horizons"] = ",".join(str(int(value)) for value in sorted(frame["horizon_hours"].dropna().unique()))
        return pd.DataFrame([summary])

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest for the benchmark view."""
        frame = self.frame
        manifest: dict[str, Any] = {
            "name": self.name,
            "rows": int(len(frame)),
            "cols": int(frame.shape[1]),
            "target_col": self.target_col,
            "time_col": self.time_col,
            "entity_id_col": self.entity_id_col,
            "feature_columns": self.task.feature_columns,
            "columns": list(frame.columns),
            "categories": sorted(_category_series(frame).dropna().astype(str).unique().tolist()) if not frame.empty else [],
            "metadata": _json_ready(self.metadata),
        }
        if self.time_col in frame.columns and not frame.empty:
            time_values = pd.to_datetime(frame[self.time_col], utc=True, errors="coerce")
            manifest["time_min"] = _json_ready(time_values.min())
            manifest["time_max"] = _json_ready(time_values.max())
        if "horizon_hours" in frame.columns:
            manifest["horizon_hours"] = [int(value) for value in sorted(frame["horizon_hours"].dropna().unique())]
        return manifest

    def export_manifest(self, path: str | Path) -> Path:
        """Write the manifest to disk as JSON."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.manifest(), indent=2, sort_keys=True), encoding="utf-8")
        return out_path

    def save(self, directory: str | Path) -> dict[str, Path]:
        """Save the view parquet and JSON manifest under one directory."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = out_dir / f"{self.name}.parquet"
        manifest_path = out_dir / f"{self.name}_manifest.json"
        self.frame.to_parquet(parquet_path, index=False)
        self.export_manifest(manifest_path)
        return {"parquet": parquet_path, "manifest": manifest_path}

    @classmethod
    def _load_saved(
        cls,
        directory: str | Path,
        *,
        expected_name: str,
        target_col: str,
        time_col: str,
        entity_id_col: str = "market_id",
    ) -> "BenchmarkView":
        """Load a saved benchmark view from parquet plus JSON manifest."""
        source_dir = Path(directory)
        parquet_path = source_dir / f"{expected_name}.parquet"
        manifest_path = source_dir / f"{expected_name}_manifest.json"

        frame = pd.read_parquet(parquet_path)
        metadata: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metadata = dict(manifest.get("metadata", {}))

        return cls(
            name=expected_name,
            frame=frame,
            target_col=target_col,
            time_col=time_col,
            metadata=metadata,
            entity_id_col=entity_id_col,
        )


def _attach_common_columns(
    frame: pd.DataFrame,
    *,
    probability_col: str,
    row_id_col: str,
    example_id: pd.Series | None = None,
) -> pd.DataFrame:
    """Attach stable ids and paper-facing slice columns shared by benchmark views."""
    out = frame.copy()
    if probability_col in out.columns:
        out["confidence_slice"] = out[probability_col].map(_confidence_slice)
        out["hard_case_10_90"] = out[probability_col].between(0.10, 0.90, inclusive="both")
    else:
        out["confidence_slice"] = "unknown"
        out["hard_case_10_90"] = False

    row_identifier = example_id if example_id is not None else pd.RangeIndex(start=0, stop=len(out)).astype(str)
    out["example_id"] = pd.Series(row_identifier, index=out.index).astype(str)
    out[row_id_col] = out["example_id"]
    return out


def _attach_splits(frame: pd.DataFrame, examples: pd.DataFrame | None) -> pd.DataFrame:
    """Attach split assignments from a protocol benchmark manifest when available."""
    out = frame.copy()
    if examples is None or out.empty or "example_id" not in out.columns:
        return out
    split_frame = examples.loc[:, ["example_id", "split"]].drop_duplicates()
    out = out.merge(split_frame, on="example_id", how="left")
    return out


def _category_series(frame: pd.DataFrame) -> pd.Series:
    if "research_category" not in frame.columns:
        return pd.Series(dtype="string")
    return frame["research_category"].fillna("unknown")


@dataclass(frozen=True)
class TerminalBenchmarkView(BenchmarkView):
    """Frozen terminal benchmark view derived from canonical market trajectories."""

    VIEW_NAME = "terminal_benchmark_view"

    @classmethod
    def from_canonical(
        cls,
        canonical: CanonicalDataset,
        *,
        horizons_hours: tuple[int, ...] = (24, 72, 168),
        max_snapshot_staleness_hours: float = 12.0,
        include_family_context: bool = True,
        show_progress: bool = False,
        examples: pd.DataFrame | None = None,
    ) -> "TerminalBenchmarkView":
        terminal = TerminalPanelBuilder(
            canonical=canonical,
            horizons_hours=horizons_hours,
            max_snapshot_staleness_hours=max_snapshot_staleness_hours,
            include_family_context=include_family_context,
            show_progress=show_progress,
        ).build().frame
        if not terminal.empty:
            example_id = format_terminal_example_ids(terminal)
            terminal = _attach_common_columns(
                terminal,
                probability_col="market_price_baseline",
                row_id_col="terminal_row_id",
                example_id=example_id,
            )
            terminal["admissible_terminal"] = 1
            terminal = terminal.sort_values(
                ["end_date", "market_id", "horizon_hours"],
                kind="stable",
            ).reset_index(drop=True)
            terminal = _attach_splits(terminal, examples)

        return cls(
            name=cls.VIEW_NAME,
            frame=terminal,
            target_col="target",
            time_col="cutoff_timestamp_utc",
            metadata={
                "horizons_hours": list(horizons_hours),
                "max_snapshot_staleness_hours": max_snapshot_staleness_hours,
                "include_family_context": include_family_context,
            },
        )

    @classmethod
    def from_parquet(cls, directory: str | Path) -> "TerminalBenchmarkView":
        """Load a saved terminal benchmark view from parquet cache."""
        return cls._load_saved(
            directory,
            expected_name=cls.VIEW_NAME,
            target_col="label",
            time_col="cutoff_timestamp_utc",
        )


@dataclass(frozen=True)
class DecisivenessBenchmarkView(BenchmarkView):
    """Frozen durable-decisiveness benchmark view derived from sampled prefixes."""

    VIEW_NAME = "decisiveness_benchmark_view"

    @classmethod
    def from_examples(
        cls,
        examples: pd.DataFrame,
        *,
        config=None,
    ) -> "DecisivenessBenchmarkView":
        frame = examples.copy()
        if not frame.empty:
            frame["admissible_decisiveness"] = 1
            frame = _attach_common_columns(
                frame,
                probability_col="current_yes_probability",
                row_id_col="decisiveness_row_id",
                example_id=format_decisiveness_example_ids(frame),
            )
            frame = frame.sort_values(["cutoff_timestamp_utc", "market_id"], kind="stable").reset_index(drop=True)

        return cls(
            name=cls.VIEW_NAME,
            frame=frame,
            target_col="label",
            time_col="cutoff_timestamp_utc",
            metadata={
                "source_view": "decisiveness_examples",
                "decisive_threshold": None if config is None else float(config.decisive_threshold),
                "ordinal_bin_edges_hours": None if config is None else [float(edge) for edge in config.ordinal_bin_edges_hours],
                "ordinal_bin_labels": None if config is None else list(config.ordinal_bin_labels),
            },
            entity_id_col="example_id",
        )

    @classmethod
    def from_canonical(
        cls,
        canonical: CanonicalDataset,
        *,
        decisive_threshold: float = 0.95,
        sample_every_hours: int = 12,
        min_history_points: int = 24,
        min_prefix_age_hours: float = 6.0,
        min_time_to_decisive_hours: float = 1.0,
        ordinal_bin_edges_hours: tuple[float, ...] = (24.0, 72.0),
        ordinal_bin_labels: tuple[str, ...] = ("short", "medium", "long"),
        target_market_only: bool = True,
        split_on: str = "decisive_timestamp_utc",
        split_timestamp_utc: pd.Timestamp | None = None,
        train_fraction: float = 0.8,
        show_progress: bool = False,
        examples: pd.DataFrame | None = None,
    ) -> "DecisivenessBenchmarkView":
        from polymarket_research.benchmarks import DecisivenessBenchmark, DecisivenessBenchmarkConfig

        benchmark = DecisivenessBenchmark.build(
            canonical,
            config=DecisivenessBenchmarkConfig(
                decisive_threshold=decisive_threshold,
                sample_every_hours=sample_every_hours,
                min_history_points=min_history_points,
                min_prefix_age_hours=min_prefix_age_hours,
                min_time_to_decisive_hours=min_time_to_decisive_hours,
                ordinal_bin_edges_hours=ordinal_bin_edges_hours,
                ordinal_bin_labels=ordinal_bin_labels,
                target_market_only=target_market_only,
                split_on=split_on,
                split_timestamp_utc=split_timestamp_utc,
                train_fraction=train_fraction,
                show_progress=show_progress,
            ),
        )
        return cls.from_examples(benchmark.examples, config=benchmark.config)

    @classmethod
    def from_parquet(cls, directory: str | Path) -> "DecisivenessBenchmarkView":
        """Load a saved decisiveness benchmark view from parquet cache."""
        return cls._load_saved(
            directory,
            expected_name=cls.VIEW_NAME,
            target_col="target",
            time_col="cutoff_timestamp_utc",
        )


@dataclass(frozen=True)
class RepricingBenchmarkView(BenchmarkView):
    """Frozen repricing benchmark view derived from canonical market trajectories."""

    VIEW_NAME = "repricing_benchmark_view"

    @classmethod
    def from_canonical(
        cls,
        canonical: CanonicalDataset,
        *,
        future_horizon_hours: int = 24,
        lookback_hours: int = 24,
        sample_every_hours: int = 12,
        move_threshold: float = 0.15,
        attach_external_shocks: bool = True,
        show_progress: bool = False,
        examples: pd.DataFrame | None = None,
    ) -> "RepricingBenchmarkView":
        repricing = RepricingPanelBuilder(
            canonical=canonical,
            future_horizon_hours=future_horizon_hours,
            lookback_hours=lookback_hours,
            sample_every_hours=sample_every_hours,
            move_threshold=move_threshold,
            attach_external_shocks=attach_external_shocks,
            show_progress=show_progress,
        ).build().frame

        if not repricing.empty:
            example_id = format_repricing_example_ids(repricing)
            repricing = _attach_common_columns(
                repricing,
                probability_col="current_yes_probability",
                row_id_col="repricing_row_id",
                example_id=example_id,
            )
            repricing["admissible_repricing"] = 1
            repricing = repricing.sort_values(
                ["timestamp_utc", "market_id"],
                kind="stable",
            ).reset_index(drop=True)
            repricing = _attach_splits(repricing, examples)

        return cls(
            name=cls.VIEW_NAME,
            frame=repricing,
            target_col="target",
            time_col="timestamp_utc",
            metadata={
                "future_horizon_hours": int(future_horizon_hours),
                "lookback_hours": int(lookback_hours),
                "sample_every_hours": int(sample_every_hours),
                "move_threshold": float(move_threshold),
                "attach_external_shocks": bool(attach_external_shocks),
            },
        )

    @classmethod
    def from_parquet(cls, directory: str | Path) -> "RepricingBenchmarkView":
        """Load a saved repricing benchmark view from parquet cache."""
        return cls._load_saved(
            directory,
            expected_name=cls.VIEW_NAME,
            target_col="target",
            time_col="timestamp_utc",
        )
