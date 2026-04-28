"""Audit and reporting helpers for frozen benchmark manifests."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping

import pandas as pd


def benchmark_manifest_summary(benchmarks: Mapping[str, Any]) -> pd.DataFrame:
    """Return a compact source-by-source summary from loaded benchmark manifests."""
    rows: list[dict[str, Any]] = []
    for source, benchmark in benchmarks.items():
        manifest = benchmark.manifest()
        split_counts = manifest.get("split_counts", {})
        split_audit = manifest.get("split_audit", {})
        label_stats = manifest.get("label_stats", {}).get("overall", {})
        row: dict[str, Any] = {
            "source": manifest.get("source", source),
            "release_name": manifest.get("release_name"),
            "task": manifest.get("task"),
            "rows": manifest.get("rows"),
            "markets": manifest.get("markets"),
            "market_timeseries_rows": manifest.get("market_timeseries_rows"),
            "train_rows": split_counts.get("train", 0),
            "test_rows": split_counts.get("test", 0),
            "split_unit": manifest.get("split_policy", {}).get("split_unit"),
            "split_unit_overlap": split_audit.get("units_with_multiple_splits"),
        }
        if "horizons_hours" in manifest:
            row["horizons_hours"] = manifest["horizons_hours"]
        if "future_horizon_hours" in manifest:
            row["future_horizon_hours"] = manifest["future_horizon_hours"]
        if "ordinal_bin_edges_hours" in manifest:
            row["ordinal_bin_edges_hours"] = manifest["ordinal_bin_edges_hours"]
        if "ordinal_bin_labels" in manifest:
            row["ordinal_bin_labels"] = manifest["ordinal_bin_labels"]
        if "positive_rate" in label_stats:
            row["positive_rate"] = label_stats["positive_rate"]
        rows.append(row)
    return pd.DataFrame(rows)


def counts_by_split(frame: pd.DataFrame, *, split_col: str = "split") -> dict[str, int]:
    """Return stable row counts by split."""
    if frame.empty or split_col not in frame.columns:
        return {}
    counts = frame[split_col].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def counts_by_split_and_group(
    frame: pd.DataFrame,
    *,
    group_col: str,
    split_col: str = "split",
) -> dict[str, dict[str, int]]:
    """Return split-conditional counts for one grouping column."""
    if frame.empty or split_col not in frame.columns or group_col not in frame.columns:
        return {}

    grouped = (
        frame.groupby([split_col, group_col], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values([split_col, group_col], kind="stable")
    )
    out: dict[str, dict[str, int]] = {}
    for split_value, split_frame in grouped.groupby(split_col, sort=True):
        out[str(split_value)] = {
            str(group_value): int(rows)
            for group_value, rows in zip(split_frame[group_col], split_frame["rows"], strict=False)
        }
    return out


def categorical_distribution(
    frame: pd.DataFrame,
    *,
    value_col: str,
    split_col: str = "split",
) -> dict[str, dict[str, int]]:
    """Return overall and by-split categorical counts."""
    if frame.empty or value_col not in frame.columns:
        return {"overall": {}, "by_split": {}}

    clean = frame.loc[frame[value_col].notna(), [value_col, split_col] if split_col in frame.columns else [value_col]].copy()
    overall = clean[value_col].value_counts(dropna=False).sort_index()
    by_split = counts_by_split_and_group(clean, group_col=value_col, split_col=split_col) if split_col in clean.columns else {}
    return {
        "overall": {str(key): int(value) for key, value in overall.items()},
        "by_split": by_split,
    }


def binary_label_stats(
    frame: pd.DataFrame,
    *,
    label_col: str = "label",
    split_col: str = "split",
) -> dict[str, Any]:
    """Return counts and positive rates for binary labels overall and by split."""
    if frame.empty or label_col not in frame.columns:
        return {"overall": {"rows": 0, "positive_rate": None}, "by_split": {}}

    labels = pd.to_numeric(frame[label_col], errors="coerce")
    overall = {
        "rows": int(labels.notna().sum()),
        "positive_rate": None if labels.dropna().empty else float(labels.dropna().mean()),
    }

    by_split: dict[str, dict[str, Any]] = {}
    if split_col in frame.columns:
        for split_value, split_frame in frame.groupby(split_col, sort=True):
            split_labels = pd.to_numeric(split_frame[label_col], errors="coerce")
            by_split[str(split_value)] = {
                "rows": int(split_labels.notna().sum()),
                "positive_rate": None if split_labels.dropna().empty else float(split_labels.dropna().mean()),
            }
    return {"overall": overall, "by_split": by_split}


def split_audit(
    frame: pd.DataFrame,
    *,
    split_unit_col: str = "market_id",
    split_col: str = "split",
    family_col: str | None = "family_id",
) -> dict[str, Any]:
    """Summarize split integrity for release manifests."""
    if frame.empty or split_unit_col not in frame.columns or split_col not in frame.columns:
        return {
            "split_unit": split_unit_col,
            "rows_by_split": {},
            "units_by_split": {},
            "units_with_multiple_splits": 0,
            "pairwise_unit_overlap": {},
            "family_overlap": None if family_col else None,
        }

    units = (
        frame.loc[:, [split_unit_col, split_col]]
        .dropna(subset=[split_unit_col, split_col])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    rows_by_split = counts_by_split(frame, split_col=split_col)
    units_by_split = counts_by_split(units, split_col=split_col)
    unit_split_sizes = units.groupby(split_unit_col, sort=True)[split_col].nunique()
    unit_overlap = int((unit_split_sizes > 1).sum())

    split_to_units = {
        str(split_value): set(split_frame[split_unit_col].astype(str))
        for split_value, split_frame in units.groupby(split_col, sort=True)
    }
    pairwise_overlap: dict[str, int] = {}
    for left, right in combinations(sorted(split_to_units), 2):
        pairwise_overlap[f"{left}__{right}"] = int(len(split_to_units[left] & split_to_units[right]))

    family_overlap: dict[str, Any] | None = None
    if family_col is not None and family_col in frame.columns:
        families = (
            frame.loc[:, [family_col, split_col]]
            .dropna(subset=[family_col, split_col])
            .assign(**{family_col: lambda df: df[family_col].astype(str).str.strip()})
        )
        families = families.loc[families[family_col] != ""].drop_duplicates().reset_index(drop=True)
        if families.empty:
            family_overlap = {
                "families_by_split": {},
                "families_with_multiple_splits": 0,
                "pairwise_family_overlap": {},
                "overlapping_family_ids_sample": [],
            }
        else:
            family_split_sizes = families.groupby(family_col, sort=True)[split_col].nunique()
            overlapping_families = sorted(
                family_split_sizes.loc[family_split_sizes > 1].index.astype(str).tolist()
            )
            split_to_families = {
                str(split_value): set(split_frame[family_col].astype(str))
                for split_value, split_frame in families.groupby(split_col, sort=True)
            }
            family_pairwise_overlap: dict[str, int] = {}
            for left, right in combinations(sorted(split_to_families), 2):
                family_pairwise_overlap[f"{left}__{right}"] = int(
                    len(split_to_families[left] & split_to_families[right])
                )
            family_overlap = {
                "families_by_split": counts_by_split(families, split_col=split_col),
                "families_with_multiple_splits": int(len(overlapping_families)),
                "pairwise_family_overlap": family_pairwise_overlap,
                "overlapping_family_ids_sample": overlapping_families[:20],
            }

    return {
        "split_unit": split_unit_col,
        "rows_by_split": rows_by_split,
        "units_by_split": units_by_split,
        "units_with_multiple_splits": unit_overlap,
        "pairwise_unit_overlap": pairwise_overlap,
        "family_overlap": family_overlap,
    }
