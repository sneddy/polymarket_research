"""Build paper-ready benchmark tables from frozen release artifacts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from polymarket_research.benchmarks.evaluation.metrics import evaluate_binary_predictions
from polymarket_research.benchmarks.io.paths import benchmark_bundle_dir, benchmark_report_dir
from polymarket_research.scripts.common import (
    VALID_BENCHMARK_TASKS,
    VALID_SOURCES,
    parse_csv_strings,
    resolve_repo_root,
)


DISPLAY_SOURCE = {
    "polymarket": "Polymarket",
    "kalshi": "Kalshi",
}

TASK_LABELS = {
    "terminal": "Terminal",
    "decisiveness": "Decisive convergence",
    "repricing": "Repricing",
}

MISSING_EXPERIMENT_ARTIFACTS = [
    {
        "id": "foundation_models_predictions",
        "description": "Saved Chronos/TimesFM predictions, evaluator outputs, configs, seeds, and environment metadata.",
    },
    {
        "id": "learned_decisiveness_model",
        "description": "Reproducible learned convergence model training and evaluation artifact.",
    },
    {
        "id": "repricing_logistic_hgb",
        "description": "Reproducible repricing logistic regression and histogram-gradient-boosted baseline artifacts.",
    },
    {
        "id": "slice_holdout_transfer",
        "description": "Stable source of slice labels plus slice-holdout manifests and evaluation outputs.",
    },
    {
        "id": "confidence_intervals_bootstrap",
        "description": "Bootstrap or equivalent uncertainty intervals for reported model comparisons.",
    },
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=fieldnames).to_csv(path, index=False)


def _write_tex(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _tex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _fmt_int(value: object) -> str:
    if value is None:
        return "--"
    return f"{int(value):,}"


def _fmt_float(value: object, digits: int = 3) -> str:
    if value is None:
        return "--"
    numeric = float(value)
    if pd.isna(numeric):
        return "--"
    return f"{numeric:.{digits}f}"


def _fmt_pct(value: object, digits: int = 1) -> str:
    if value is None:
        return "--"
    numeric = float(value)
    if pd.isna(numeric):
        return "--"
    return f"{numeric * 100.0:.{digits}f}\\%"


def _fmt_pct_text(value: object, digits: int = 1) -> str:
    return _fmt_pct(value, digits=digits).replace("\\%", "%")


def _split_count(manifest: dict[str, Any], split: str) -> int:
    return int(manifest.get("split_counts", {}).get(split, 0))


def _event_summary(task: str, manifest: dict[str, Any]) -> str:
    if task in {"terminal", "repricing"}:
        rate = manifest.get("label_stats", {}).get("overall", {}).get("positive_rate")
        return f"positive {_fmt_pct_text(rate)}"

    distribution = manifest.get("ordinal_label_distribution", {}).get("overall", {})
    rows = max(int(manifest.get("rows", 0)), 1)
    parts = []
    for label in ("short", "medium", "long"):
        count = int(distribution.get(label, 0))
        parts.append(f"{label} {_fmt_pct_text(count / rows)}")
    return ", ".join(parts)


def _load_release_data(
    artifact_root: Path,
    *,
    sources: tuple[str, ...],
    version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    release_data: list[dict[str, Any]] = []
    input_files: list[dict[str, str]] = []

    for source in sources:
        report_path = benchmark_report_dir(artifact_root, source=source, version=version) / "release_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing release report: {report_path}")
        report = _read_json(report_path)
        input_files.append({"role": "release_report", "source": source, "path": str(report_path)})

        tasks: dict[str, dict[str, Any]] = {}
        for task in VALID_BENCHMARK_TASKS:
            manifest_path = benchmark_bundle_dir(
                artifact_root,
                source=source,
                task=task,
                version=version,
            ) / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing benchmark manifest: {manifest_path}")
            tasks[task] = _read_json(manifest_path)
            input_files.append(
                {
                    "role": "task_manifest",
                    "source": source,
                    "task": task,
                    "path": str(manifest_path),
                }
            )

        release_data.append({"source": source, "report": report, "tasks": tasks})

    return release_data, input_files


def _release_summary_rows(release_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in release_data:
        source = item["source"]
        summary = item["report"].get("canonical_summary", {})
        rows.append(
            {
                "source": DISPLAY_SOURCE.get(source, source),
                "canonical_markets": int(summary.get("markets", 0)),
                "probability_rows": int(summary.get("probability_rows", 0)),
                "unique_families": int(summary.get("unique_families", 0)),
                "external_covariates_rows": int(summary.get("external_covariates_rows", 0)),
                "download_status_rows": int(summary.get("download_status_rows", 0)),
            }
        )

    total = {
        "source": "Total",
        "canonical_markets": sum(row["canonical_markets"] for row in rows),
        "probability_rows": sum(row["probability_rows"] for row in rows),
        "unique_families": sum(row["unique_families"] for row in rows),
        "external_covariates_rows": max(row["external_covariates_rows"] for row in rows) if rows else 0,
        "download_status_rows": sum(row["download_status_rows"] for row in rows),
    }
    return [*rows, total]


def _task_count_rows(release_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in release_data:
        source = item["source"]
        for task in VALID_BENCHMARK_TASKS:
            manifest = item["tasks"][task]
            rows.append(
                {
                    "source": DISPLAY_SOURCE.get(source, source),
                    "task": TASK_LABELS[task],
                    "examples": int(manifest.get("rows", 0)),
                    "markets": int(manifest.get("markets", 0)),
                    "families": int(manifest.get("families", 0)),
                    "train_examples": _split_count(manifest, "train"),
                    "test_examples": _split_count(manifest, "test"),
                    "event_summary": _event_summary(task, manifest),
                }
            )
    return rows


def _selection_funnel_rows(release_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in release_data:
        source = item["source"]
        funnel = item["report"].get("selection_funnel", {})
        if "stages" in funnel:
            stage_groups = [("market", funnel.get("stages", []))]
        else:
            stage_groups = [
                ("series", funnel.get("series_stages", [])),
                ("market", funnel.get("market_stages", [])),
            ]
        for group, stages in stage_groups:
            for stage in stages:
                rows.append(
                    {
                        "source": DISPLAY_SOURCE.get(source, source),
                        "stage_group": group,
                        "stage": str(stage.get("name", "")),
                        "rows": int(stage.get("rows", 0)),
                    }
                )
    return rows


def _split_audit_rows(release_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in release_data:
        source = item["source"]
        for task in VALID_BENCHMARK_TASKS:
            audit = item["tasks"][task].get("split_audit", {})
            family_overlap = audit.get("family_overlap", {})
            pairwise_unit = audit.get("pairwise_unit_overlap", {})
            pairwise_family = family_overlap.get("pairwise_family_overlap", {})
            rows.append(
                {
                    "source": DISPLAY_SOURCE.get(source, source),
                    "task": TASK_LABELS[task],
                    "split_unit": audit.get("split_unit", ""),
                    "train_units": int(audit.get("units_by_split", {}).get("train", 0)),
                    "test_units": int(audit.get("units_by_split", {}).get("test", 0)),
                    "units_with_multiple_splits": int(audit.get("units_with_multiple_splits", 0)),
                    "train_test_unit_overlap": int(pairwise_unit.get("test__train", 0)),
                    "families_with_multiple_splits": int(family_overlap.get("families_with_multiple_splits", 0)),
                    "train_test_family_overlap": int(pairwise_family.get("test__train", 0)),
                }
            )
    return rows


def _label_event_rows(release_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in release_data:
        source = item["source"]
        for task in VALID_BENCHMARK_TASKS:
            manifest = item["tasks"][task]
            if task in {"terminal", "repricing"}:
                label_stats = manifest.get("label_stats", {})
                for split in ("overall", "train", "test"):
                    stats = label_stats.get(split if split == "overall" else "by_split", {})
                    if split != "overall":
                        stats = stats.get(split, {})
                    rows.append(
                        {
                            "source": DISPLAY_SOURCE.get(source, source),
                            "task": TASK_LABELS[task],
                            "split": split,
                            "rows": int(stats.get("rows", 0)),
                            "metric": "positive_rate",
                            "value": float(stats.get("positive_rate", 0.0)),
                            "display_value": _fmt_pct_text(stats.get("positive_rate")),
                        }
                    )
            else:
                distribution = manifest.get("ordinal_label_distribution", {})
                for split in ("overall", "train", "test"):
                    counts = distribution.get(split if split == "overall" else "by_split", {})
                    if split != "overall":
                        counts = counts.get(split, {})
                    total = sum(int(value) for value in counts.values())
                    rows.append(
                        {
                            "source": DISPLAY_SOURCE.get(source, source),
                            "task": TASK_LABELS[task],
                            "split": split,
                            "rows": int(total),
                            "metric": "short_medium_long",
                            "value": "",
                            "display_value": ", ".join(
                                f"{label} {_fmt_pct_text(int(counts.get(label, 0)) / total if total else 0.0)}"
                                for label in ("short", "medium", "long")
                            ),
                        }
                    )
    return rows


def _terminal_market_baseline_rows(
    artifact_root: Path,
    *,
    sources: tuple[str, ...],
    version: str,
    input_files: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        bundle_dir = benchmark_bundle_dir(artifact_root, source=source, task="terminal", version=version)
        examples_path = bundle_dir / "examples.parquet"
        targets_path = bundle_dir / "targets.parquet"
        if not examples_path.exists() or not targets_path.exists():
            raise FileNotFoundError(f"Missing terminal examples/targets in {bundle_dir}")

        input_files.extend(
            [
                {"role": "terminal_examples", "source": source, "path": str(examples_path)},
                {"role": "terminal_targets", "source": source, "path": str(targets_path)},
            ]
        )

        examples = pd.read_parquet(
            examples_path,
            columns=["market_id", "horizon_hours", "split", "current_yes_probability"],
        )
        targets = pd.read_parquet(targets_path, columns=["market_id", "horizon_hours", "split", "label"])
        gold = targets.loc[targets["split"] == "test"].copy()
        predictions = examples.loc[examples["split"] == "test", ["market_id", "horizon_hours", "current_yes_probability"]].copy()
        predictions = predictions.rename(columns={"current_yes_probability": "pred_prob"})
        scored = evaluate_binary_predictions(
            gold=gold.merge(predictions.rename(columns={"pred_prob": "market_pred_prob"}), on=["market_id", "horizon_hours"], how="left"),
            predictions=predictions,
            split="test",
            group_col="horizon_hours",
            id_columns=("market_id", "horizon_hours"),
            reference_prob_col="market_pred_prob",
        )

        for frame_name, frame in scored.items():
            for record in frame.to_dict(orient="records"):
                row = {
                    "source": DISPLAY_SOURCE.get(source, source),
                    "view": "overall" if frame_name == "overall" else "by_horizon",
                    "horizon_hours": record.get("horizon_hours", ""),
                    "rows": int(record.get("rows", 0)),
                    "log_loss": float(record.get("log_loss", float("nan"))),
                    "brier_score": float(record.get("brier_score", float("nan"))),
                    "roc_auc": float(record.get("roc_auc", float("nan"))),
                    "delta_log_loss_vs_market": float(record.get("delta_log_loss_vs_market", float("nan"))),
                    "market_skill_log_loss": float(record.get("market_skill_log_loss", float("nan"))),
                }
                rows.append(row)
    return rows


def _render_release_summary_table(rows: list[dict[str, Any]]) -> str:
    body = "\n".join(
        f"{_tex_escape(row['source'])} & {_fmt_int(row['canonical_markets'])} & "
        f"{_fmt_int(row['probability_rows'])} & {_fmt_int(row['unique_families'])} \\\\"
        for row in rows
    )
    return rf"""
\begin{{table}}[t]
\centering
\small
\caption{{Current v1 source-level release summary generated from checked-in release reports. Family counts are source-local and are summed in the total row.}}
\label{{tab:generated-release-summary}}
\begin{{tabular}}{{lrrr}}
\toprule
Source & Canonical markets & Probability rows & Source-local families \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def _render_task_counts_table(rows: list[dict[str, Any]]) -> str:
    body = "\n".join(
        f"{_tex_escape(row['source'])} & {_tex_escape(row['task'])} & {_fmt_int(row['examples'])} & "
        f"{_fmt_int(row['markets'])} & {_fmt_int(row['families'])} & {_fmt_int(row['train_examples'])} & "
        f"{_fmt_int(row['test_examples'])} \\\\"
        for row in rows
    )
    return rf"""
\begin{{table*}}[t]
\centering
\small
\caption{{Current v1 task counts by source from frozen benchmark manifests. Label summaries are reported separately in Table~\ref{{tab:generated-label-event-rates}}.}}
\label{{tab:generated-task-counts}}
\begin{{tabular}}{{llrrrrr}}
\toprule
Source & Task & Examples & Markets & Families & Train & Test \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def _render_selection_funnel_table(rows: list[dict[str, Any]]) -> str:
    body = "\n".join(
        f"{_tex_escape(row['source'])} & {_tex_escape(row['stage_group'])} & "
        f"{_tex_escape(row['stage'])} & {_fmt_int(row['rows'])} \\\\"
        for row in rows
    )
    return rf"""
\begin{{table}}[t]
\centering
\small
\caption{{Source-selection funnel used by the current v1 release reports.}}
\label{{tab:generated-source-funnel}}
\begin{{tabular}}{{lllr}}
\toprule
Source & Group & Stage & Rows \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def _render_split_audit_table(rows: list[dict[str, Any]]) -> str:
    body = "\n".join(
        f"{_tex_escape(row['source'])} & {_tex_escape(row['task'])} & {_tex_escape(row['split_unit'])} & "
        f"{_fmt_int(row['train_units'])} & {_fmt_int(row['test_units'])} & "
        f"{_fmt_int(row['train_test_unit_overlap'])} & {_fmt_int(row['train_test_family_overlap'])} \\\\"
        for row in rows
    )
    return rf"""
\begin{{table*}}[t]
\centering
\small
\caption{{Split audit for current v1 task manifests. The split unit is market-level; family overlap is reported as an audit statistic rather than a leakage-safe guarantee.}}
\label{{tab:generated-split-audit}}
\begin{{tabular}}{{lllrrrr}}
\toprule
Source & Task & Split unit & Train units & Test units & Unit overlap & Family overlap \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def _render_label_event_table(rows: list[dict[str, Any]]) -> str:
    body = "\n".join(
        f"{_tex_escape(row['source'])} & {_tex_escape(row['task'])} & {_tex_escape(row['split'])} & "
        f"{_fmt_int(row['rows'])} & {_tex_escape(row['display_value'])} \\\\"
        for row in rows
    )
    return rf"""
\begin{{table*}}[t]
\centering
\small
\caption{{Label and event-rate summaries generated from current v1 task manifests.}}
\label{{tab:generated-label-event-rates}}
\begin{{tabular}}{{lllrl}}
\toprule
Source & Task & Split & Rows & Summary \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def _render_terminal_market_baseline_table(rows: list[dict[str, Any]]) -> str:
    overall_rows = [row for row in rows if row["view"] == "overall"]
    body = "\n".join(
        f"{_tex_escape(row['source'])} & {_fmt_int(row['rows'])} & {_fmt_float(row['log_loss'])} & "
        f"{_fmt_float(row['brier_score'])} & {_fmt_float(row['roc_auc'])} & "
        f"{_fmt_float(row['delta_log_loss_vs_market'])} \\\\"
        for row in overall_rows
    )
    return rf"""
\begin{{table}}[t]
\centering
\small
\caption{{Terminal raw market-price baseline on the current v1 test splits, computed directly from each example's cutoff-safe current probability.}}
\label{{tab:generated-terminal-market-baseline}}
\begin{{tabular}}{{lrrrrr}}
\toprule
Source & Test rows & Log loss & Brier & ROC-AUC & $\Delta$ log loss vs. market \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def _write_outputs(
    out_dir: Path,
    *,
    release_summary_rows: list[dict[str, Any]],
    task_count_rows: list[dict[str, Any]],
    selection_funnel_rows: list[dict[str, Any]],
    split_audit_rows: list[dict[str, Any]],
    label_event_rows: list[dict[str, Any]],
    terminal_market_baseline_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    csv_dir = out_dir / "csv"
    table_dir = out_dir / "tables"

    csv_specs = [
        (
            csv_dir / "release_summary.csv",
            release_summary_rows,
            [
                "source",
                "canonical_markets",
                "probability_rows",
                "unique_families",
                "external_covariates_rows",
                "download_status_rows",
            ],
        ),
        (
            csv_dir / "task_counts_by_source.csv",
            task_count_rows,
            [
                "source",
                "task",
                "examples",
                "markets",
                "families",
                "train_examples",
                "test_examples",
                "event_summary",
            ],
        ),
        (
            csv_dir / "source_selection_funnel.csv",
            selection_funnel_rows,
            ["source", "stage_group", "stage", "rows"],
        ),
        (
            csv_dir / "split_audit.csv",
            split_audit_rows,
            [
                "source",
                "task",
                "split_unit",
                "train_units",
                "test_units",
                "units_with_multiple_splits",
                "train_test_unit_overlap",
                "families_with_multiple_splits",
                "train_test_family_overlap",
            ],
        ),
        (
            csv_dir / "label_event_rates.csv",
            label_event_rows,
            ["source", "task", "split", "rows", "metric", "value", "display_value"],
        ),
        (
            csv_dir / "terminal_market_baseline.csv",
            terminal_market_baseline_rows,
            [
                "source",
                "view",
                "horizon_hours",
                "rows",
                "log_loss",
                "brier_score",
                "roc_auc",
                "delta_log_loss_vs_market",
                "market_skill_log_loss",
            ],
        ),
    ]

    output_files: list[dict[str, str]] = []
    for path, rows, fieldnames in csv_specs:
        _write_csv(path, rows, fieldnames)
        output_files.append({"role": "csv", "path": str(path)})

    tex_specs = [
        (table_dir / "release_summary.tex", _render_release_summary_table(release_summary_rows)),
        (table_dir / "task_counts_by_source.tex", _render_task_counts_table(task_count_rows)),
        (table_dir / "source_selection_funnel.tex", _render_selection_funnel_table(selection_funnel_rows)),
        (table_dir / "split_audit.tex", _render_split_audit_table(split_audit_rows)),
        (table_dir / "label_event_rates.tex", _render_label_event_table(label_event_rows)),
        (table_dir / "terminal_market_baseline.tex", _render_terminal_market_baseline_table(terminal_market_baseline_rows)),
    ]
    for path, text in tex_specs:
        _write_tex(path, text)
        output_files.append({"role": "latex_table", "path": str(path)})

    return output_files


def _attach_hashes(file_records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    hashed: list[dict[str, str]] = []
    seen: set[Path] = set()
    for record in file_records:
        path = Path(record["path"])
        if path in seen:
            continue
        seen.add(path)
        enriched = dict(record)
        enriched["sha256"] = _sha256_file(path)
        hashed.append(enriched)
    return hashed


def build_paper_artifacts(
    *,
    repo_root: Path,
    artifact_root: Path,
    out_dir: Path,
    sources: tuple[str, ...],
    version: str,
) -> dict[str, Any]:
    release_data, input_files = _load_release_data(artifact_root, sources=sources, version=version)

    release_summary_rows = _release_summary_rows(release_data)
    task_count_rows = _task_count_rows(release_data)
    selection_funnel_rows = _selection_funnel_rows(release_data)
    split_audit_rows = _split_audit_rows(release_data)
    label_event_rows = _label_event_rows(release_data)
    terminal_market_baseline_rows = _terminal_market_baseline_rows(
        artifact_root,
        sources=sources,
        version=version,
        input_files=input_files,
    )

    output_files = _write_outputs(
        out_dir,
        release_summary_rows=release_summary_rows,
        task_count_rows=task_count_rows,
        selection_funnel_rows=selection_funnel_rows,
        split_audit_rows=split_audit_rows,
        label_event_rows=label_event_rows,
        terminal_market_baseline_rows=terminal_market_baseline_rows,
    )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "repo_root": str(repo_root),
        "artifact_root": str(artifact_root),
        "sources": list(sources),
        "version": version,
        "input_files": _attach_hashes(input_files),
        "output_files": _attach_hashes(output_files),
        "missing_experiment_artifacts": MISSING_EXPERIMENT_ARTIFACTS,
        "summary": {
            "release_summary": release_summary_rows,
            "task_counts_by_source": task_count_rows,
            "terminal_market_baseline_overall": [
                row for row in terminal_market_baseline_rows if row["view"] == "overall"
            ],
        },
    }
    manifest_path = out_dir / "paper_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None, help="Repository root. Defaults to auto-detection.")
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Directory containing source release folders. Defaults to <repo>/benchmark_releases.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <repo>/writing/benchmark/generated.",
    )
    parser.add_argument(
        "--sources",
        default=",".join(VALID_SOURCES),
        help="Comma-separated source names to include.",
    )
    parser.add_argument("--version", default="v1", help="Benchmark release version to read.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    artifact_root = Path(args.artifact_root) if args.artifact_root else repo_root / "benchmark_releases"
    if not artifact_root.is_absolute():
        artifact_root = repo_root / artifact_root
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "writing" / "benchmark" / "generated"
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    sources = parse_csv_strings(args.sources)
    unknown_sources = sorted(set(sources) - set(VALID_SOURCES))
    if unknown_sources:
        valid = ", ".join(VALID_SOURCES)
        raise ValueError(f"Unknown sources {unknown_sources}. Expected one of: {valid}.")
    if not sources:
        raise ValueError("At least one source must be provided.")

    manifest = build_paper_artifacts(
        repo_root=repo_root,
        artifact_root=artifact_root,
        out_dir=out_dir,
        sources=sources,
        version=str(args.version),
    )
    print(f"[paper artifacts] wrote {out_dir}")
    print(f"[paper artifacts] inputs: {len(manifest['input_files'])}")
    print(f"[paper artifacts] outputs: {len(manifest['output_files'])}")


if __name__ == "__main__":
    main()
