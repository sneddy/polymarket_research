from __future__ import annotations

import json

import pandas as pd

from polymarket_research.benchmarks import (
    DecisivenessBenchmarkConfig,
    RepricingBenchmarkConfig,
    TerminalBenchmarkConfig,
    evaluate_decisiveness,
    evaluate_repricing,
    evaluate_terminal,
    load_decisiveness,
    load_decisiveness_release,
    load_repricing,
    load_repricing_release,
    load_terminal,
    load_terminal_release,
)
from polymarket_research.benchmarks.audit.reporting import benchmark_manifest_summary
from polymarket_research.benchmarks.io.paths import benchmark_bundle_dir, benchmark_release_report_dir
from polymarket_research.benchmarks.utils.splits import select_split_rows
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
from polymarket_research.benchmarks.baselines import (
    fit_decisiveness_majority_baseline,
    fit_repricing_train_rate_baseline,
    fit_terminal_last_probability_baseline,
    fit_terminal_train_rate_baseline,
)
from polymarket_research.benchmarks.builders import (
    build_decisiveness_analysis_frame,
    build_decisiveness_from_canonical,
    build_repricing_analysis_frame,
    build_repricing_from_canonical,
    build_terminal_from_canonical,
)
from polymarket_research.benchmarks.io.paths import benchmark_release_dir
from polymarket_research.data.canonical import CanonicalDataset
from polymarket_research.scripts.build_benchmarks import build_benchmark_releases


def _linear_values(start: float, end: float, points: int) -> list[float]:
    if points <= 1:
        return [float(start)]
    step = (float(end) - float(start)) / float(points - 1)
    return [float(start) + step * index for index in range(points)]


def _synthetic_canonical() -> CanonicalDataset:
    market_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    base = pd.Timestamp("2026-01-01T00:00:00Z")

    market_specs = [
        ("m1", base, 0.55, 0.98, 1.0),
        ("m2", base + pd.Timedelta(days=1), 0.45, 0.02, 0.0),
        ("m3", base + pd.Timedelta(days=2), 0.40, 0.97, 1.0),
    ]

    for market_id, created_at, start_prob, end_prob, final_yes_probability in market_specs:
        end_date = created_at + pd.Timedelta(hours=4)
        market_rows.append(
            {
                "market_id": market_id,
                "market_slug": market_id,
                "question": f"Question for {market_id}",
                "description": f"Description for {market_id}",
                "created_at": created_at,
                "end_date": end_date,
                "final_yes_probability": final_yes_probability,
                "platform_category": "news",
                "research_category": "news",
                "family_id": f"family-{market_id}",
                "volume_num": 1000.0,
                "trade_rows": 49,
                "probability_rows": 49,
            }
        )

        timestamps = pd.date_range(created_at, end_date, freq="5min", tz="UTC")
        probabilities = _linear_values(start_prob, end_prob, len(timestamps))
        for timestamp_utc, yes_probability in zip(timestamps, probabilities, strict=False):
            probability_rows.append(
                {
                    "market_id": market_id,
                    "timestamp_utc": timestamp_utc,
                    "yes_probability": float(yes_probability),
                    "observed_trade": 1,
                    "trade_count": 1,
                    "total_size": 100.0,
                    "last_trade_price": float(yes_probability),
                }
            )

    markets = pd.DataFrame(market_rows)
    probabilities = pd.DataFrame(probability_rows).sort_values(["market_id", "timestamp_utc"], kind="stable").reset_index(drop=True)
    return CanonicalDataset(markets=markets, probabilities=probabilities)


def _fit_split_with_rows(target_fn) -> str:
    for split in ("train", "test"):
        if not target_fn(split).empty:
            return split
    raise ValueError("Expected at least one non-empty split.")


def _assert_no_columns(frame: pd.DataFrame, forbidden_columns: set[str]) -> None:
    leaked = forbidden_columns.intersection(set(frame.columns))
    assert leaked == set()


def test_benchmark_release_dir_contract():
    assert benchmark_bundle_dir("/artifacts", source="polymarket", task="terminal").as_posix() == "/artifacts/polymarket/terminal/v1"
    assert benchmark_release_dir("/repo", source="polymarket", task="terminal").as_posix() == "/repo/benchmark_releases/polymarket/terminal/v1"
    assert benchmark_release_dir("/repo", source="kalshi", task="repricing", version="v2").as_posix() == "/repo/benchmark_releases/kalshi/repricing/v2"
    assert benchmark_release_report_dir("/repo", source="kalshi", version="v2").as_posix() == "/repo/benchmark_releases/kalshi/reports/v2"


def test_artifact_root_loaders_and_manifest_summary(tmp_path):
    canonical = _synthetic_canonical()
    artifact_root = tmp_path / "benchmark_releases"

    terminal = build_terminal_from_canonical(
        canonical,
        TerminalBenchmarkConfig(
            horizons_hours=(1,),
            split_timestamp_utc=pd.Timestamp("2026-01-02T12:00:00Z"),
        ),
        source="kalshi",
    )
    decisiveness = build_decisiveness_from_canonical(
        canonical,
        DecisivenessBenchmarkConfig(
            sample_every_hours=1,
            min_history_points=2,
            min_prefix_age_hours=0.25,
            min_time_to_decisive_hours=0.25,
            split_timestamp_utc=pd.Timestamp("2026-01-02T02:00:00Z"),
        ),
        source="kalshi",
    )
    repricing = build_repricing_from_canonical(
        canonical,
        RepricingBenchmarkConfig(
            future_horizon_hours=1,
            lookback_hours=1,
            sample_every_hours=1,
            move_threshold=0.05,
            split_timestamp_utc=pd.Timestamp("2026-01-02T02:00:00Z"),
            attach_external_shocks=False,
        ),
        source="kalshi",
    )

    terminal.save(benchmark_bundle_dir(artifact_root, source="kalshi", task="terminal"))
    decisiveness.save(benchmark_bundle_dir(artifact_root, source="kalshi", task="decisiveness"))
    repricing.save(benchmark_bundle_dir(artifact_root, source="kalshi", task="repricing"))

    loaded = {
        "terminal": load_terminal_release(artifact_root, source="kalshi"),
        "decisiveness": load_decisiveness_release(artifact_root, source="kalshi"),
        "repricing": load_repricing_release(artifact_root, source="kalshi"),
    }
    summary = benchmark_manifest_summary(loaded)
    assert set(summary["release_name"]) == {
        "kalshi-terminal-1h",
        "kalshi-decisive-belief-tau95",
        "kalshi-repricing-1h",
    }
    assert set(summary["source"]) == {"kalshi"}


def test_terminal_roundtrip_and_evaluate(tmp_path):
    canonical = _synthetic_canonical()
    benchmark = build_terminal_from_canonical(
        canonical,
        TerminalBenchmarkConfig(
            horizons_hours=(1, 2, 3),
            split_timestamp_utc=pd.Timestamp("2026-01-02T12:00:00Z"),
        ),
        source="kalshi",
    )
    out_dir = tmp_path / "terminal"
    paths = benchmark.save(out_dir)
    assert paths["targets"].exists()
    assert paths["readme"].exists()

    loaded = load_terminal(out_dir)
    assert loaded.manifest()["source"] == "kalshi"
    assert loaded.manifest()["release_name"] == "kalshi-terminal-1h-2h-3h"
    assert loaded.manifest()["target_columns"] == ["market_id", "horizon_hours", "label", "split"]
    assert loaded.manifest()["split_audit"]["units_with_multiple_splits"] == 0
    _assert_no_columns(loaded.examples, {"label"})
    _assert_no_columns(loaded.input_frame(split="test"), {"label"})
    assert "label" in loaded.targets(split="test").columns
    assert loaded.examples.groupby("market_id")["split"].nunique().max() == 1
    assert loaded.examples.groupby("market_id")["horizon_hours"].nunique().min() >= 1

    test_targets = select_split_rows(loaded.targets_frame, "test")
    predictions = test_targets.loc[:, ["market_id", "horizon_hours"]].copy()
    predictions["pred_prob"] = 0.5
    evaluation = evaluate_terminal(loaded, predictions, split="test")
    assert int(evaluation["overall"].iloc[0]["rows"]) == len(test_targets)
    assert {"delta_log_loss_vs_market", "market_skill_log_loss"}.issubset(evaluation["overall"].columns)
    assert {"delta_log_loss_vs_market", "market_skill_log_loss"}.issubset(evaluation["by_horizon"].columns)
    assert set(evaluation["by_horizon"]["horizon_hours"]) == {1, 2, 3}

    fit_split = _fit_split_with_rows(lambda split: select_split_rows(loaded.targets_frame, split))
    baseline = fit_terminal_train_rate_baseline(loaded, split=fit_split)
    baseline_predictions = baseline.predict(loaded, split="test")
    baseline_evaluation = baseline.evaluate(loaded, split="test")
    assert {"market_id", "horizon_hours", "pred_prob"} == set(baseline_predictions.columns)
    assert int(baseline.train_rows) == len(select_split_rows(loaded.targets_frame, fit_split))
    assert int(baseline_evaluation["overall"].iloc[0]["rows"]) == len(test_targets)

    market_baseline = fit_terminal_last_probability_baseline(loaded, split=fit_split)
    market_predictions = market_baseline.predict(loaded, split="test")
    market_evaluation = market_baseline.evaluate(loaded, split="test")
    assert {"market_id", "horizon_hours", "pred_prob"} == set(market_predictions.columns)
    assert market_predictions["pred_prob"].between(0.0, 1.0).all()
    assert int(market_evaluation["overall"].iloc[0]["rows"]) == len(test_targets)
    assert abs(float(market_evaluation["overall"].iloc[0]["delta_log_loss_vs_market"])) < 1e-12
    assert abs(float(market_evaluation["overall"].iloc[0]["market_skill_log_loss"])) < 1e-12
    assert market_evaluation["by_horizon"]["delta_log_loss_vs_market"].abs().max() < 1e-12
    assert market_evaluation["by_horizon"]["market_skill_log_loss"].abs().max() < 1e-12


def test_decisiveness_roundtrip_evaluate_and_analysis_frame(tmp_path):
    canonical = _synthetic_canonical()
    benchmark = build_decisiveness_from_canonical(
        canonical,
        DecisivenessBenchmarkConfig(
            sample_every_hours=1,
            min_history_points=2,
            min_prefix_age_hours=0.25,
            min_time_to_decisive_hours=0.25,
            split_timestamp_utc=pd.Timestamp("2026-01-02T02:00:00Z"),
        ),
        source="kalshi",
    )
    out_dir = tmp_path / "decisiveness"
    paths = benchmark.save(out_dir)
    assert paths["targets"].exists()

    loaded = load_decisiveness(out_dir)
    assert loaded.manifest()["source"] == "kalshi"
    assert loaded.manifest()["release_name"] == "kalshi-decisive-belief-tau95"
    _assert_no_columns(
        loaded.examples,
        {"label", "label_name", "hours_to_decisive", "decisive_side", "decisive_timestamp_utc"},
    )
    assert {"label", "label_name", "hours_to_decisive", "decisive_side", "decisive_timestamp_utc"}.issubset(
        loaded.targets_frame.columns
    )
    _assert_no_columns(
        loaded.input_frame(split="test"),
        {"label", "label_name", "hours_to_decisive", "decisive_side", "decisive_timestamp_utc"},
    )
    assert {"label", "hours_to_decisive"}.issubset(loaded.targets(split="test").columns)
    analysis_frame = build_decisiveness_analysis_frame(loaded)
    assert {"example_id", "confidence_slice", "admissible_decisiveness", "label"}.issubset(analysis_frame.columns)
    assert loaded.manifest()["split_audit"]["units_with_multiple_splits"] == 0
    assert loaded.examples.groupby("market_id")["split"].nunique().max() == 1

    test_targets = select_split_rows(loaded.targets_frame, "test")
    predictions = test_targets.loc[:, ["market_id", "cutoff_timestamp_utc"]].copy()
    predictions["pred_label"] = int(test_targets["label"].mode().iloc[0])
    predictions["pred_hours_to_decisive"] = float(test_targets["hours_to_decisive"].median())
    evaluation = evaluate_decisiveness(loaded, predictions, split="test")
    assert int(evaluation["overall"].iloc[0]["rows"]) == len(test_targets)
    assert "continuous_overall" in evaluation

    fit_split = _fit_split_with_rows(lambda split: select_split_rows(loaded.targets_frame, split))
    baseline = fit_decisiveness_majority_baseline(loaded, split=fit_split)
    baseline_predictions = baseline.predict(loaded, split="test")
    baseline_evaluation = baseline.evaluate(loaded, split="test")
    assert {"market_id", "cutoff_timestamp_utc", "pred_label", "pred_hours_to_decisive"} == set(baseline_predictions.columns)
    assert int(baseline.train_rows) == len(select_split_rows(loaded.targets_frame, fit_split))
    assert int(baseline_evaluation["overall"].iloc[0]["rows"]) == len(test_targets)
    assert "continuous_overall" in baseline_evaluation


def test_repricing_roundtrip_evaluate_and_analysis_frame(tmp_path):
    canonical = _synthetic_canonical()
    benchmark = build_repricing_from_canonical(
        canonical,
        RepricingBenchmarkConfig(
            future_horizon_hours=1,
            lookback_hours=1,
            sample_every_hours=1,
            move_threshold=0.05,
            split_timestamp_utc=pd.Timestamp("2026-01-02T02:00:00Z"),
            attach_external_shocks=False,
        ),
        source="kalshi",
    )
    out_dir = tmp_path / "repricing"
    paths = benchmark.save(out_dir)
    assert paths["targets"].exists()

    loaded = load_repricing(out_dir)
    assert loaded.manifest()["source"] == "kalshi"
    assert loaded.manifest()["release_name"] == "kalshi-repricing-1h"
    _assert_no_columns(loaded.examples, {"label", "future_move"})
    assert {"label", "future_move"}.issubset(loaded.targets_frame.columns)
    _assert_no_columns(loaded.input_frame(split="test"), {"label", "future_move"})
    assert {"label", "future_move"}.issubset(loaded.targets(split="test").columns)
    analysis_frame = build_repricing_analysis_frame(loaded)
    assert {"example_id", "confidence_slice", "target", "admissible_repricing"}.issubset(analysis_frame.columns)
    assert loaded.manifest()["split_audit"]["units_with_multiple_splits"] == 0
    assert loaded.examples.groupby("market_id")["split"].nunique().max() == 1

    test_targets = select_split_rows(loaded.targets_frame, "test")
    predictions = test_targets.loc[:, ["market_id", "timestamp_utc"]].copy()
    predictions["pred_prob"] = 0.5
    evaluation = evaluate_repricing(loaded, predictions, split="test")
    assert int(evaluation["overall"].iloc[0]["rows"]) == len(test_targets)

    fit_split = _fit_split_with_rows(lambda split: select_split_rows(loaded.targets_frame, split))
    baseline = fit_repricing_train_rate_baseline(loaded, split=fit_split)
    baseline_predictions = baseline.predict(loaded, split="test")
    baseline_evaluation = baseline.evaluate(loaded, split="test")
    assert {"market_id", "timestamp_utc", "pred_prob"} == set(baseline_predictions.columns)
    assert int(baseline.train_rows) == len(select_split_rows(loaded.targets_frame, fit_split))
    assert int(baseline_evaluation["overall"].iloc[0]["rows"]) == len(test_targets)


def test_optional_plotting_helpers_return_axes():
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canonical = _synthetic_canonical()

    terminal = build_terminal_from_canonical(
        canonical,
        TerminalBenchmarkConfig(
            horizons_hours=(1,),
            split_timestamp_utc=pd.Timestamp("2026-01-02T12:00:00Z"),
        ),
        source="kalshi",
    )
    terminal_baseline = fit_terminal_last_probability_baseline(terminal, split="train")
    terminal_predictions = terminal_baseline.predict(terminal, split="test")
    terminal_report = terminal_baseline.evaluate(terminal, split="test")
    terminal_prefix_examples = select_terminal_history_prefix_examples(terminal, split="train")
    assert {"current_yes_probability", "cutoff_age_hours", "prefix_rows"}.issubset(terminal.examples.columns)

    repricing = build_repricing_from_canonical(
        canonical,
        RepricingBenchmarkConfig(
            future_horizon_hours=1,
            lookback_hours=1,
            sample_every_hours=1,
            move_threshold=0.05,
            split_timestamp_utc=pd.Timestamp("2026-01-02T02:00:00Z"),
            attach_external_shocks=False,
        ),
        source="kalshi",
    )
    repricing_baseline = fit_repricing_train_rate_baseline(repricing, split="train")
    repricing_predictions = repricing_baseline.predict(repricing, split="test")

    decisiveness = build_decisiveness_from_canonical(
        canonical,
        DecisivenessBenchmarkConfig(
            sample_every_hours=1,
            min_history_points=2,
            min_prefix_age_hours=0.25,
            min_time_to_decisive_hours=0.25,
            split_timestamp_utc=pd.Timestamp("2026-01-02T02:00:00Z"),
        ),
        source="kalshi",
    )
    decisiveness_baseline = fit_decisiveness_majority_baseline(decisiveness, split="train")
    decisiveness_predictions = decisiveness_baseline.predict(decisiveness, split="test")

    axes = [
        plot_metric_by_horizon(terminal_report, metric="log_loss"),
        plot_label_distribution(terminal, split="test"),
        plot_binary_label_rate_by_split(terminal, group_col="horizon_hours"),
        plot_binary_calibration(terminal, terminal_predictions, split="test"),
        plot_precision_recall(repricing, repricing_predictions, split="test"),
        plot_confusion_matrix(decisiveness, decisiveness_predictions, split="test"),
        plot_numeric_distribution(decisiveness, column="hours_to_decisive", split="test"),
        plot_market_history(terminal, market_id=str(terminal.examples.iloc[0]["market_id"])),
        *plot_terminal_visible_history_diagnostics(terminal, split="train"),
        *plot_terminal_history_prefix_examples(terminal, terminal_prefix_examples),
    ]
    assert all(ax is not None for ax in axes)
    plt.close("all")


def test_build_benchmarks_writes_release_report(tmp_path):
    canonical = _synthetic_canonical()
    repo_root = tmp_path / "synthetic_repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[build-system]\nrequires = []\n", encoding="utf-8")
    (repo_root / "polymarket_research").mkdir(exist_ok=True)
    result = build_benchmark_releases(
        canonical=canonical,
        repo_root=repo_root,
        source="polymarket",
        tasks=("terminal", "repricing"),
        version="vtest",
            terminal_config=TerminalBenchmarkConfig(
                horizons_hours=(1,),
                split_timestamp_utc=pd.Timestamp("2026-01-02T12:00:00Z"),
        ),
        repricing_config=RepricingBenchmarkConfig(
            future_horizon_hours=1,
            lookback_hours=1,
            sample_every_hours=1,
            move_threshold=0.05,
            split_timestamp_utc=pd.Timestamp("2026-01-02T02:00:00Z"),
            attach_external_shocks=False,
        ),
    )
    assert result.report_paths is not None
    report_path = result.report_paths["json"]
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["source"] == "polymarket"
    assert report["version"] == "vtest"
    assert report["benchmark_manifests"]["repricing"]["source"] == "polymarket"
    assert report["selection_funnel"]["available"] is False
    assert report["benchmark_manifests"]["repricing"]["split_audit"]["units_with_multiple_splits"] == 0
