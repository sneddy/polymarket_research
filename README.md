# polymarket_research

This repository has three separate roles:

- `polymarket_export/` and `kalshi_export/` build the source SQLite datasets.
- `polymarket_research/` materializes canonical data and benchmark releases from those SQLite datasets.
- `benchmark_releases/` stores frozen benchmark artifacts for public loading and evaluation.

The dependency direction is intentionally one-way:

`polymarket_export / kalshi_export -> SQLite -> data.raw -> canonical -> benchmark builders -> benchmark_releases -> loaders/evaluators`

For benchmark users, the public path starts from frozen benchmark bundles, not from raw DB access and not from `CanonicalDataset`.

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install -e ./polymarket_export
pip install -e ./kalshi_export
```

`pip install -e .` installs the `polymarket_research` package.
The two export packages are needed only for internal ingestion and SQLite refresh workflows.

## Build Workflow

Use this when you want to rebuild canonical data and frozen benchmark artifacts from a local SQLite dataset.

Polymarket:

```bash
python -m polymarket_research.scripts.materialize_artifacts \
  --source polymarket \
  --db-path db/resolved_probability_dataset.sqlite \
  --output-dir internal_artifacts/polymarket/canonical_dataset
```

By default, the terminal benchmark is materialized at `24h`, `168h`, and `336h`
horizons (`1d`, `7d`, `14d`). Use `--terminal-horizons-hours 24` if you want a
compact single-horizon release.

Kalshi:

```bash
python -m polymarket_research.scripts.materialize_artifacts \
  --source kalshi \
  --db-path db/kalshi_probability_dataset.sqlite \
  --output-dir internal_artifacts/kalshi/canonical_dataset
```

This does both internal stages:

1. `SQLite/raw -> CanonicalDataset`
2. `CanonicalDataset -> benchmark_releases/<source>/<task>/v1`

If you want raw parquet snapshots as well:

```bash
python -m polymarket_research.scripts.materialize_artifacts \
  --source polymarket \
  --db-path db/resolved_probability_dataset.sqlite \
  --output-dir internal_artifacts/polymarket/canonical_dataset \
  --save-raw-snapshot \
  --raw-snapshot-dir internal_artifacts/polymarket/raw
```

You can also run the stages separately:

```bash
python -m polymarket_research.scripts.build_canonical \
  --source polymarket \
  --db-path db/resolved_probability_dataset.sqlite \
  --output-dir internal_artifacts/polymarket/canonical_dataset
```

```bash
python -m polymarket_research.scripts.build_benchmarks \
  --source polymarket \
  --db-path db/resolved_probability_dataset.sqlite \
  --canonical-dir internal_artifacts/polymarket/canonical_dataset
```

Console entrypoints from `pyproject.toml`:

- `pmr-build-canonical`
- `pmr-build-benchmarks`
- `pmr-materialize-artifacts`

## Public Benchmark API

For benchmark consumers, start from local frozen artifacts:

```python
from pathlib import Path

from polymarket_research.benchmarks import load_terminal
from polymarket_research.benchmarks.baselines import fit_terminal_last_probability_baseline

artifact_root = Path("/path/to/benchmark_releases")
bundle_dir = artifact_root / "polymarket" / "terminal" / "v1"
benchmark = load_terminal(bundle_dir)

train_inputs = benchmark.input_frame(split="train")
test_targets = benchmark.targets(split="test")

baseline = fit_terminal_last_probability_baseline(benchmark, split="train")
report = baseline.evaluate(benchmark, split="test")
print(report["overall"])
```

Stable benchmark-facing entrypoints:

- `load_terminal(path)`
- `load_decisiveness(path)`
- `load_repricing(path)`
- `evaluate_terminal(benchmark, predictions, ...)`
- `evaluate_decisiveness(benchmark, predictions, ...)`
- `evaluate_repricing(benchmark, predictions, ...)`

Terminal evaluation reports include ordinary probabilistic metrics plus
`delta_log_loss_vs_market` and `market_skill_log_loss`, both computed against the
market-implied cutoff probability baseline.

Reference baselines live under `polymarket_research.benchmarks.baselines`.

The same loaders consume both sources. Switch the source path segment to
`artifact_root / "kalshi" / "terminal" / "v1"` to load Kalshi bundles.

Leakage-safe public access follows this convention:

- `benchmark.examples` and `benchmark.input_frame(split=...)` contain observable input metadata and current-state features only.
- `benchmark.targets_frame` and `benchmark.targets(split=...)` contain labels and other future-derived target fields.
- `benchmark.market_timeseries` stores full market histories for compactness; use `history_until(...)` or filter by the example timestamp/cutoff before deriving features.
- Analysis helpers may join inputs and targets for diagnostics, but model training code should start from `input_frame(...)`.

## Artifact Layout

Internal build cache:

```text
internal_artifacts/
  polymarket/
    canonical_dataset/
    raw/                    # optional
  kalshi/
    canonical_dataset/
    raw/                    # optional
```

Frozen benchmark releases:

```text
benchmark_releases/
  polymarket/
    terminal/v1/
    decisiveness/v1/
    repricing/v1/
    reports/v1/
  kalshi/
    terminal/v1/
    decisiveness/v1/
    repricing/v1/
    reports/v1/
```

Each benchmark bundle contains:

- `manifest.json`
- `examples.parquet` with leakage-safe input rows
- `market_timeseries.parquet`
- `targets.parquet` with labels and auxiliary target fields
- `README.md`

Each `manifest.json` is source-aware:

- `source`: `polymarket` or `kalshi`
- `release_name`: for example `polymarket-terminal-24h` or `kalshi-repricing-24h`
- split policy and split audit fields
- task row counts, market counts, target columns, and label summaries

Each source/version report directory contains:

- `release_report.json`
- `release_report.md`

The release report is the best place for paper-facing aggregate counts: source-level selection funnel, canonical summary, task manifests, and bundle paths.

`benchmark_releases/` is the frozen output layer.
`internal_artifacts/` is disposable internal cache.

## Package Map

Public benchmark consumption:

- `polymarket_research/benchmarks/io/loaders.py`
- `polymarket_research/benchmarks/evaluation/evaluators.py`
- `polymarket_research/benchmarks/schemas/{terminal,decisiveness,repricing}.py`
- `polymarket_research/benchmarks/baselines/`
- `polymarket_research/benchmarks/io/paths.py`
- `polymarket_research/benchmarks/audit/`
- `polymarket_research/benchmarks/visualization/`

Build-time benchmark materialization:

- `polymarket_research/benchmarks/builders/`

Internal data layer:

- `polymarket_research/data/raw/`
  Raw source access and DB-backed ingestion.
- `polymarket_research/data/canonical/`
  Canonical normalized substrate.
- `polymarket_research/data/representations/`
  Internal feature/materialization helpers still used by some builders.

Internal orchestration:

- `polymarket_research/scripts/`
  Script entrypoints for canonical and benchmark materialization.

Internal research consumers:

- `frozen_notebooks/`
- `research_notebooks/`
- `polymarket_research/research/`
- `polymarket_research/belief_updating/`

These are not the stable public benchmark API.

## Notebooks

Public API demos live in `polymarket_research/benchmarks/notebooks/`:

- `1_terminal_public_api_demo.ipynb`
- `2_repricing_public_api_demo.ipynb`
- `3_decisiveness_public_api_demo.ipynb`

They use only frozen artifacts, loaders/evaluators, reference baselines, and optional plotting helpers. They demonstrate both `polymarket` and `kalshi`.

Set `ARTIFACT_ROOT` in the first notebook cell to the local `benchmark_releases` directory. When artifacts are downloaded from a hosted release, point `ARTIFACT_ROOT` at the downloaded folder.

Internal notebooks remain under `frozen_notebooks/` and `research_notebooks/`. They are useful for audit, visualization, and research workflows, but they do not define the benchmark release contract.

## Raw Export Pipeline

The raw export packages build and refresh the underlying SQLite datasets:

- `polymarket_export/`
- `kalshi_export/`

Those packages contain source-specific `scripts/`, `clients/`, `collectors/`, registry logic, and docs. That workflow is internal ingestion, not public benchmark consumption.
