# Benchmark Notebooks

These notebooks demonstrate the release-facing benchmark API only.

They are intentionally limited to:

- local frozen artifact paths under `benchmark_releases/...`
- `load_terminal`, `load_decisiveness`, `load_repricing`
- leakage-safe `benchmark.input_frame(...)` and `benchmark.targets(...)`
- `evaluate_terminal`, `evaluate_decisiveness`, `evaluate_repricing`
- reference baselines from `polymarket_research.benchmarks.baselines`
- `benchmark_manifest_summary(...)` from `polymarket_research.benchmarks.audit.reporting`
- optional plotting helpers from `polymarket_research.benchmarks.visualization.plotting`

They do not import:

- canonical builders
- raw DB loaders
- benchmark builders
- notebook-only legacy helpers

Available demos:

- `1_terminal_public_api_demo.ipynb`
- `2_repricing_public_api_demo.ipynb`
- `3_decisiveness_public_api_demo.ipynb`

Each notebook demonstrates both `polymarket` and `kalshi` in sequence.

Set `ARTIFACT_ROOT` in the first notebook cell to the local `benchmark_releases`
directory. Each notebook then loads explicit bundle directories such as
`ARTIFACT_ROOT / "polymarket" / "terminal" / "v1"`.
