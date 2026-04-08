# NeurIPS Submission Notes

## Benchmark Story

This benchmark package frames resolved prediction-market histories as a suite of learning problems over strategic probabilistic signals:

1. `multi-horizon terminal forecasting`
2. `trustworthiness / selective prediction`
3. `large repricing prediction`

The intended scientific angle is not next-tick trading alpha. The benchmark instead studies:

- when market-implied probabilities are already strong terminal forecasters,
- when they should be trusted or abstained on,
- and when a market is about to undergo a large probabilistic revision.

## Submission Strategy

The cleanest way to think about the current package is not as five separate paper candidates, but as:

- one strong paper spine:
  - `01_multi_horizon_terminal_forecasting.ipynb`
  - `02_trustworthiness_selective_prediction.ipynb`
- one bridge benchmark:
  - `03_large_repricing_prediction.ipynb`
- two supporting or diagnostic notebooks:
  - `04_cross_market_information_uptake.ipynb`
  - `05_hybrid_crypto_terminal_forecasting.ipynb`

If this remains primarily a benchmark and evaluation paper, then the current package appears to fit especially well with the NeurIPS `Evaluations & Datasets` track:

- evaluation itself is a scientific object there;
- accepted E&D papers appear in the same proceedings;
- submission does not need to introduce a new model class.

By contrast, if the goal is the NeurIPS main track, then the benchmark package is probably not enough by itself. The stronger main-track version would need one unifying learning problem or method layered on top of these benchmarks rather than only a benchmark release.

## Current Scope

- Data source: `db/resolved_probability_dataset.sqlite`
- Fully supported domains for the current benchmark story: `geopolitics` and `finance_economy`
- Current benchmark package has two layers:
  - a reproducible script runner centered on the original benchmark suite and stored reports under `benchmarks/results/`
  - richer notebook-first benchmark variants that already add text-derived structure, weak graph features, external BTC/ETH covariates, manipulation proxies, and stronger trust formulations
- The canonical core benchmarks remain:
  - `multi-horizon terminal forecasting`
  - `trustworthiness / selective prediction`
  - `large repricing prediction`
- The current notebook story is broader than the old single-domain runner and should be treated as the main source of scientific framing.
- Important reproducibility note:
  - the saved notebook outputs are currently uneven;
  - `04` and `05` contain executed outputs, while `01`-`03` are currently evaluated mainly from benchmark design and framing;
  - at least one mismatch is already visible: `04` and `05` now declare `DOMAINS = ["geopolitics", "finance_economy"]`, but some saved summaries still reflect earlier one-domain runs, so the notebook package should be fully rerun before any submission freeze.

## Protocol

- strict out-of-time rolling splits
- no post-resolution leakage
- explicit snapshot-staleness filtering for terminal/trust tasks
- per-task baseline suites with both market-native and learned models

## Current Baseline Results

The baseline picture now has to be read in two layers.

From the original script-based runner in `benchmarks/results/geopolitics/benchmark_report.md`:

### Terminal forecasting

- Best overall baseline: `market_price`
- Mean metrics:
  - `log_loss = 0.2625`
  - `brier = 0.0823`
  - `roc_auc = 0.9218`

Interpretation:
- raw market probability is already a very strong terminal baseline;
- learned tabular models improve some horizons or some metrics, but do not trivially dominate;
- this is a useful benchmark characteristic rather than a weakness.

### Trustworthiness from the original compact runner

- `confidence_margin` is the strongest simple trust policy at low coverage;
- `learned_trust` becomes competitive in the middle-coverage regime;
- this remains a useful baseline reference, but it no longer reflects the full trust benchmark used in the current notebooks.

### Repricing

- Best overall baseline: `hist_gradient_boosting`
- Mean metrics:
  - `average_precision = 0.1756`
  - `roc_auc = 0.9070`
  - `log_loss = 0.0654`

Interpretation:
- large repricing events are sparse but predictable from recent microstructure-derived features;
- simple heuristics remain competitive, which provides a healthy floor for future multimodal models.

### What the current notebooks add beyond those baseline reports

- `01_multi_horizon_terminal_forecasting.ipynb` now treats terminal forecasting as world-state inference and evaluates text, graph/coherence, external-covariate, and manipulation-aware features across `geopolitics` and `finance_economy`.
- `02_trustworthiness_selective_prediction.ipynb` expands trust from a narrow terminal-error proxy into a composite object including terminal loss, future instability, coherence gaps, and manipulation-sensitive states.
- `03_large_repricing_prediction.ipynb` reframes repricing as posterior revision or incomplete-belief detection rather than generic short-horizon alpha.
- `04_cross_market_information_uptake.ipynb` serves as supporting evidence for external-information uptake.
- `05_hybrid_crypto_terminal_forecasting.ipynb` serves as a diagnostic multimodal stress test rather than a flagship benchmark.

So the old report remains useful as a compact baseline artifact, but it is no longer a full description of the benchmark story you would submit.

## Current Assessment By Notebook

### `01_multi_horizon_terminal_forecasting.ipynb`

This is the best anchor benchmark.

Why it is strong:

- the story is closest to world-state inference rather than micro-alpha;
- multi-horizon design tests early information aggregation instead of only near-resolution calibration;
- it is a natural place to show when text, graph, and coherence features help.

Current concerns:

- the effective sample still appears limited relative to the ambition of the claim;
- this benchmark is the most sensitive to contamination or leakage from lifetime or post-cutoff features;
- any feature such as `probability_rows`, `trade_rows`, `volume_num`, or family-level totals must be audited carefully to ensure they are legitimate snapshot-time inputs.

Current role:

- main benchmark #1

### `02_trustworthiness_selective_prediction.ipynb`

This is currently the most NeurIPS-native angle.

Why it is strong:

- it turns the package from forecasting-only into trustworthy ML;
- selective prediction and abstention fit naturally with uncertainty-aware evaluation;
- the benchmark asks when the market can be treated as a reliable posterior, not only what number it outputs.

Current concerns:

- the setup may still be data-limited because it is tied to a fixed horizon on top of terminal snapshots;
- the current trust target mixes future failure with current incoherence, which is scientifically interesting but also invites reviewer questions about whether the model is predicting future unreliability or simply measuring present inconsistency.

Current role:

- main benchmark #2

### `03_large_repricing_prediction.ipynb`

This is the best bridge benchmark.

Why it is strong:

- it connects raw market traces to visible belief updates;
- the task is easier to sample repeatedly along the life of a market;
- it creates a natural bridge between forecasting and trust.

Current concerns:

- it remains vulnerable to being read as short-horizon event prediction or micro-alpha unless the framing stays tightly focused on posterior revision;
- it should not become the headline claim of the paper;
- it also needs contamination checks and preferably sensitivity analysis across move thresholds.

Current role:

- bridge benchmark, not flagship result

### `04_cross_market_information_uptake.ipynb`

This is currently the most useful executed supporting notebook.

Why it is strong:

- it asks a real scientific question about cross-market information flow;
- it can motivate multimodal evidence uptake without claiming that naive fusion is enough;
- even a modest effect is useful if it shows that external liquid markets sometimes lead slower event repricing.

Current concerns:

- it should remain a supporting experiment rather than a core benchmark;
- the most likely scientific contribution here is not raw hybrid improvement, but evidence that external shocks matter only in some regimes;
- the saved outputs need a clean rerun under the current two-domain configuration.

Current role:

- supporting evidence for external-information uptake

### `05_hybrid_crypto_terminal_forecasting.ipynb`

This is best interpreted as a diagnostic negative control.

Why it is still useful:

- it tests the simplest prior-plus-update idea directly;
- a weak or negative result is scientifically informative because it argues against naive always-on feature concatenation;
- it can justify a later gated or trust-aware multimodal method.

Current concerns:

- this should not be framed as a core paper result;
- the data appear relatively limited for a standalone claim;
- it is highly sensitive to feature hygiene and split design consistency.

Current role:

- diagnostic stress test / negative control

## What Makes This Submission-Useful Already

- clear task definitions
- reproducible runner
- exported datasets and fold-level metrics
- compact baseline suite
- out-of-time evaluation
- notebook + script versions
- tests around benchmark utilities

## What Still Needs To Be Added For A Strong Paper

1. Stronger release-quality integration of multimodal features:
   right now BTC/ETH covariates are already used in the notebooks, but they should be hardened into a cleaner benchmark pipeline and extended to broader macro covariates such as gold, oil, rates, and FX where relevant

2. Better trust-target justification and reporting:
   the notebooks already include instability, coherence, and manipulation-sensitive proxies, but the composite trust target should be presented more formally and reported separately by domain and risk slice

3. Stronger benchmark baselines:
   calibrated models, conformal or selective-prediction baselines, and more explicit event-update baselines for repricing and uptake tasks

4. Benchmark presentation:
   formal task section, contamination discussion, ablations across domains and hard slices, explicit benchmark packaging, and a cleaner connection between the script runner and the richer notebook benchmarks

5. Feature decontamination and leakage audit:
   all snapshot-time tasks need a careful pass over lifetime counters, post-hoc market aggregates, and family-level totals so the benchmark can defend itself under reviewer scrutiny

6. Full reruns and frozen outputs:
   the notebook package should be rerun end-to-end under the current domain settings before any submission claims are treated as final

## Auxiliary Evidence

Two additional notebooks now play a supporting role rather than defining the core benchmark package:

- `04_cross_market_information_uptake.ipynb`
  This is best treated as auxiliary evidence for external-information uptake: can a liquid external market act as a fast evidence channel for slower event markets? It strengthens the multimodal story but should not replace the three core benchmarks.

- `05_hybrid_crypto_terminal_forecasting.ipynb`
  This is best treated as a diagnostic multimodal stress test or negative control: does naive feature concatenation help, or do external signals need to enter through smarter gating and trust mechanisms? It is useful scientifically, but it should not be framed as a flagship benchmark.

## Recommended Paper Positioning

If submitted to NeurIPS, this benchmark should be pitched as:

`Prediction markets as strategic probabilistic sensors: forecasting, trust, and repricing benchmarks from resolved high-frequency event markets.`

That is stronger than positioning it as a finance-alpha system and aligns better with uncertainty, structured forecasting, and trustworthy ML.

For a main-track submission, the stronger version of that pitch would be:

`Can we recover a coherent and trustworthy latent belief state of the world from strategic market traces and external evidence?`

In that framing, `01 + 02` provide the paper spine, `03` provides the belief-update bridge, and `04 + 05` act as supporting evidence and diagnostic negatives.
