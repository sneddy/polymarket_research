# Belief Updating Experiments

This package now supports two related protocol variants built on one shared manifest.

## 1. MVP outcome protocol

The legacy MVP keeps the original ladder:

1. `stale_only`
2. `stale_plus_raw`
3. `stale_plus_embedding`
4. `stale_plus_corrupted`

Target:
- `label` / `label_terminal_outcome`

Implementation:
- builder: `dataset.py`
- model: `model.py`
- runner: `train.py`

This protocol answers the first proof-of-concept question:

> does fresh non-local family context help terminal prediction when the target market is only observed through a stale local view?

## 2. Main belief-update protocol

The main protocol follows the stricter repository-level `experiment.md`.

Primary target:
- `label_update_logit = logit(p_A(t)) - logit(p_A(t-\Delta))`

Stored auxiliary targets:
- `target_current_probability`
- `target_current_logit`
- `label_stale_error_abs`
- `label_stale_error_ge_015`
- `future_24h_repricing_label`

Main ladder:

1. `stale_only`
2. `stale_plus_raw`
3. `stale_plus_large_embedding`
4. `stale_plus_compact_embedding`
5. `stale_plus_corrupted`
6. `current_local_oracle`

Implementation:
- shared manifest: `dataset.py`
- shared encoder/head: `model.py`
- runner: `main.py`

This protocol answers the cleaner representation question:

> does contemporaneous non-local market context contain a compressible signal that can recover the hidden current-state update of the target market?

## 3. Shared data substrate

Both protocols use the same example construction:

- target market `A`
- context time `t`
- stale time `t-\Delta`
- stale local features at `t-\Delta`
- sibling context snapshots at `t`
- aggregated context summaries for tabular baselines
- optional BTC / ETH return covariates aligned at `t`

The shared manifest lets both experiments run side by side without rebuilding the dataset logic twice.
