# Polymarket Benchmark Release Report

- Version: `diagnostic-24-72-168`
- Generated at: `2026-04-26T07:00:38Z`
- SQLite: `db/resolved_probability_dataset.sqlite`

## Canonical Summary
- Markets: 15662
- Probability rows: 208040673
- Unique families: 12697

## Selection Funnel
- market_universe: 732998
- resolved_binary_candidates: 226916
- with_clob_token_ids: 226916
- without_short_horizon_updown: 226916
- selected_markets_registry: 17247
- Note: The final delta from without_short_horizon_updown to selected_markets_registry includes remaining semantic/tag exclusions from the persisted Polymarket registry selection step.

## Benchmarks
- terminal: rows=28424 markets=11397
