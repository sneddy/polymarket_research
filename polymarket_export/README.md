# Polymarket Export

This directory is a near-lifted copy of the original ingestion/export project.

## Purpose

Use this root for:
- downloading resolved Polymarket market metadata and 5-minute probability panels
- downloading raw trade history
- downloading external covariates and external event series
- polling and recording order books

The code is intentionally kept close to the original project structure:
- `config.py`
- `clients/`
- `collectors/`
- `configs/`
- `storage/`
- `scripts/`

## Project Root

Treat `polymarket_export/` as the working root when running ingestion commands.

```bash
cd /Users/sneddy/research/polymarket_research/polymarket_export
```

The code lives in this directory, but the shared storage layer stays at the repository root:
- `../db/`
- `../cached_data/`
- `../logs/`

No local `db/`, `cached_data/`, or `logs/` directories are kept inside `polymarket_export/`.

## Common Entry Points

Download and refresh the broad market universe metadata:

```bash
python -m scripts.download_market_meta \
  --db-path ../db/resolved_probability_dataset.sqlite \
  --max-metadata-pages 10000 \
  --log-dir ../logs
```

By default this refreshes only `closed=true` markets into `market_universe`. Add `--include-active` if you want the full active + closed universe. It does not rewrite the `probabilities` table. The metadata refresh also stores event-level fields when available:
- `event_id`
- `event_slug`
- `event_title`
- `event_series_slug`

This script is now intentionally universe-only: it does not build resolved candidates or rewrite the filtered `markets` registry used by history downloads.

Build the filtered market registry used by history downloads:

```bash
python -m scripts.market_selection \
  --db-path ../db/resolved_probability_dataset.sqlite \
  --log-dir ../logs
```

This step reads the already saved `market_universe` table from SQLite, applies the current local candidate filter plus tag enrichment, and replaces the filtered `markets` table used by `get_history`. It does not re-download market metadata pages.

Download 5-minute probability histories for one domain:

```bash
python -m scripts.get_history \
  --category geopolitics \
  --db-path ../db/resolved_probability_dataset.sqlite \
  --log-dir ../logs
```

`scripts.get_history` is incremental: it only downloads histories for market ids that are present in `markets` but not yet present in `added_markets`.

Download external covariates:

```bash
python -m scripts.download_external_covariates \
  --start-date 2025-01-01 \
  --end-date 2026-01-01 \
  --out ../cached_data/external_covariates
```

Download external event series:

```bash
python -m scripts.get_events \
  --out ../cached_data/external_events \
  --log-dir ../logs
```

Download trade history for one market:

```bash
python -m scripts.download_trades \
  --market-id <condition_id> \
  --out ../cached_data/trades.parquet
```

Record short-lived order book updates:

```bash
python -m scripts.record_orderbook \
  --url <polymarket_url> \
  --seconds 60
```

Poll order book snapshots into SQLite:

```bash
python -m scripts.poll_orderbooks \
  --url <polymarket_url> \
  --db ../db/orderbooks.sqlite
```

## Notes

- `scripts.inspect_market_meta` is a raw inspector/export utility for market-universe metadata and ranking stats; it does not touch SQLite.
- `scripts.download_market_meta` only updates `market_universe`; it is no longer responsible for research filtering.
- `scripts.market_selection` is the explicit bridge from `market_universe` to the filtered `markets` registry used by history downloads.
- This export root is intentionally kept close to the old layout to minimize churn.
- The research package under `polymarket_research/` should consume exported local artifacts rather than own ingestion logic.
- In practice, `download_market_meta` should usually be run with a large `--max-metadata-pages` value. Small values like `10` only scan a shallow prefix of the market universe.
