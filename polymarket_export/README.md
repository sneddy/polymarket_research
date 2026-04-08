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

Prepare the resolved market registry:

```bash
python -m scripts.prepare_meta \
  --db-path ../db/resolved_probability_dataset.sqlite \
  --log-dir ../logs
```

Download 5-minute probability histories for one domain:

```bash
python -m scripts.get_history \
  --category geopolitics \
  --db-path ../db/resolved_probability_dataset.sqlite \
  --log-dir ../logs
```

Run the end-to-end resolved probability export:

```bash
python -m scripts.download_resolved_probability_dataset \
  --category geopolitics \
  --db-path ../db/resolved_probability_dataset.sqlite \
  --log-dir ../logs
```

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

- This export root is intentionally kept close to the old layout to minimize churn.
- The research package under `polymarket_research/` should consume exported local artifacts rather than own ingestion logic.
