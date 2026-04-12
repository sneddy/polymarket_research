# Kalshi Export

This directory contains the Kalshi ingestion/export scaffold.

The current end-to-end flow is:

1. `download_market_meta` -> build `raw_markets`
2. `market_selection` -> build `selected_markets`
3. `enrichment` -> fetch `event_metadata` and materialize `market_universe`
4. `get_history` -> reserved candlesticks-first history stage

## 1. Metadata Index

Raw market base from `GET /markets`:

```bash
python -m scripts.download_market_meta \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs
```

Live index, narrowed by close date:

```bash
python -m scripts.download_market_meta \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs \
  --min-close-date 2026-01-01
```

Historical raw market base from `GET /historical/markets`:

```bash
python -m scripts.download_market_meta \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs \
  --historical
```

Notes:

- `--historical` switches the source endpoint to `GET /historical/markets`.
- By default the loader preserves existing `raw_markets` rows and upserts new ones. Use `--force-remove` only when you explicitly want a clean rebuild.
- SQLite writes happen incrementally during indexing. Use `--write-batch-pages` to control how often buffered pages are upserted.

## 2. Selection

Current selection is intentionally simple and uses only one criterion:

```bash
python -m scripts.market_selection \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs \
  --min-volume 20000
```

This builds `selected_markets` from `raw_markets`.

## 3. Enrichment

Event enrichment is a separate stage:

```bash
python -m scripts.enrichment \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs
```

This stage:

- fetches missing event rows into `event_metadata`
- updates `selected_markets` with event-level columns
- materializes an enriched `market_universe` for the selected subset

## 4. History

The history stage is scaffolded but not yet implemented:

```bash
python -m scripts.get_history \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs
```

Planned behavior:

- read `selected_markets`
- download 1-minute candles
- build 5-minute `probabilities`
- optionally save `raw_trades`
