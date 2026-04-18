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
conda activate polymarket
cd /Users/sneddy/research/polymarket_research/polymarket_export
```

Before running scripts or notebooks from this subtree, install the editable package from the repository root:

```bash
pip install -r requirements.txt
pip install -e .
pip install -e ./polymarket_export
```

`pip install -e ./polymarket_export` is the part required for imports such as `clients`, `configs`, and `polymarket_registry`. `pip install -e .` installs the top-level `polymarket_research` package and is recommended for full-repo work, but is not strictly required if you only use export-side scripts.

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

This script is now intentionally universe-only: it updates `market_universe` and does not rewrite `selected_markets`, `added_markets`, or `probabilities`.

Build the filtered market registry used by history downloads:

```bash
python -m scripts.market_selection \
  --db-path ../db/resolved_probability_dataset.sqlite \
  --log-dir ../logs
```

This step reads the already saved `market_universe` table from SQLite, applies the current block-filter pipeline, and replaces the filtered `selected_markets` table used by `get_history`. It does not re-download market metadata pages.

The default selection protocol is:
- ultra-short recurring template markets
- financial price-derived or benchmark-derived markets
- attention, speech, and mention-count markets
- weather markets
- sports and esports markets
- final resolved-volume screen with default `--min-resolved-volume 20000`

Tag enrichment is now optional and disabled by default. Enable it only if you explicitly want extra Gamma-derived domain labels attached to the selected registry:

```bash
python -m scripts.market_selection \
  --db-path ../db/resolved_probability_dataset.sqlite \
  --log-dir ../logs \
  --tag_enrichment
```

Download 5-minute probability histories for all currently selected markets:

```bash
python -m scripts.get_history \
  --db-path ../db/resolved_probability_dataset.sqlite \
  --log-dir ../logs
```

`scripts.get_history` is incremental: it only downloads histories for market ids that are present in `selected_markets` but not yet present in `added_markets`.

By default `scripts.get_history` also stores normalized fill-level rows in `raw_trades`. Disable that with `--no-save_trades` if you only want the 5-minute panel.

For a small smoke test:

```bash
python -m scripts.get_history \
  --db-path ../db/resolved_probability_dataset.sqlite \
  --log-dir ../logs \
  --max-markets 5
```

Recommended end-to-end refresh order:

```bash
python -m scripts.download_market_meta \
  --db-path ../db/resolved_probability_dataset.sqlite \
  --max-metadata-pages 10000 \
  --log-dir ../logs

python -m scripts.market_selection \
  --db-path ../db/resolved_probability_dataset.sqlite \
  --log-dir ../logs

python -m scripts.get_history \
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

- Column references:
  [market_universe_columns.md](/Users/sneddy/research/polymarket_research/polymarket_export/docs/market_universe_columns.md)
  [raw_trades_columns.md](/Users/sneddy/research/polymarket_research/polymarket_export/docs/raw_trades_columns.md)
- `scripts.inspect_market_meta` is a raw inspector/export utility for market-universe metadata and ranking stats; it does not touch SQLite.
- `scripts.download_market_meta` only updates `market_universe`; it is no longer responsible for research filtering.
- `scripts.market_selection` is the explicit bridge from `market_universe` to the filtered `selected_markets` registry used by history downloads.
- `scripts.market_selection` no longer stores a category subset; it rebuilds the full selected registry in one pass.
- `scripts.get_history` now reads the whole `selected_markets` table; it no longer takes `--category`.
- Current working tables are `market_universe`, `selected_markets`, `added_markets`, `probabilities`, and `raw_trades`.
- This export root is intentionally kept close to the old layout to minimize churn.
- The research package under `polymarket_research/` should consume exported local artifacts rather than own ingestion logic.
- In practice, `download_market_meta` should usually be run with a large `--max-metadata-pages` value. Small values like `10` only scan a shallow prefix of the market universe.
