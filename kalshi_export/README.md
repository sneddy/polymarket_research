# Kalshi Export

This directory now follows a series-first Kalshi pipeline.

Current implemented stages:

1. `download_series_meta` -> build `raw_series`
2. `series_selection` -> build `selected_series`
3. `download_market_meta` -> build `raw_markets` for `selected_series` from live, historical, or both
4. `market_selection` -> build `selected_markets`
5. `get_history` -> build `minute_candles`, `probabilities`, and `added_markets`

Planned later stages:

6. `enrichment`

## Clean Rerun From Scratch

If you want to fully rebuild the Kalshi pipeline from an empty database:

```bash
cd /Users/sneddy/research/polymarket_research
rm -f db/kalshi_probability_dataset.sqlite
```

Then run the four implemented stages in order:

```bash
cd /Users/sneddy/research/polymarket_research/kalshi_export

conda run -n polymarket python -m scripts.download_series_meta \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs \
  --force-remove

conda run -n polymarket python -m scripts.series_selection \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs \
  --force-remove

conda run -n polymarket python -m scripts.download_market_meta \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs \
  --source-mode both \
  --force-remove

conda run -n polymarket python -m scripts.market_selection \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs \
  --min-volume 20000 \
  --force-remove

conda run -n polymarket python -m scripts.get_history \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs
```

## 1. Download Series Metadata

```bash
cd /Users/sneddy/research/polymarket_research/kalshi_export

conda run -n polymarket python -m scripts.download_series_meta \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs \
  --force-remove
```

This populates `raw_series` from `GET /series`.

## 2. Select Series

```bash
conda run -n polymarket python -m scripts.series_selection \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs \
  --force-remove
```

This populates `selected_series` from `raw_series`.

Current logic first drops series with:

- `frequency = fifteen_min`
- `frequency = hourly`
- `frequency = daily`

Then it keeps only:

- `Entertainment`
- `Elections`
- `Politics`
- `Economics`
- `Companies`
- `Financials`
- `Science and Technology`
- `World`
- `Health`
- `Social`

Then it drops additional short-term junk by simple `title` / `subtitle` deny patterns.

## 3. Download Market Metadata

```bash
conda run -n polymarket python -m scripts.download_market_meta \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs \
  --force-remove
```

This stage:

- reads `selected_series`
- calls `GET /markets?series_ticker=...` in `--source-mode live`
- calls `GET /historical/markets?series_ticker=...` in `--source-mode historical`
- can run both branches in one command with `--source-mode both`
- excludes MVE markets by default
- upserts market rows into `raw_markets`

Optional knobs:

- `--source-mode live|historical|both`
- `--status` for live `/markets`
- `--min-close-date YYYY-MM-DD` for live `/markets`
- `--max-close-date YYYY-MM-DD` for live `/markets`
- `--max-pages-per-series`
- `--include-mve`

Examples:

Live only:

```bash
conda run -n polymarket python -m scripts.download_market_meta \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs \
  --source-mode live \
  --force-remove
```

Historical only, appended into existing `raw_markets`:

```bash
conda run -n polymarket python -m scripts.download_market_meta \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs \
  --source-mode historical
```

Both in one run:

```bash
conda run -n polymarket python -m scripts.download_market_meta \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs \
  --source-mode both \
  --force-remove
```

## 4. Select Markets

```bash
conda run -n polymarket python -m scripts.market_selection \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs \
  --min-volume 20000 \
  --force-remove
```

This stage:

- reads `raw_markets`
- keeps only `market_type = binary`
- keeps only markets with `volume_num >= 20000`
- writes the result into `selected_markets`
- precomputes:
  - `history_start_utc = COALESCE(open_time, created_at)`
  - `history_end_utc = COALESCE(settlement_ts, close_time, end_date)`
  - `history_ready`

The goal is to make `selected_markets` a self-contained operational queue for the future `get_history` stage.

## 5. Download History

```bash
conda run -n polymarket python -m scripts.get_history \
  --db-path ../db/kalshi_probability_dataset.sqlite \
  --log-dir ../logs
```

This stage:

- reads `selected_markets` where `history_ready = 1`
- skips markets already present in `added_markets`
- fetches the live/historical cutoff once at run start
- downloads 1-minute candle history into `minute_candles`
- resamples those candles to 5-minute rows in `probabilities`
- writes one manifest row per market into `added_markets`

Useful knobs:

- `--max-markets`
- `--chunk-days`
- `--force-refresh`

## Current Tables

- `raw_series`
- `selected_series`
- `raw_markets`
- `selected_markets`
- `minute_candles`
- `probabilities`
- `added_markets`

Reference docs:

- [raw_series_columns.md](/Users/sneddy/research/polymarket_research/kalshi_export/docs/raw_series_columns.md)
- [selected_series_columns.md](/Users/sneddy/research/polymarket_research/kalshi_export/docs/selected_series_columns.md)
- [raw_markets_columns.md](/Users/sneddy/research/polymarket_research/kalshi_export/docs/raw_markets_columns.md)
- [minute_candles_columns.md](/Users/sneddy/research/polymarket_research/kalshi_export/docs/minute_candles_columns.md)

Future tables will be populated in later stages:

- `event_metadata`
- `market_universe`
- `raw_trades`
