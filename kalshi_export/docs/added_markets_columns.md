# Added Markets Columns

The table below describes the intended `added_markets` columns for `kalshi_export`.

This table is the operational manifest for history downloads. It records what has already been materialized locally and how that market's probability panel was constructed.

| column | meaning | reason |
|---|---|---|
| `market_id` | Stable local market identifier | Primary market key for download-manifest joins |
| `source` | Exchange / venue identifier | Useful for future multi-source manifests |
| `venue_market_id` | Native Kalshi market ticker | Useful for debugging and re-download jobs |
| `series_ticker` | Kalshi series identifier | Useful for diagnostics and grouped inspection |
| `primary_domain` | Local normalized research domain | Useful for quick SQL diagnostics and downstream summaries |
| `added_at_utc` | Local timestamp when the market history was written | Main download-manifest timestamp |
| `storage_path` | Local storage target, typically the SQLite db path | Useful for audit and debugging |
| `history_source_mode` | How the panel was sourced, for example `candles_live`, `candles_historical`, `candles_split`, or `candles_plus_trades` | Useful for debugging data provenance and split-cutoff behavior |
| `probability_rows` | Number of saved rows in `probabilities` for this market | Main completeness measure for the panel |
| `probability_start_utc` | Earliest saved probability timestamp | Useful for coverage diagnostics |
| `probability_end_utc` | Latest saved probability timestamp | Useful for coverage diagnostics |
| `candle_rows_1m` | Number of normalized one-minute candle rows used to build the panel | Useful for validating candlestick-first ingestion |
| `raw_trade_rows` | Number of saved rows in `raw_trades` for this market | Useful when raw-trade download is enabled |
| `raw_trade_start_utc` | Earliest saved raw-trade timestamp | Useful for raw-trade coverage diagnostics |
| `raw_trade_end_utc` | Latest saved raw-trade timestamp | Useful for raw-trade coverage diagnostics |
| `raw_trades_saved` | Whether raw trades were saved for this market, typically `0` or `1` | Makes raw-trade download optional without ambiguity |
| `cutoff_ts_used` | Historical cutoff timestamp used for the download decision | Useful for debugging live / historical stitching logic |
| `download_warnings_json` | JSON-serialized warnings or anomalies encountered during download | Useful for auditability and troubleshooting |

Notes:

- In `kalshi_export`, `raw_trades_saved` should default to `0` because the primary path is candlesticks-first.
- A market can be complete for probability-panel purposes even when `raw_trades_saved = 0`.
