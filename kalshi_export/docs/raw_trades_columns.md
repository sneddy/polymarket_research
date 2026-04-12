# Raw Trades Columns

The table below describes the intended `raw_trades` columns for `kalshi_export`.

In `kalshi_export`, raw trades are optional and should be disabled by default. This table exists for microstructure analysis, auditability, and cases where exact trade-level reconstruction is still useful.

| column | meaning | reason / interpretation |
|---|---|---|
| `trade_id` | Unique Kalshi trade identifier | Main deduplication key for raw fills |
| `market_id` | Stable local market identifier | Primary market join key |
| `source` | Exchange / venue identifier | Useful for future multi-source expansion |
| `venue_market_id` | Native Kalshi market ticker | Useful for venue-native joins and debugging |
| `timestamp_utc` | Trade timestamp in UTC | Primary time index for raw-trade analysis |
| `price` | Normalized `Yes`-side trade price | Main cross-source comparable trade price field |
| `size` | Normalized trade size | Should be read as the reported Kalshi trade size field, typically from `count_fp` |
| `side` | Trade side indicator, typically from `taker_side` | Useful for flow analysis and trade-direction diagnostics |
| `yes_price` | Raw `Yes` price as returned by the venue | Useful for preserving source-native fields |
| `no_price` | Raw `No` price as returned by the venue | Useful for preserving source-native fields |
| `trade_status` | Optional venue-native trade status or provenance indicator | Useful if the downloader needs to distinguish live vs historical records |
| `raw_payload_json` | JSON-serialized source payload | Useful for auditability and future schema refinement |

Suggested downstream derived fields:

- `notional = price * size`
- `signed_notional` when a stable side convention is adopted
- `trade_minute` and `trade_bucket_5m` for diagnostics against the candlestick-first probability panel
