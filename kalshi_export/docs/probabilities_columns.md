# Probabilities Columns

The table below describes the intended `probabilities` columns for `kalshi_export`.

This table should remain as close as possible to the existing research contract used by `polymarket_research`, even though the underlying Kalshi ingestion path is candlesticks-first rather than trades-first.

| column | meaning | reason / interpretation |
|---|---|---|
| `market_id` | Stable local market identifier | Primary market join key |
| `timestamp_utc` | Five-minute panel timestamp in UTC | Primary time index used by downstream research |
| `yes_probability` | Normalized `Yes` probability at the panel timestamp | Main target / feature time series |
| `observed_trade` | Whether the bucket had an observed market-data update, typically `0` or `1` | Useful for distinguishing observed vs forward-filled buckets |
| `trade_count` | Number of raw trades observed in the bucket | Should default to `0` in the default candlesticks-first path unless explicit trade stitching is enabled |
| `total_size` | Aggregated bucket size / volume | In the default path this should be derived from one-minute candle volume |
| `last_trade_price` | Latest observed price in the bucket | Typically the five-minute close built from one-minute candle closes |

Interpretation notes:

- `yes_probability` should be constructed from Kalshi minute-candle price data rather than reconstructed from full fills by default.
- `trade_count` is retained for compatibility with existing downstream code, even if it is initially populated conservatively.
- `observed_trade = 1` should mean that the five-minute bucket had a direct observed data point rather than only a forward fill.
