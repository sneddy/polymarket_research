# Minute Candles Columns

The table below describes the intended `minute_candles` columns for `kalshi_export`.

This table is the raw candle-level history layer for Kalshi.
It plays the role that `raw_trades` played in the Polymarket pipeline, except that Kalshi history is candles-first rather than trades-first.

| column | meaning | reason |
|---|---|---|
| `market_id` | Stable local market identifier | Primary join key into `selected_markets`, `probabilities`, and `added_markets` |
| `source` | Exchange / venue identifier | Keeps the raw history layer future-safe |
| `venue_market_id` | Native Kalshi market ticker | Useful for debugging and re-download jobs |
| `timestamp_utc` | End timestamp of the 1-minute candle in UTC | Primary time index for raw candle data |
| `yes_open_probability` | First observed `Yes` price in the minute | Raw OHLC component for future feature work |
| `yes_high_probability` | Maximum observed `Yes` price in the minute | Raw OHLC component for future feature work |
| `yes_low_probability` | Minimum observed `Yes` price in the minute | Raw OHLC component for future feature work |
| `yes_close_probability` | Last observed `Yes` price in the minute | Main minute-level price field used for resampling |
| `yes_mean_probability` | Mean `Yes` price in the minute when provided by the API | Useful for future diagnostics and smoothing |
| `volume_num` | Minute candle volume | Used to aggregate `total_size` in the 5-minute panel |
| `open_interest_num` | Minute candle open interest | Useful for future diagnostics and feature work |

Notes:

- `minute_candles` is the raw history layer for Kalshi by default.
- `probabilities` should be treated as the downstream 5-minute research representation built from `minute_candles`.
