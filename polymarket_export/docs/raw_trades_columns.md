# Raw Trades Columns

The table below describes the intended `raw_trades` columns and how to interpret their units.

Important notes:

- `size` is normalized during trade finalization when the collector detects raw base-unit magnitudes. In practice this should be read as contract-size / share units, not raw token base units.
- `price` should be read as the trade price per share, typically in probability-like units for binary markets.
- `fee` is currently stored as returned by the subgraph and is **not** normalized by the same logic used for `size`. Treat it as a raw exchange field until a separate fee-normalization policy is added.

| column | meaning | reason / interpretation |
|---|---|---|
| `trade_id` | Unique fill identifier within a market | Main deduplication key for raw fills |
| `market_id` | Polymarket market identifier | Primary market join key |
| `condition_id` | Polymarket condition id | Needed for joins to market metadata and external trade downloads |
| `asset_id` | Outcome token id involved in the fill | Needed to distinguish `Yes` and `No` token fills |
| `timestamp_utc` | Fill timestamp in UTC | Primary time index for raw-trade analysis |
| `price` | Fill price per share | Read as the trade probability / price level for the selected outcome token |
| `size` | Fill size in normalized share units | Already normalized away from likely raw base units when detected |
| `outcome` | Outcome label, typically `Yes` or `No` | Useful for mapping token-level fills into binary market semantics |
| `transaction_hash` | On-chain transaction hash | Useful for grouping fills that happened in the same transaction |
| `maker` | Maker wallet address | Useful for wallet concentration and whale-style analyses |
| `taker` | Taker wallet address | Useful for wallet concentration and flow analyses |
| `order_hash` | Order hash associated with the fill | Useful for order-level diagnostics and deduplication checks |
| `fee` | Raw fee field from the orderbook subgraph | Stored as-is for now; do not assume it is normalized to the same units as `size` |

Suggested downstream derived fields:

- `notional = price * size`
- `unique_wallet_count` over a time window from `maker` and `taker`
- `fee_normalized` only after an explicit normalization decision
