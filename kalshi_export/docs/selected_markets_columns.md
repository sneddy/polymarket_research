# Selected Markets Columns

The table below describes the intended `selected_markets` columns for `kalshi_export`.

This table is the curated research-facing registry built from `market_universe`. It should stay compact enough for downstream work while preserving the metadata most useful for filtering, joins, and labeling.

| column | meaning | reason |
|---|---|---|
| `market_id` | Stable local market identifier | Primary market key used throughout the research layer |
| `source` | Exchange / venue identifier | Keeps the registry future-safe for multi-source work |
| `venue_market_id` | Native Kalshi market ticker | Main venue-native join key |
| `event_id` | Stable local event identifier | Primary event join key |
| `venue_event_id` | Native Kalshi event ticker | Useful for joins back to event metadata |
| `series_ticker` | Kalshi series identifier | Useful for recurring-series and family grouping |
| `ticker` | Kalshi market ticker | Retained for readability and operational debugging |
| `event_ticker` | Kalshi event ticker | Required for deterministic joins into event-level enrichment |
| `question` | Normalized market question text | Main semantic signal for downstream research |
| `description` | Normalized market description text | Useful for richer semantic context |
| `event_title` | Parent event title | Human-readable event context |
| `kalshi_category` | Raw Kalshi category | Preserves venue-native taxonomy without forcing it into research logic |
| `primary_domain` | Local normalized research domain | Main downstream domain field used by benchmarks and analyses |
| `created_at` | Market creation timestamp | Needed for temporal ordering and cutoffs |
| `open_time` | Market open timestamp | Useful for lifecycle analysis |
| `close_time` | Venue close timestamp | Useful for lifecycle analysis and diagnostics |
| `settlement_ts` | Venue settlement timestamp when present | Useful for history routing and resolved-market coverage |
| `end_date` | Normalized market end timestamp | Main downstream lifecycle end field |
| `status` | Venue market status | Useful for diagnostics and lifecycle filtering |
| `market_type` | Venue market type | Used to retain only compatible markets, usually `binary` |
| `is_binary` | Whether the market is treated as binary locally | Useful for compatibility checks |
| `is_resolved` | Whether the market appears resolved locally | Useful for resolved-only dataset construction |
| `is_active` | Whether the market appears active locally | Useful for active-market jobs and diagnostics |
| `is_closed` | Whether the market appears closed locally | Useful for lifecycle filters |
| `mutually_exclusive` | Parent event exclusivity flag | Useful for family / event interpretation |
| `strike_type` | Venue strike type | Useful for structured-market filtering and diagnostics |
| `custom_strike_json` | JSON-serialized custom strike payload | Useful for preserving structured strike metadata |
| `volume_num` | Normalized lifetime market volume | Required for candidate filtering and diagnostics |
| `volume_24h_num` | Normalized recent volume | Useful for recency-sensitive filtering |
| `open_interest_num` | Normalized open interest | Useful for market-size diagnostics |
| `liquidity_dollars` | Venue liquidity estimate | Useful for quality diagnostics |
| `final_outcome` | Normalized resolved outcome | Main label-like field for downstream work |
| `final_yes_probability` | Normalized final `Yes` probability | Compatibility field for benchmark and labeling code |
| `rules_primary` | Primary rules text | Useful for auditability and semantic analysis |
| `rules_secondary` | Secondary rules text | Useful for detailed interpretation and edge cases |
| `selection_reason` | Human-readable or coded reason the market was admitted | Useful for auditability and debugging the selection pipeline |
| `selection_version` | Local selection-protocol version string | Useful when selection logic changes over time |
| `history_start_utc` | Precomputed history start bound | Keeps `get_history` lightweight and deterministic |
| `history_end_utc` | Precomputed history end bound | Keeps `get_history` lightweight and deterministic |
| `history_ready` | Whether the row has the minimum keys required for history download | Lets `get_history` read a self-contained operational queue |
| `synced_at_utc` | Local ingestion / registry build timestamp | Useful for freshness checks and ingestion audit |

Notes:

- `primary_domain` should be a local derived field, not a direct copy of `kalshi_category`.
- `kalshi_category` should be preserved as raw source metadata even if it is not used directly by downstream research.
- `history_start_utc` should default to `COALESCE(open_time, created_at)`.
- `history_end_utc` should default to `COALESCE(settlement_ts, close_time, end_date)`.
