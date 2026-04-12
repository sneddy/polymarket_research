# Raw Markets Columns

The table below describes the intended `raw_markets` columns for `kalshi_export`.

This table is the first raw market base built from either `GET /markets` or `GET /historical/markets`.
It is no longer treated as a lightweight index. Instead, it should be treated as the venue-native market-level base table used by selection and later enrichment.

| column | meaning | reason |
|---|---|---|
| `market_id` | Stable local market identifier, for example `kalshi:<ticker>` | Primary local market key |
| `source` | Exchange / venue identifier, always `kalshi` in this export | Keeps the schema future-safe for multi-source work |
| `venue_market_id` | Native Kalshi market ticker | Main venue-native join key |
| `event_id` | Stable local event identifier, for example `kalshi:event:<event_ticker>` | Primary local event join key |
| `venue_event_id` | Native Kalshi event ticker | Useful for targeted later event enrichment |
| `ticker` | Native Kalshi market ticker | Human-readable market id retained for debugging |
| `event_ticker` | Native Kalshi event ticker | Required for later event lookup |
| `title` | Raw market title | Main raw semantic field available from the market endpoints |
| `question` | Normalized question text, initially mapped from `title` | Stable downstream semantic field |
| `subtitle` | Raw market subtitle when present | Useful extra market context |
| `yes_sub_title` | Venue label for the `Yes` side | Useful for structured and grouped market interpretation |
| `no_sub_title` | Venue label for the `No` side | Useful for structured and grouped market interpretation |
| `market_type` | Kalshi market type, typically `binary` | Needed for filtering and compatibility checks |
| `status` | Venue market status | Primary lifecycle filter available at raw market stage |
| `created_at` | Market creation timestamp | Useful for later selection logic |
| `updated_at` | Last venue update timestamp | Useful for freshness checks |
| `open_time` | Market open timestamp | Useful for lifecycle analysis |
| `close_time` | Market close timestamp | Useful for server-side narrowing and lifecycle analysis |
| `expected_expiration_time` | Venue expected expiration timestamp | Useful for schedule diagnostics |
| `expiration_time` | Venue expiration timestamp | Useful for lifecycle analysis |
| `latest_expiration_time` | Venue latest expiration timestamp | Useful when markets can roll or extend |
| `settlement_ts` | Venue settlement timestamp | Useful for resolved-market diagnostics |
| `last_price_dollars` | Latest market price | Useful for sanity checks and rough market-state inspection |
| `previous_price_dollars` | Previous reference price | Useful for diagnostics |
| `yes_bid_dollars` | Current best `Yes` bid | Useful for microstructure context |
| `yes_ask_dollars` | Current best `Yes` ask | Useful for microstructure context |
| `no_bid_dollars` | Current best `No` bid | Useful for microstructure context |
| `no_ask_dollars` | Current best `No` ask | Useful for microstructure context |
| `yes_bid_size_fp` | Current best `Yes` bid size | Useful for depth diagnostics |
| `yes_ask_size_fp` | Current best `Yes` ask size | Useful for depth diagnostics |
| `volume_num` | Normalized lifetime market volume | Useful for candidate filtering |
| `volume_24h_num` | Normalized recent volume | Useful for recency-aware filtering |
| `open_interest_num` | Normalized open interest | Useful for market-size diagnostics |
| `liquidity_dollars` | Venue liquidity estimate | Useful for quality diagnostics |
| `notional_value_dollars` | Venue notional value field | Useful for diagnostics |
| `response_price_units` | Venue price units descriptor | Useful for normalization checks |
| `price_level_structure` | Venue price-level structure descriptor | Useful for microstructure interpretation |
| `tick_size` | Venue tick size | Useful for price-grid diagnostics |
| `strike_type` | Venue strike type | Useful for structured-market detection |
| `floor_strike` | Venue lower strike boundary | Useful for structured-market interpretation |
| `cap_strike` | Venue upper strike boundary | Useful for structured-market interpretation |
| `functional_strike` | Venue functional strike text | Useful for structured-market interpretation |
| `custom_strike_json` | JSON-serialized custom strike payload | Preserves venue-native structured strike metadata |
| `mve_collection_ticker` | Multivariate event collection ticker when present | Useful for later exclusion / grouping |
| `mve_selected_legs_json` | JSON-serialized multivariate selected legs payload | Preserves collection structure |
| `rules_primary` | Primary settlement rules text | Important for later research and filtering |
| `rules_secondary` | Secondary settlement rules text | Useful for edge cases and auditability |
| `can_close_early` | Whether the market may close before nominal expiration | Useful for lifecycle analysis |
| `early_close_condition` | Condition under which early close may occur | Useful for diagnostics |
| `is_provisional` | Venue provisional-state flag | Useful as a quality / status diagnostic |
| `result` | Venue-reported resolved result, typically `yes` / `no` when resolved | Main raw outcome field |
| `settlement_value_dollars` | Venue-reported settlement value | Useful for audit and edge cases |
| `description` | Normalized description, typically derived from rules | Stable downstream text field |
| `end_date` | Normalized lifecycle end timestamp | Useful for downstream research compatibility |
| `final_outcome` | Normalized resolved outcome | Useful for downstream label construction |
| `final_yes_probability` | Normalized final `Yes` probability | Compatibility field for benchmark-style code |
| `is_binary` | Whether the market is a plain binary market | Useful for filtering and compatibility checks |
| `is_resolved` | Whether the market appears resolved locally | Useful for resolved-only selection |
| `is_active` | Whether the market appears active locally | Useful for lifecycle filters |
| `is_closed` | Whether the market appears closed locally | Useful for lifecycle filters |
| `data_source_kind` | Local provenance flag, currently `markets_live_index` or `markets_historical_index` | Useful for debugging and later merges |
| `indexed_at_utc` | Local indexing timestamp | Useful for freshness checks and ingestion audit |

Notes:

- `raw_markets` is intentionally market-centric and should be treated as the raw market base layer.
- Richer event-level fields such as `kalshi_category`, `series_ticker`, and `mutually_exclusive` should be added later through targeted enrichment keyed by `event_ticker`.
