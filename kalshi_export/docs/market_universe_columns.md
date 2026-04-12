# Market Universe Columns

The table below describes the intended `market_universe` columns for `kalshi_export`.

This table is the enriched source-of-truth snapshot for Kalshi market metadata after the initial `raw_markets` stage.

Unlike `raw_markets`, this table is expected to merge the market-centric base table with targeted enrichment from the separate `event_metadata` table. It should preserve useful venue-native fields while also exposing normalized fields that make downstream research easier.

| column | meaning | reason |
|---|---|---|
| `market_id` | Stable local market identifier, for example `kalshi:<ticker>` | Primary market key used across the local dataset |
| `source` | Exchange / venue identifier, always `kalshi` in this export | Makes multi-source expansion easier later |
| `venue_market_id` | Native Kalshi market ticker | Main venue-native join key |
| `event_id` | Stable local event identifier, for example `kalshi:event:<event_ticker>` | Primary event join key |
| `venue_event_id` | Native Kalshi event ticker | Needed for joins back to event metadata |
| `series_ticker` | Kalshi series identifier | Useful for recurring-series analysis and family grouping |
| `ticker` | Kalshi market ticker | Human-readable native market id retained for debugging and export parity |
| `event_ticker` | Kalshi event ticker | Useful for event-level joins and diagnostics |
| `title` | Raw Kalshi market title | Main venue-native semantic text field |
| `question` | Normalized question text, initially mapped from `title` | Main semantic signal consumed by research code |
| `subtitle` | Kalshi market subtitle | Useful extra market context when present |
| `yes_sub_title` | Venue label for the `Yes` side | Useful for structured and multi-leg market interpretation |
| `no_sub_title` | Venue label for the `No` side | Useful for structured and multi-leg market interpretation |
| `market_type` | Kalshi market type, typically `binary` | Needed for filtering and compatibility checks |
| `status` | Venue market status | Primary market-state flag |
| `event_title` | Parent event title | Human-readable event context |
| `event_sub_title` | Parent event subtitle | Useful for richer event context |
| `kalshi_category` | Raw Kalshi event category | Direct venue taxonomy; should be preserved as-is |
| `mutually_exclusive` | Whether the parent event is mutually exclusive | Useful for group / family reasoning |
| `strike_period` | Event strike period metadata | Useful for recurring series and schedule interpretation |
| `rules_primary` | Primary settlement / market rules text | Important for research, filtering, and auditability |
| `rules_secondary` | Secondary settlement / market rules text | Useful for detailed interpretation and resolution edge cases |
| `can_close_early` | Whether the market may close before its nominal expiration | Useful for lifecycle analysis |
| `early_close_condition` | Condition under which early close may occur | Useful for diagnostics and structured filtering |
| `is_provisional` | Venue provisional-state flag | Useful as a data-quality / status diagnostic |
| `result` | Venue-reported final result, typically `yes` / `no` when resolved | Main raw resolution field |
| `settlement_value_dollars` | Venue-reported settlement value | Useful for audit and non-binary edge cases |
| `created_at` | Market creation timestamp | Needed for temporal cutoffs and panel construction |
| `updated_at` | Last venue update timestamp | Useful for freshness and change diagnostics |
| `open_time` | Venue market open timestamp | Useful for lifecycle analysis |
| `close_time` | Venue market close timestamp | Useful for lifecycle analysis |
| `expected_expiration_time` | Venue expected expiration timestamp | Useful for schedule diagnostics |
| `expiration_time` | Venue expiration timestamp | Main candidate for normalized market end time |
| `latest_expiration_time` | Venue latest expiration timestamp | Useful when markets can roll or extend |
| `settlement_ts` | Venue settlement timestamp | Useful for resolution-timing analysis |
| `last_price_dollars` | Latest market price | Useful for metadata inspection and sanity checks |
| `previous_price_dollars` | Previous reference price | Useful for diagnostics and rough market-state inspection |
| `yes_bid_dollars` | Current best `Yes` bid | Useful for microstructure context |
| `yes_ask_dollars` | Current best `Yes` ask | Useful for microstructure context |
| `no_bid_dollars` | Current best `No` bid | Useful for microstructure context |
| `no_ask_dollars` | Current best `No` ask | Useful for microstructure context |
| `yes_bid_size_fp` | Current best `Yes` bid size | Useful for market-depth diagnostics |
| `yes_ask_size_fp` | Current best `Yes` ask size | Useful for market-depth diagnostics |
| `volume_num` | Normalized lifetime market volume | Required for candidate filtering and research diagnostics |
| `volume_24h_num` | Normalized recent volume | Useful for recency-weighted candidate filtering |
| `open_interest_num` | Normalized open interest | Useful for market-size diagnostics |
| `liquidity_dollars` | Venue liquidity estimate | Useful for quality and microstructure diagnostics |
| `notional_value_dollars` | Venue notional value field | Useful for diagnostics and structured-market interpretation |
| `response_price_units` | Venue price units descriptor | Useful for price normalization checks |
| `price_level_structure` | Venue price-level structure descriptor | Useful for microstructure interpretation |
| `tick_size` | Venue tick size | Useful for price-grid diagnostics |
| `strike_type` | Venue strike type | Useful for structured-market detection and filtering |
| `floor_strike` | Venue lower strike boundary | Useful for structured-market interpretation |
| `cap_strike` | Venue upper strike boundary | Useful for structured-market interpretation |
| `functional_strike` | Venue functional strike text | Useful for structured-market interpretation |
| `custom_strike_json` | JSON-serialized custom strike payload | Preserves venue-native structured strike metadata |
| `mve_collection_ticker` | Multivariate event collection ticker when present | Useful for excluding or grouping collection-style markets |
| `mve_selected_legs_json` | JSON-serialized multivariate selected legs payload | Preserves venue-native collection structure |
| `description` | Normalized description, typically derived from rules or subtitle | Provides a stable downstream text field for research code |
| `end_date` | Normalized market end timestamp, typically from `expiration_time` or `close_time` | Main lifecycle end field used downstream |
| `final_outcome` | Normalized resolved outcome, typically `yes` / `no` / null | Main downstream label field |
| `final_yes_probability` | Normalized final `Yes` probability, typically `1.0`, `0.0`, or null | Compatibility field for benchmark and labeling code |
| `is_binary` | Whether the market is a plain binary market | Useful for selection and compatibility checks |
| `is_resolved` | Whether the market appears resolved locally | Useful for filtering resolved datasets |
| `is_active` | Whether the market appears active locally | Useful for lifecycle filters |
| `is_closed` | Whether the market appears closed locally | Useful for lifecycle filters |
| `synced_at_utc` | Local ingestion timestamp | Useful for freshness checks and ingestion audit |

Suggested downstream derived fields:

- `family_id` from `series_ticker`, `event_title`, and normalized text
- `is_structured_market` from `strike_type`, `custom_strike_json`, and title patterns
- `is_multivariate_like` from `mve_collection_ticker`, `mve_selected_legs_json`, and title patterns
