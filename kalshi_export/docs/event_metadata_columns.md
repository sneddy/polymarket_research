# Event Metadata Columns

The table below describes the intended `event_metadata` columns for `kalshi_export`.

This table is the targeted event-enrichment layer keyed by `event_id` / `event_ticker`.
It should be populated from `GET /events/{event_ticker}` after local market selection.

The table should remain event-centric:

- keep useful scalar event fields
- do not store nested markets here
- merge back into market-level tables via `event_ticker`

| column | meaning | reason |
|---|---|---|
| `event_id` | Stable local event identifier, for example `kalshi:event:<event_ticker>` | Primary local event key |
| `source` | Exchange / venue identifier, always `kalshi` in this export | Keeps the schema future-safe |
| `venue_event_id` | Native Kalshi event ticker | Main venue-native join key |
| `event_ticker` | Native Kalshi event ticker | Human-readable event id retained for joins |
| `series_ticker` | Kalshi series identifier | Useful for family grouping and recurring-series analysis |
| `event_title` | Parent event title | Cleaner event-level text than market titles |
| `event_sub_title` | Parent event subtitle | Useful extra event context |
| `kalshi_category` | Raw Kalshi event category | Main venue-native taxonomy field |
| `mutually_exclusive` | Whether the parent event is mutually exclusive | Useful for family structure and grouped-outcome reasoning |
| `strike_period` | Event strike-period metadata | Useful for recurring schedule interpretation |
| `status` | Event status | Useful for diagnostics |
| `created_at` | Event creation timestamp | Useful for event-level audit and filtering |
| `close_time` | Event close timestamp | Useful for lifecycle diagnostics |
| `last_updated_ts` | Venue event update timestamp | Useful for freshness checks |
| `event_url` | Venue event URL when present | Useful for debugging and manual review |
| `rules_primary` | Event-level rule text when present | Useful as a fallback context field |
| `subtitle` | Raw event subtitle alias | Useful for compatibility with inconsistent payload naming |
| `synced_at_utc` | Local enrichment timestamp | Useful for freshness and ingestion audit |

Notes:

- `event_metadata` should store event-native columns only.
- Heavy market-native payloads should remain in `raw_markets` or `market_universe`, not be duplicated here.
