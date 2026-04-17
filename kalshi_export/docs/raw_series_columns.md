# Raw Series Columns

The table below describes the intended `raw_series` columns for `kalshi_export`.

This table is the first Kalshi taxonomy layer built from `GET /series`.
It is intentionally small and is used to narrow the market universe before downloading raw market rows.

| column | meaning | reason |
|---|---|---|
| `series_ticker` | Native Kalshi series ticker | Primary local series key |
| `title` | Raw Kalshi series title | Main semantic field for series-level filtering |
| `subtitle` | Raw Kalshi series subtitle when present | Useful for extra context and deny-pattern matching |
| `category` | Raw Kalshi category | Primary top-level benchmark gate |
| `tags_json` | JSON-serialized list of Kalshi tags | Useful for denylist / allowlist filtering |
| `frequency` | Kalshi series frequency, for example `daily` or `monthly` | Useful for filtering recurring short-horizon junk |
| `status` | Venue-native series status | Useful for diagnostics |
| `created_at` | Series creation timestamp | Useful for provenance and audit |
| `updated_at` | Series update timestamp | Useful for freshness checks |
| `close_time` | Venue-reported close timestamp when present | Useful for diagnostics |
| `settlement_time` | Venue-reported settlement timestamp when present | Useful for diagnostics |
| `raw_payload_json` | JSON-serialized raw series payload | Keeps the original venue response available without widening the normalized columns |
| `synced_at_utc` | Local ingestion timestamp | Useful for audit and reproducibility |

Notes:

- `raw_series` is the series-first replacement for broad market crawling.
- The main job of `raw_series` is to support category / title / tags / ticker filtering before market download.
