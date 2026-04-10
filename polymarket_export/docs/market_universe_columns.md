# Market Universe Columns

The table below describes the intended `market_universe` columns after the schema cleanup.

| gamma_column | meaning | reason |
|---|---|---|
| `market_id` | Unique market identifier | Primary market key |
| `condition_id` | Market condition id | Required for trade-history and order-book linkage |
| `market_slug` | Human-readable market slug | Useful for logging, debugging, and ad hoc inspection |
| `event_id` | Parent event identifier | Core event-level join key |
| `event_slug` | Parent event slug | Useful for grouping and de-duplication |
| `event_title` | Parent event title | Human-readable event context |
| `event_series_slug` | Event series slug | Useful for recurring-series analysis |
| `question` | Market question text | Main semantic signal for research and categorization |
| `description` | Market-level description | Provides market-specific resolution context |
| `resolution_source` | Resolution source URL or label | Useful for audit and market-type diagnostics |
| `created_at` | Market creation timestamp | Needed for time cutoffs and temporal analysis |
| `end_date` | Market end timestamp | Useful for lifecycle analysis |
| `closed` | Whether the market is closed | Primary market-state flag used in the pipeline |
| `archived` | Whether the market is archived | Retained as a secondary diagnostic flag |
| `volume_num` | Normalized market volume | Required for candidate filtering |
| `liquidity_num` | Normalized market liquidity | Useful for market-quality diagnostics |
| `outcomes` | List of market outcomes | Needed to determine whether a market is binary `Yes/No` |
| `outcome_prices` | List of market outcome prices | Needed to derive `final_outcome` and `final_yes_probability` during selection |
| `clob_token_ids` | Outcome token ids on the CLOB | Required for trade-history and order-book linkage |
| `closed_time` | Market closed timestamp | Useful for resolution-timing analysis |
| `uma_resolution_status` | UMA resolution status | Helps distinguish truly resolved markets from merely closed markets |
| `neg_risk` | Whether the market belongs to a neg-risk group | Useful for recognizing structurally linked markets |
| `neg_risk_market_id` | Shared neg-risk group identifier | Useful for grouping related market rows |
| `group_item_title` | Item label within a grouped market | Improves readability of grouped and repetitive markets |
| `event_description` | Event-level description | Useful for richer event context and de-duplication diagnostics |
| `event_start_time` | Event start timestamp | Useful for temporal analysis |
| `event_score` | Event score, when available | Particularly useful for sports markets |
| `event_period` | Event period, for example `FT` | Useful for sports-market interpretation |
| `event_series_id` | Event series identifier | Useful for grouping recurring event families |
| `event_recurrence` | Event series recurrence, for example `daily` | Useful for repetitive-series analysis |
| `event_series_type` | Event series type | Useful for understanding recurring-series structure |
| `synced_at_utc` | Local ingestion timestamp | Useful for freshness checks and ingestion audit |
