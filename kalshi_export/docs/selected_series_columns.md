# Selected Series Columns

The table below describes the intended `selected_series` columns for `kalshi_export`.

This table is the local benchmark-universe shortlist built from `raw_series`.

| column | meaning | reason |
|---|---|---|
| `series_ticker` | Native Kalshi series ticker | Primary join key into market download |
| `title` | Raw series title | Useful for debugging and benchmark review |
| `subtitle` | Raw series subtitle when present | Useful extra context |
| `category` | Raw Kalshi category retained from `raw_series` | Makes the benchmark universe explainable |
| `tags_json` | JSON-serialized tags list | Useful for auditability and future refinement |
| `frequency` | Series frequency | Useful for understanding recurring series kept in the benchmark |
| `status` | Venue-native status | Useful for diagnostics |
| `selection_reason` | JSON-serialized local explanation of why the series passed the filters | Makes the selection process auditable |
| `selection_version` | Version label for the selection ruleset | Helps compare future iterations of the benchmark universe |
| `synced_at_utc` | Local write timestamp | Useful for audit and reproducibility |

Notes:

- `selected_series` is the only input to the current `download_market_meta` step.
- The current ruleset first removes any series with:
  - `frequency = fifteen_min`
  - `frequency = hourly`
  - `frequency = daily`
- The current ruleset keeps only:
  - `Entertainment`
  - `Elections`
  - `Politics`
  - `Economics`
  - `Companies`
  - `Financials`
  - `Science and Technology`
  - `World`
  - `Health`
  - `Social`
- It then removes additional short-term junk by simple deny patterns over `title`, `subtitle`, and `tags_json`.
