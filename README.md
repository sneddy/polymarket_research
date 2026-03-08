# polymarket_research

Reusable, debuggable (non-daemon) Python tools for Polymarket research:

- Full historical trades (orderbook subgraph)
- Market metadata universe + ranking (Gamma Markets API)
- Live order book recording (CLOB WebSocket + REST snapshot)
- News search (GDELT 2.1 DOC API)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main Components

- `clients/gamma_client.py`: Gamma/Data API wrapper + URL-to-market resolution.
- `clients/orderbook_subgraph_client.py`: GraphQL client for full trade history.
- `collectors/trades_collector.py`: Full-history trades downloader with normalization + progress.
- `collectors/markets_collector.py`: Market-universe download, summary stats, market ranking.
- `collectors/orderbook_recorder.py`: Live CLOB recorder (sync + async APIs).
- `collectors/news_collector.py`: News search wrapper.
- `storage/parquet_store.py`: Save/load pandas or polars DataFrames to parquet.

## Notebooks

- `examples/download_trades.ipynb`
- `examples/record_orderbook.ipynb`

## 1) Resolve Market(s) From Polymarket URL

```python
from clients.gamma_client import GammaClient

gamma = GammaClient()
url = "https://polymarket.com/event/fed-decision-in-march-885"

markets = gamma.resolve_markets_from_polymarket_url(url)
[(i, m.get("slug"), m.get("conditionId")) for i, m in enumerate(markets)]
```

Notes:
- `/market/<slug>` resolves to one market.
- `/event/<slug>` can resolve to multiple markets; choose with `market_index` or `market_slug` in collectors.

## 2) Download Full Trade History

`TradesCollector` downloads full history from the orderbook subgraph (not Data API pagination), so it is not constrained by Data API offset limits.

### CLI

By condition id:

```bash
python -m examples.download_trades \
  --market-id 0x... \
  --out data/trades.parquet
```

By Polymarket URL:

```bash
python -m examples.download_trades \
  --url "https://polymarket.com/event/fed-decision-in-march-885" \
  --market-index 0 \
  --out data/trades.parquet
```

Useful flags:
- `--start-date "2025-01-01T00:00:00Z"`
- `--limit 500`
- `--max-pages 50`
- `--market-slug <slug>`

### Python

```python
from clients.gamma_client import GammaClient
from collectors.trades_collector import TradesCollector

gamma = GammaClient()
collector = TradesCollector(gamma)

# Option A: by condition id
df = collector.download_all_trades(
    market_id="0x...",
    start_date=None,
    limit=500,
    max_pages=None,
    show_progress=True,
    estimate_total=True,
)

# Option B: from URL (+ market picker when /event/ has multiple markets)
# df = collector.download_all_trades_from_url(
#     "https://polymarket.com/event/fed-decision-in-march-885",
#     market_index=0,
# )
```

### Trade output schema

Normalized columns:
- `timestamp_utc` (UTC datetime)
- `price`
- `size`
- `outcome`
- `transaction_hash`

### Progress behavior

- Progress bars are enabled by default (`show_progress=True`).
- Total is estimated by default (`estimate_total=True`) via subgraph count probes.
- If the active subgraph deployment cannot provide a count, downloads still work but total may be unknown (`None`), so the bar runs open-ended.

## 3) Market Metadata + Ranking

`MarketsCollector.download_market_meta(...)` returns:
- `markets`: full market universe DataFrame
- `summary`: aggregate stats dict
- `top_markets`: ranked subset for discovery

### CLI

```bash
python -m examples.market_meta \
  --top 50 \
  --out-markets data/markets.parquet \
  --out-top data/top_markets.parquet \
  --out-summary data/market_summary.json
```

Useful flags:
- `--active-only`
- `--limit 200`
- `--max-pages 10`
- `--min-liquidity 10000`
- `--min-volume-24h 5000`
- `--min-created-at "2025-01-01T00:00:00Z"`
- `--no-progress`
- `--no-estimate`
- `--frame-type pandas` or `--frame-type polars`

Active only:

```bash
python -m examples.market_meta --active-only --top 25
```

Recent markets only (creation date cutoff):

```bash
python -m examples.market_meta \
  --min-created-at "2025-01-01T00:00:00Z" \
  --top 25
```

Finished/closed questions example:

```python
from clients.gamma_client import GammaClient
from collectors.markets_collector import MarketsCollector

mc = MarketsCollector(GammaClient())
finished = mc.download_market_meta(
    include_active=True,
    include_inactive=False,
    closed="true",
    min_created_at="2025-01-01T00:00:00Z",
    top_n=30,
    show_progress=True,
    estimate_total=True,
)
```

### Python

```python
from clients.gamma_client import GammaClient
from collectors.markets_collector import MarketsCollector

gamma = GammaClient()
mc = MarketsCollector(gamma)

report = mc.download_market_meta(
    include_active=True,
    include_inactive=True,
    limit=200,
    max_pages=None,
    top_n=25,
    min_liquidity=None,
    min_volume_24h=None,
    min_created_at="2025-01-01T00:00:00Z",  # optional
    show_progress=True,
    estimate_total=True,
)

markets_df = report["markets"]
summary = report["summary"]
top_markets_df = report["top_markets"]
```

Lower-level APIs:
- `download_markets(active=True/False, ...)`: download one state only.
- `download_market_universe(include_active=..., include_inactive=..., ...)`: download and optionally dedupe both states.
- `summarize_markets(df)`: aggregate stats dictionary.
- `rank_markets(df, top_n=...)`: ranking DataFrame without re-downloading.

### Ranking and `top_n`

- `top_n` limits only `top_markets` output size.
- It does not limit universe download size.
- To reduce download work, use `min_created_at`, `max_pages`, and/or Gamma filters (passed through as `**params`, e.g. `closed="true"`).

### Metadata columns

`markets` contains Gamma market payload keys (normalized to snake_case by default), typically including:
- identifiers: `id`, `condition_id`, `slug`, `question`
- status: `active`, `closed`, `archived`
- timing: `created_at`, `end_date`
- liquidity/volume: `liquidity_clob`, `volume_clob`, `volume24hr_clob`, `volume1wk_clob`, `spread`

`top_markets` adds derived ranking fields:
- `liquidity`, `volume_24h`, `volume_1w`, `volume_total`, `spread`
- `market_score`

### Progress behavior

- Progress bars are enabled by default for market metadata.
- Total estimation is enabled by default (`estimate_total=True`) using offset probes.
- With `min_created_at`, collector estimates a start offset first (binary seek on `createdAt`) to skip older pages quickly.

## 4) Record Live Order Book

### CLI

```bash
python -m examples.record_orderbook \
  --url "https://polymarket.com/event/fed-decision-in-march-885" \
  --market-index 0 \
  --seconds 60 \
  --snapshot \
  --out data/orderbook.parquet
```

Alternative source:

```bash
python -m examples.record_orderbook --id <TOKEN_ID_OR_CONDITION_ID> --seconds 60
```

### Python (sync wrappers)

```python
from clients.gamma_client import GammaClient
from collectors.orderbook_recorder import OrderBookRecorder

gamma = GammaClient()
recorder = OrderBookRecorder(gamma_client=gamma)

recorder.connect()
recorder.subscribe_url("https://polymarket.com/event/fed-decision-in-march-885", market_index=0)

snapshot = recorder.get_snapshot()   # optional
df = recorder.record(60)             # seconds
recorder.save_to_parquet(df, "data/orderbook.parquet")
```

`OrderBookRecorder` also provides async methods (`aconnect`, `asubscribe`, `arecord`, etc.) for notebook event-loop workflows.

## 5) News Search

```python
from clients.news_client import NewsClient
from collectors.news_collector import NewsCollector

news = NewsCollector(NewsClient())

df = news.search(
    query='("Polymarket" OR "prediction market")',
    start_date="2025-01-01T00:00:00Z",
    end_date="2025-01-31T23:59:59Z",
    max_records=250,
)

news.save_to_parquet(df, "data/news.parquet")
```

## Storage

Any collector can persist outputs through `save_to_parquet(...)`.

Direct usage:

```python
from storage.parquet_store import ParquetStore

store = ParquetStore()
# loaded = store.load("data/trades.parquet", frame_type="pandas")
```

## Operational Notes

- No hidden daemons/background loops: collection runs only when called.
- Most classes support endpoint/base URL overrides through config objects.
- If progress bars do not render in notebooks, verify `tqdm` is installed and notebook widget support is enabled.
