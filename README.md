# polymarket_research

Reusable, debuggable (non-daemon) Python tools to collect:

- market metadata (Gamma Markets API)
- historical trades (Polymarket orderbook subgraph / GraphQL)
- news (GDELT by default)
- live order book (Polymarket CLOB WebSocket + REST snapshot)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Notebooks

- `examples/download_trades.ipynb`
- `examples/record_orderbook.ipynb`

### Download trades (CLI)

```bash
python -m examples.download_trades --market-id 0x... --out data/trades.parquet
```

Or from a Polymarket link:

```bash
python -m examples.download_trades --url "https://polymarket.com/event/fed-decision-in-march-885" --out data/trades.parquet
```

Output schema (minimal): `timestamp_utc`, `price`, `size`, `outcome`, `transaction_hash`.

If an `/event/` URL contains multiple markets, select one:

```bash
python -m examples.download_trades --url "https://polymarket.com/event/fed-decision-in-march-885" --market-index 0 --out data/trades.parquet
```

### Download trades (Python)

Resolve condition id(s) from a Polymarket URL:

```python
from clients.gamma_client import GammaClient

gamma = GammaClient()
url = "https://polymarket.com/event/fed-decision-in-march-885"

markets = gamma.resolve_markets_from_polymarket_url(url)
[(i, m.get("slug"), m.get("conditionId")) for i, m in enumerate(markets)]
```

Then download:

```python
from collectors.trades_collector import TradesCollector

collector = TradesCollector(gamma)

# Option A: resolve condition id yourself
condition_id = markets[0]["conditionId"]
df = collector.download_all_trades(condition_id)

# Option B: let the collector resolve from URL
# df = collector.download_all_trades_from_url(url, market_index=0)
```

### Market Meta (CLI)

Download broad market metadata and a ranked "best markets" shortlist:

```bash
python -m examples.market_meta --top 50 --out-markets data/markets.parquet --out-top data/top_markets.parquet --out-summary data/market_summary.json
```

By default, metadata download shows a progress bar (works in terminal and notebooks) and tries to estimate totals.
- Disable bar with `--no-progress`
- Disable total estimation with `--no-estimate`

`top_n` controls only the size of the ranked output (`top_markets`).  
It does **not** limit how many markets are downloaded; use filters / `max-pages` for that.

Active markets only:

```bash
python -m examples.market_meta --active-only --top 25
```

### Market Meta (Python)

```python
from clients.gamma_client import GammaClient
from collectors.markets_collector import MarketsCollector

gamma = GammaClient()
collector = MarketsCollector(gamma)

report = collector.download_market_meta(
    include_active=True,
    include_inactive=True,  # set False for active-only
    top_n=25,
    show_progress=True,
    estimate_total=True,
)

summary = report["summary"]      # dict with universe stats
markets = report["markets"]      # full market frame
top_markets = report["top_markets"]  # ranked frame for discovery
```

### Record order book (CLI)

```bash
python -m examples.record_orderbook --id <TOKEN_ID_OR_CONDITION_ID> --seconds 60 --out data/orderbook.parquet
```

Or from a Polymarket link:

```bash
python -m examples.record_orderbook --url "https://polymarket.com/event/fed-decision-in-march-885" --market-index 0 --seconds 60 --out data/orderbook.parquet
```

### Record order book (Python)

```python
from clients.gamma_client import GammaClient
from collectors.orderbook_recorder import OrderBookRecorder

gamma = GammaClient()
recorder = OrderBookRecorder(gamma_client=gamma)

url = "https://polymarket.com/event/fed-decision-in-march-885"
recorder.connect()

# Option A: resolve condition id from URL, then subscribe
markets = gamma.resolve_markets_from_polymarket_url(url)
market_index = 0
condition_id = markets[market_index]["conditionId"]
recorder.subscribe(condition_id)

# Option B: subscribe directly from URL (equivalent)
# recorder.subscribe_url(url, market_index=0)
df = recorder.record(60)
```

## Notes

- Everything is explicitly callable; no background loops/threads unless you call `record(...)`.
- Most classes accept base URLs so you can override endpoints if Polymarket changes them.
