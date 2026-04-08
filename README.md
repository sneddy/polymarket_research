# polymarket_research

`polymarket_research` is a research-oriented toolkit for building datasets and benchmarks around Polymarket and related external signals.

At a high level, the repository supports:

- downloading the Polymarket market universe and resolved-market metadata
- downloading full historical trades and converting them to 5-minute `yes_probability` panels
- recording or polling order books for active markets
- collecting external covariates such as `BTC/ETH`, oil, rates, and FX
- running benchmark tasks for forecasting, trustworthiness, and repricing

The codebase is organized around one principle:

- `clients/` talk to external APIs
- `collectors/` normalize and assemble source-specific data
- `scripts/` are job-style entrypoints / runnable pipeline commands
- `examples/` are demo notebooks only
- `benchmarks/` contains benchmark construction and evaluation code

## Quick Start

If you already use the project conda environment:

```bash
conda activate polymarket
```

Otherwise:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository Map

### Core packages

- `clients/`
  Low-level API clients such as Gamma, orderbook subgraph, Binance, FRED, and GDELT.
- `collectors/`
  Source-aware download + normalization logic.
- `storage/`
  Shared parquet and SQLite storage helpers.
- `configs/`
  Registry/config files for domain grouping and external covariate specs.
- `scripts/`
  CLI entrypoints for dataset-building jobs.
- `examples/`
  Demo notebooks only.
- `benchmarks/`
  Benchmark datasets, notebook analyses, runner, and exported results.

### Main script entrypoints

- `scripts.download_market_meta`
  Refresh the broad market-universe metadata table.
- `scripts.market_selection`
  Build the filtered market registry used by history downloads.
- `scripts.get_history`
  Download full trade history for prepared markets and build 5-minute probability panels.
- `scripts.download_trades`
  Download historical trades for one market.
- `scripts.inspect_market_meta`
  Download and inspect market-universe metadata.
- `scripts.record_orderbook`
  Record live websocket order-book updates.
- `scripts.poll_orderbooks`
  Poll REST order-book snapshots into SQLite.
- `scripts.download_external_covariates`
  Download external market covariates into normalized parquet.
- `scripts.get_events`
  Download normalized SEC EDGAR event-count series for the benchmark window.

## Main Research Workflow

For most research tasks in this repository, the intended pipeline is:

1. download the broad market-universe metadata
2. build the filtered market registry for history download
3. download 5-minute Polymarket histories for one or more research domains
4. optionally download external covariates
5. run benchmarks / experiments on top of the resulting datasets

### What gets created

`scripts.download_market_meta`, `scripts.market_selection`, and `scripts.get_history` write to:

- `db/resolved_probability_dataset.sqlite`

Main tables:

- `market_universe`
  Broad Polymarket metadata universe from the listing endpoint
- `markets`
  Filtered market registry used by the legacy history pipeline
- `added_markets`
  Markets whose history has already been downloaded
- `probabilities`
  5-minute `yes_probability` panels

`scripts.download_external_covariates` writes:

- parquet datasets such as `cached_data/external_covariates/`

`scripts.get_events` writes:

- parquet datasets such as `cached_data/external_events/`

`benchmarks/run_benchmarks.py` writes:

- `benchmarks/results/<domain>/`

## Reproducible Shell Runs

The following commands reflect the main end-to-end jobs currently used in this repository.

## What To Run

| Goal | Run | Main output |
| --- | --- | --- |
| Refresh the broad market universe | `python -m scripts.download_market_meta --db-path db/resolved_probability_dataset.sqlite ...` | `market_universe` table in `db/resolved_probability_dataset.sqlite` |
| Build the filtered market registry | `python -m scripts.market_selection --db-path db/resolved_probability_dataset.sqlite ...` | `markets` table in `db/resolved_probability_dataset.sqlite` |
| Download 5-minute Polymarket panels for one domain | `python -m scripts.get_history --category <domain> --db-path db/resolved_probability_dataset.sqlite` | `added_markets` and `probabilities` tables in `db/resolved_probability_dataset.sqlite` |
| Download full historical trades for one market | `python -m scripts.download_trades --market-id ... --out cached_data/trades.parquet` | `cached_data/trades.parquet` |
| Inspect/download market-universe metadata | `python -m scripts.inspect_market_meta ...` | parquet/JSON exports of market metadata and rankings |
| Record live websocket order-book updates | `python -m scripts.record_orderbook ... --out cached_data/orderbook.parquet` | `cached_data/orderbook.parquet` |
| Poll REST order-book snapshots into SQLite | `python -m scripts.poll_orderbooks ... --db cached_data/orderbooks.sqlite` | SQLite order-book snapshot database |
| Download external covariates | `python -m scripts.download_external_covariates ... --out cached_data/external_covariates` | parquet dataset under `cached_data/external_covariates` |
| Download benchmark-window EDGAR events | `python scripts/get_events.py` | parquet dataset under `cached_data/external_events` |
| Run benchmark suite | `python benchmarks/run_benchmarks.py --domain <domain>` | `benchmarks/results/<domain>/` |

### 1) Download the broad market-universe metadata

```bash
conda activate polymarket
python -m scripts.download_market_meta \
  --db-path db/resolved_probability_dataset.sqlite \
  --min-created-at 2025-01-01T00:00:00Z \
  --max-metadata-pages 10000
```

By default this refreshes only `closed=true` markets. Add `--include-active` to include open markets too.

### 2) Build the filtered market registry from the saved universe

```bash
conda activate polymarket
python -m scripts.market_selection \
  --db-path db/resolved_probability_dataset.sqlite \
  --min-created-at 2025-01-01T00:00:00Z
```

### 3) Download 5-minute Polymarket histories for `geopolitics`

```bash
conda activate polymarket
python -m scripts.get_history \
  --category geopolitics \
  --db-path db/resolved_probability_dataset.sqlite
```

### 4) Download 5-minute Polymarket histories for `finance_economy`

```bash
conda activate polymarket
python -m scripts.get_history \
  --category finance_economy \
  --db-path db/resolved_probability_dataset.sqlite
```

### 5) Download external covariates

Crypto (`BTC/ETH`) uses Binance archive backfill by default. Non-crypto default series use FRED daily data. SEC EDGAR filing-count series are available as opt-in daily covariates.

```bash
conda activate polymarket
python -m scripts.download_external_covariates \
  --start-date 2025-01-01T00:00:00Z \
  --end-date 2026-04-04T00:00:00Z \
  --series-id btc_usd \
  --series-id eth_usd \
  --series-id wti_oil_usd \
  --series-id brent_oil_usd \
  --series-id us_10y_yield \
  --series-id fed_funds_effective \
  --series-id eur_usd \
  --series-id usd_jpy \
  --series-id broad_usd_index \
  --binance-source archive \
  --out cached_data/external_covariates \
  --partition-cols series_id
```

EDGAR example:

```bash
python -m scripts.download_external_covariates \
  --start-date 2025-01-01T00:00:00Z \
  --end-date 2025-03-31T00:00:00Z \
  --series-id edgar_total_filings \
  --series-id edgar_8k_filings \
  --series-id edgar_10q_filings \
  --series-id edgar_10k_filings \
  --out cached_data/external_covariates \
  --partition-cols series_id
```

Benchmark-window EDGAR events job:

```bash
python scripts/get_events.py
```

This defaults to the benchmark window configured in [benchmark_window_config.py](/Users/sneddy/research/polymarket_research/configs/benchmark_window_config.py) and writes to `cached_data/external_events`.

Key behavior:

- uses the benchmark defaults from [benchmark_window_config.py](/Users/sneddy/research/polymarket_research/configs/benchmark_window_config.py) for `start-date`, `end-date`, and output path
- resumes from an existing `cached_data/external_events` dataset and downloads only missing daily tail data per `series_id`
- shows progress bars by default for both the outer series loop and the inner SEC daily download loop
- supports `--no-progress` to disable progress bars

### 5) Run the benchmark suite

```bash
conda activate polymarket
python benchmarks/run_benchmarks.py --domain geopolitics
```

Results are exported to:

```text
benchmarks/results/geopolitics/
```

## Benchmarks

The benchmark package lives in [benchmarks/README.md](/Users/sneddy/research/polymarket_research/benchmarks/README.md).

Current benchmark tasks:

- multi-horizon terminal forecasting
- trustworthiness / selective prediction
- large repricing prediction

Reproducible benchmark run:

```bash
conda activate polymarket
python benchmarks/run_benchmarks.py --domain geopolitics
```

## External Covariates

External market series are collected through the same `client -> collector -> script` pattern as the Polymarket data.

Current implementation:

- `Binance` provider for high-frequency crypto bars (`BTC/ETH`)
- `FRED` provider for daily macro / rates / FX series
- `SEC EDGAR` provider for daily filing-count series from the public form index
- `scripts/get_events.py` thin job wrapper for benchmark-window EDGAR event collection with resume support
- canonical registry in [configs/external_covariates_config.py](/Users/sneddy/research/polymarket_research/configs/external_covariates_config.py)

Recommended usage:

- `BTC/ETH`: `--binance-source archive` for full historical backfill
- `oil/rates/FX`: currently normalized from FRED daily series
- `SEC EDGAR`: configure `SecEdgarConfig.user_agent` in [config.py](/Users/sneddy/research/polymarket_research/config.py) with contact info before downloading `edgar_*` series
- `get_events.py`: use this for the benchmark-window EDGAR dataset rather than retyping `start-date` / `end-date`
- for the freshest crypto tail, the archive flow supplements monthly files with recent daily files

## Examples

`examples/` is notebook-only. It is meant for demo exploration, not for job-style execution.

Representative notebooks:

- `examples/btc_eth_covariates_demo.ipynb`
- `examples/download_trades.ipynb`
- `examples/record_orderbook.ipynb`
- `examples/resolved_market_probability_panel_demo.ipynb`
- `examples/lob_lookup.ipynb`
- `examples/structural_break.ipynb`

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
python -m scripts.download_trades \
  --market-id 0x... \
  --out cached_data/trades.parquet
```

By Polymarket URL:

```bash
python -m scripts.download_trades \
  --url "https://polymarket.com/event/fed-decision-in-march-885" \
  --market-index 0 \
  --out cached_data/trades.parquet
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
python -m scripts.inspect_market_meta \
  --top 50 \
  --out-markets cached_data/markets.parquet \
  --out-top cached_data/top_markets.parquet \
  --out-summary cached_data/market_summary.json
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
python -m scripts.inspect_market_meta --active-only --top 25
```

Recent markets only (creation date cutoff):

```bash
python -m scripts.inspect_market_meta \
  --min-created-at "2025-01-01T00:00:00Z" \
  --top 25
```

Finished/closed questions example:

```python
from clients.gamma_client import GammaClient
from collectors.markets_collector import MarketsCollector

mc = MarketsCollector(GammaClient())
finished = mc.download_market_meta(
    include_active=False,
    include_closed=True,
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
    include_closed=True,
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
- `download_markets(active=True/False, ...)`: download one slice; `active=False` maps to the closed slice.
- `download_market_universe(include_active=..., include_closed=..., ...)`: download and optionally dedupe both slices.
- `summarize_markets(df)`: aggregate stats dictionary.
- `rank_markets(df, top_n=...)`: ranking DataFrame without re-downloading.

### Ranking and `top_n`

- `top_n` limits only `top_markets` output size.
- It does not limit universe download size.
- To reduce download work, use `min_created_at`, `max_pages`, and/or Gamma filters (passed through as `**params`, e.g. `closed="true"`).

### Metadata columns

`markets` contains Gamma market payload keys (normalized to snake_case by default), typically including:
- identifiers: `id`, `condition_id`, `slug`, `question`
- taxonomy: `category`
- status: `active`, `closed`, `archived`
- timing: `created_at`, `end_date`
- liquidity/volume: `liquidity_clob`, `volume_clob`, `volume24hr_clob`, `volume1wk_clob`, `spread`

`top_markets` adds derived ranking fields:
- `category`
- `liquidity`, `volume_24h`, `volume_1w`, `volume_total`, `spread`
- `market_score`

### Progress behavior

- Progress bars are enabled by default for market metadata.
- Total estimation is enabled by default (`estimate_total=True`) using offset probes.
- With `min_created_at`, collector estimates a start offset first (binary seek on `createdAt`) to skip older pages quickly.

## 4) Record Live Order Book

### CLI

```bash
python -m scripts.record_orderbook \
  --url "https://polymarket.com/event/fed-decision-in-march-885" \
  --market-index 0 \
  --seconds 60 \
  --snapshot \
  --out cached_data/orderbook.parquet
```

Alternative source:

```bash
python -m scripts.record_orderbook --id <TOKEN_ID_OR_CONDITION_ID> --seconds 60
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
recorder.save_to_parquet(df, "cached_data/orderbook.parquet")
```

`OrderBookRecorder` also provides async methods (`aconnect`, `asubscribe`, `arecord`, etc.) for notebook event-loop workflows.

## 5) Poll Limited Order Books Into SQLite

For stable periodic snapshotting, prefer `OrderBookSnapshotCollector` over the websocket recorder.

What it does:
- resolves all open order-book outcomes from a `/event/...` or `/market/...` URL
- polls REST `/book` snapshots on a fixed interval
- sorts levels to true top-of-book before trimming
- stores snapshots, levels, and per-token errors in SQLite

### CLI

```bash
python -m scripts.poll_orderbooks \
  --url "https://polymarket.com/event/iran-x-israelus-conflict-ends-by" \
  --db cached_data/iran_conflict_orderbooks.sqlite \
  --interval-seconds 10 \
  --levels 5 \
  --polls 3
```

### Python

```python
from collectors.orderbook_snapshot_collector import OrderBookSnapshotCollector

collector = OrderBookSnapshotCollector()
results = collector.run(
    url="https://polymarket.com/event/iran-x-israelus-conflict-ends-by",
    db_path="cached_data/iran_conflict_orderbooks.sqlite",
    interval_seconds=10,
    levels=5,
    max_polls=3,
)
```

SQLite tables:
- `market_outcomes`: URL-resolved market/outcome/token mapping
- `poll_cycles`: one row per polling timestamp
- `orderbook_snapshots`: one row per token per poll cycle
- `orderbook_levels`: bid/ask levels for each snapshot
- `poll_errors`: failures captured without dropping the whole cycle

`levels` is validated in `[1, 10]`.

## 6) News Search

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

news.save_to_parquet(df, "cached_data/news.parquet")
```

## Storage

Any collector can persist outputs through `save_to_parquet(...)`.

Direct usage:

```python
from storage.parquet_store import ParquetStore

store = ParquetStore()
# loaded = store.load("cached_data/trades.parquet", frame_type="pandas")
```

For periodic order book snapshots:

```python
from storage.sqlite_orderbook_store import SqliteOrderBookStore

store = SqliteOrderBookStore()
counts = store.get_counts("cached_data/iran_conflict_orderbooks.sqlite")
```

## Operational Notes

- No hidden daemons/background loops: collection runs only when called.
- Most classes support endpoint/base URL overrides through config objects.
- If progress bars do not render in notebooks, verify `tqdm` is installed and notebook widget support is enabled.
