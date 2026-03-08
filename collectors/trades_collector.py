from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import json
import logging
from typing import Any, Literal

from clients.gamma_client import GammaClient
from clients.orderbook_subgraph_client import OrderbookSubgraphClient, TradeCursor
from storage.parquet_store import ParquetStore
from utils import ensure_datetime_utc, pick_condition_id_from_markets


logger = logging.getLogger(__name__)

FrameType = Literal["pandas", "polars"]


def _default_frame_type() -> FrameType:
    try:
        import pandas as _  # noqa: F401

        return "pandas"
    except Exception:
        return "polars"


def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _resolve_tqdm(show_progress: bool) -> Any | None:
    if not show_progress:
        return None

    if _running_in_notebook():
        try:
            from tqdm.notebook import tqdm as _tqdm

            return _tqdm
        except Exception:
            pass

    try:
        from tqdm.auto import tqdm as _tqdm

        return _tqdm
    except Exception:
        pass

    try:
        from tqdm import tqdm as _tqdm

        return _tqdm
    except Exception:
        return None


class TradesCollector:
    def __init__(
        self,
        client: GammaClient,
        *,
        subgraph_client: OrderbookSubgraphClient | None = None,
        store: ParquetStore | None = None,
    ) -> None:
        self._client = client
        self._subgraph = subgraph_client or OrderbookSubgraphClient()
        self._store = store or ParquetStore()

    def download_all_trades_from_url(
        self,
        url: str,
        start_date: datetime | str | None = None,
        *,
        limit: int = 500,
        max_pages: int | None = None,
        frame_type: FrameType | None = None,
        market_index: int | None = None,
        market_slug: str | None = None,
        show_progress: bool = True,
    ) -> Any:
        """
        Convenience wrapper: resolve condition id(s) from a Polymarket URL, then download trades.

        For /event/ URLs that contain multiple markets, pass `market_index` or `market_slug`.
        """

        markets = self._client.resolve_markets_from_polymarket_url(url)
        condition_id = pick_condition_id_from_markets(markets, market_index=market_index, market_slug=market_slug)
        return self.download_all_trades(
            condition_id,
            start_date=start_date,
            limit=limit,
            max_pages=max_pages,
            frame_type=frame_type,
            show_progress=show_progress,
        )

    def download_all_trades(
        self,
        market_id: str,
        start_date: datetime | str | None = None,
        *,
        limit: int = 500,
        max_pages: int | None = None,
        frame_type: FrameType | None = None,
        show_progress: bool = True,
        estimate_total: bool = True,
    ) -> Any:
        """
        Download full historical trades for a market (condition id).

        Uses Polymarket's orderbook subgraph with time-based backward pagination:
          - fetch latest trades (ordered by timestamp desc)
          - take oldest (timestamp, id) cursor from batch
          - request next batch using (timestamp_lt OR (timestamp == cursor AND id_lt))
          - repeat until empty

        Immediately normalizes to a minimal schema:
          timestamp_utc, price, size, outcome, transaction_hash

        Size semantics:
          - subgraph amounts are often token base units
          - collector auto-detects likely base-unit scaling and normalizes `size`
            to contract-size units (with warnings when magnitudes look implausible)
        """

        start_dt = ensure_datetime_utc(start_date) if start_date is not None else None
        start_ts = int(start_dt.timestamp()) if start_dt is not None else None

        token_ids, token_outcomes = self._resolve_token_ids_and_outcomes(market_id)

        rows: list[dict[str, Any]] = []
        batches = 0
        fills_downloaded = 0
        assets_done = 0
        assets_total = len(token_ids)

        tqdm = _resolve_tqdm(show_progress=show_progress)
        if show_progress and tqdm is None:
            logger.warning("show_progress=True but tqdm is unavailable in the active environment.")

        total_rows = None
        if estimate_total:
            est = self.estimate_trade_count(market_id, start_date=start_dt)
            if est is not None and est > 0:
                total_rows = int(est)

        pbar = (
            tqdm(
                total=total_rows,
                disable=False,
                unit="trade",
                desc="Downloading trades",
                leave=True,
            )
            if tqdm is not None
            else None
        )

        try:
            def _collect_for_asset_ids(asset_ids: list[str]) -> None:
                nonlocal assets_done, batches, fills_downloaded
                for asset_id in asset_ids:
                    assets_done += 1
                    cursor: TradeCursor | None = None
                    pages_for_asset = 0

                    while True:
                        if max_pages is not None and pages_for_asset >= max_pages:
                            break

                        page = self._subgraph.get_order_filled_events_page_for_asset(
                            asset_id,
                            first=limit,
                            cursor=cursor,
                            start_ts=start_ts,
                        )
                        if not page:
                            break

                        page_fills = len(page)
                        for ev in page:
                            if not isinstance(ev, Mapping):
                                continue
                            row, _ = self._normalize_filled_event_for_asset(
                                ev,
                                asset_id=asset_id,
                                token_outcomes=token_outcomes,
                            )
                            if row is None:
                                continue
                            rows.append(row)

                        pages_for_asset += 1
                        batches += 1
                        fills_downloaded += page_fills
                        if pbar is not None:
                            pbar.update(page_fills)

                        last_ev = page[-1] if isinstance(page[-1], Mapping) else None
                        if not isinstance(last_ev, Mapping):
                            logger.warning("Could not read cursor fields from last fill; stopping.")
                            break

                        oldest_ts = TradesCollector._extract_timestamp_seconds(last_ev)
                        last_id = last_ev.get("id")
                        if oldest_ts is None or not isinstance(last_id, str):
                            logger.warning("Could not parse cursor from fills batch; stopping.")
                            break

                        oldest_dt = datetime.fromtimestamp(oldest_ts, tz=UTC)
                        if pbar is not None:
                            pbar.set_postfix(
                                {
                                    "asset": f"{assets_done}/{max(assets_total, assets_done)}",
                                    "pages": batches,
                                    "fills": fills_downloaded,
                                    "kept": len(rows),
                                    "earliest": oldest_dt.isoformat(),
                                }
                            )

                        next_cursor = TradeCursor(timestamp=oldest_ts, trade_id=last_id)
                        if cursor is not None and next_cursor == cursor:
                            logger.warning(
                                "Cursor did not advance (timestamp=%s id=%s); stopping to avoid an infinite loop.",
                                oldest_ts,
                                last_id,
                            )
                            break
                        cursor = next_cursor

            _collect_for_asset_ids(token_ids)

            # Fallback: if nothing returned, derive candidate asset ids from subgraph MarketData.
            if not rows and isinstance(market_id, str) and market_id.startswith("0x"):
                try:
                    md = self._subgraph.get_market_datas_by_condition_id(market_id, first=5)
                    candidates: list[str] = []
                    for m in md:
                        candidates.extend(self._subgraph.extract_asset_ids(m))
                    candidates = [c for c in candidates if c and c != "0"]
                    # Avoid retrying identical ids.
                    candidates = [c for c in candidates if c not in set(token_ids)]
                    if candidates:
                        assets_total += len(candidates)
                        _collect_for_asset_ids(candidates)
                except Exception:
                    pass
        finally:
            if pbar is not None:
                pbar.close()

        df = self._to_frame(rows, frame_type=frame_type)
        df = self._finalize(df, start_dt=start_dt, frame_type=frame_type)
        return df

    def estimate_trade_count(self, market_id: str, *, start_date: datetime | str | None = None) -> int | None:
        """
        Best-effort estimate of total trades (fill events) for a market.

        Returns None if the active subgraph deployment cannot provide a usable count.
        """

        start_dt = ensure_datetime_utc(start_date) if start_date is not None else None
        start_ts = int(start_dt.timestamp()) if start_dt is not None else None

        token_ids, _ = self._resolve_token_ids_and_outcomes(market_id)
        totals: list[int] = []
        for asset_id in token_ids:
            n = self._subgraph.try_count_order_filled_events_for_asset(asset_id, start_ts=start_ts)
            # Fallback: some deployments expose all-time per-orderbook totals but not connection counts.
            if n is None and start_ts is None:
                n = self._subgraph.try_count_orderbook_trades_for_asset(asset_id)
            if n is None:
                return None
            totals.append(int(n))
        return int(sum(totals))

    def save_to_parquet(self, df: Any, path: str, *, partition_cols: list[str] | None = None) -> str:
        out = self._store.save(df, path, partition_cols=partition_cols)
        return str(out)

    @staticmethod
    def _extract_items(payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "trades", "results", "items"):
                val = payload.get(key)
                if isinstance(val, list):
                    return val
        return []

    @staticmethod
    def _normalize_trade(
        trade: Mapping[str, Any],
        *,
        token_outcomes: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any] | None, int | None]:
        ts = TradesCollector._extract_timestamp_seconds(trade)
        if ts is None:
            return None, None

        dt = datetime.fromtimestamp(ts, tz=UTC)

        price = TradesCollector._to_float(trade.get("price") or trade.get("trade_price") or trade.get("rate"))
        size = TradesCollector._to_float(trade.get("size") or trade.get("amount") or trade.get("shares"))

        outcome = trade.get("outcome") or trade.get("outcome_id") or trade.get("outcomeId")
        if outcome is None:
            token = trade.get("token")
            if isinstance(token, Mapping):
                outcome = token.get("outcome")

        if outcome is None and token_outcomes is not None:
            token = trade.get("token")
            if isinstance(token, Mapping):
                tid = token.get("id")
                if isinstance(tid, str):
                    outcome = token_outcomes.get(tid)

        if outcome is not None:
            outcome = str(outcome)

        txh = (
            trade.get("transaction_hash")
            or trade.get("transactionHash")
            or trade.get("tx_hash")
            or trade.get("txHash")
            or trade.get("hash")
        )
        if txh is not None:
            txh = str(txh)

        return (
            {
                "timestamp_utc": dt,
                "price": price,
                "size": size,
                "outcome": outcome,
                "transaction_hash": txh,
            },
            ts,
        )

    @staticmethod
    def _normalize_filled_event(
        ev: Mapping[str, Any],
        *,
        token_outcomes: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any] | None, int | None]:
        """
        Normalize an `orderFilledEvent` to the minimal trade schema.

        We infer (price, size, outcome) from maker/taker asset ids + filled amounts.
        Note: maker/taker filled amounts are ingested as raw values; size normalization
        (base units -> contract units) is applied centrally in `_finalize`.
        """

        ts = TradesCollector._extract_timestamp_seconds(ev)
        if ts is None:
            return None, None

        dt = datetime.fromtimestamp(ts, tz=UTC)

        txh = ev.get("transactionHash") or ev.get("transaction_hash") or ev.get("hash")
        txh = None if txh is None else str(txh)

        maker_asset = ev.get("makerAssetId") or ev.get("makerAssetID")
        taker_asset = ev.get("takerAssetId") or ev.get("takerAssetID")
        maker_amt = TradesCollector._to_float(ev.get("makerAmountFilled") or ev.get("maker_amount_filled"))
        taker_amt = TradesCollector._to_float(ev.get("takerAmountFilled") or ev.get("taker_amount_filled"))

        if maker_asset is not None:
            maker_asset = str(maker_asset)
        if taker_asset is not None:
            taker_asset = str(taker_asset)

        # Determine which side is the outcome token.
        outcome_token_id: str | None = None
        shares: float | None = None
        quote: float | None = None

        if maker_asset is not None and token_outcomes is not None and maker_asset in token_outcomes:
            outcome_token_id = maker_asset
            shares = maker_amt
            quote = taker_amt
        elif taker_asset is not None and token_outcomes is not None and taker_asset in token_outcomes:
            outcome_token_id = taker_asset
            shares = taker_amt
            quote = maker_amt
        else:
            # Fallback heuristic if token_outcomes isn't available.
            # Prefer the side that produces a probability-like price in [0, 1.05].
            def _safe_price(q: float | None, s: float | None) -> float | None:
                if q is None or s is None or s == 0:
                    return None
                return q / s

            p1 = _safe_price(taker_amt, maker_amt)
            p2 = _safe_price(maker_amt, taker_amt)
            if p1 is not None and 0 <= p1 <= 1.05:
                shares, quote = maker_amt, taker_amt
            elif p2 is not None and 0 <= p2 <= 1.05:
                shares, quote = taker_amt, maker_amt
            else:
                shares, quote = maker_amt, taker_amt

        price = None
        if quote is not None and shares is not None and shares != 0:
            price = quote / shares

        outcome = None
        if outcome_token_id is not None and token_outcomes is not None:
            outcome = token_outcomes.get(outcome_token_id)

        return (
            {
                "timestamp_utc": dt,
                "price": TradesCollector._to_float(price),
                "size": TradesCollector._to_float(shares),
                "outcome": outcome,
                "transaction_hash": txh,
            },
            ts,
        )

    @staticmethod
    def _normalize_filled_event_for_asset(
        ev: Mapping[str, Any],
        *,
        asset_id: str,
        token_outcomes: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any] | None, int | None]:
        """
        Normalize an orderFilledEvent and attribute it to a specific asset id.

        This avoids ambiguous outcome inference: `asset_id` is the outcome token id
        we requested from the subgraph, and we compute (price, size) accordingly.
        Note: filled amounts are kept raw here; size normalization is applied in `_finalize`.
        """

        ts = TradesCollector._extract_timestamp_seconds(ev)
        if ts is None:
            return None, None

        dt = datetime.fromtimestamp(ts, tz=UTC)

        txh = ev.get("transactionHash") or ev.get("transaction_hash") or ev.get("hash")
        txh = None if txh is None else str(txh)

        maker_asset = ev.get("makerAssetId") or ev.get("makerAssetID")
        taker_asset = ev.get("takerAssetId") or ev.get("takerAssetID")

        maker_amt = TradesCollector._to_float(
            ev.get("makerAmountFilled")
            or ev.get("maker_amount_filled")
            or ev.get("makerAmount")
            or ev.get("maker_amount")
        )
        taker_amt = TradesCollector._to_float(
            ev.get("takerAmountFilled")
            or ev.get("taker_amount_filled")
            or ev.get("takerAmount")
            or ev.get("taker_amount")
        )

        maker_asset = None if maker_asset is None else str(maker_asset)
        taker_asset = None if taker_asset is None else str(taker_asset)
        asset_id = str(asset_id)

        shares: float | None = None
        quote: float | None = None
        if maker_asset == asset_id:
            shares, quote = maker_amt, taker_amt
        elif taker_asset == asset_id:
            shares, quote = taker_amt, maker_amt
        else:
            return None, ts

        price = None
        if shares is not None and shares != 0 and quote is not None:
            price = quote / shares

        outcome = None
        if token_outcomes is not None:
            outcome = token_outcomes.get(asset_id)

        return (
            {
                "timestamp_utc": dt,
                "price": TradesCollector._to_float(price),
                "size": TradesCollector._to_float(shares),
                "outcome": outcome,
                "transaction_hash": txh,
            },
            ts,
        )

    @staticmethod
    def _extract_timestamp_seconds(trade: Mapping[str, Any]) -> int | None:
        raw = trade.get("timestamp") or trade.get("ts") or trade.get("time")
        if raw is None:
            raw = trade.get("created_at") or trade.get("createdAt")

        if raw is None:
            return None

        # Numeric epoch.
        if isinstance(raw, (int, float)):
            return TradesCollector._coerce_epoch_seconds(raw)
        if isinstance(raw, str):
            s = raw.strip()
            if s.isdigit():
                return TradesCollector._coerce_epoch_seconds(int(s))
            # ISO-like string.
            try:
                dt = ensure_datetime_utc(s)
            except Exception:
                return None
            return int(dt.timestamp())

        return None

    @staticmethod
    def _coerce_epoch_seconds(value: int | float) -> int:
        ts = int(value)
        # Heuristics for ms/us/ns epochs.
        if ts > 10**17:  # ns
            return int(ts / 1_000_000_000)
        if ts > 10**14:  # us
            return int(ts / 1_000_000)
        if ts > 10**11:  # ms
            return int(ts / 1_000)
        return ts

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _coerce_list(value: Any) -> list[Any] | None:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return []
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = json.loads(s)
                except Exception:
                    parsed = None
                if isinstance(parsed, list):
                    return parsed
            if "," in s:
                parts = [p.strip().strip("\"'") for p in s.split(",")]
                parts = [p for p in parts if p]
                if parts:
                    return parts
            return [s]
        return None

    def _resolve_token_ids_and_outcomes(self, market_id: str) -> tuple[list[str], dict[str, str]]:
        """
        Resolve outcome token ids for a market.

        Primary mode: treat `market_id` as a condition id (0x...) and resolve via Gamma metadata.
        """

        # If market_id looks like a token id (e.g., numeric string), just return it.
        if isinstance(market_id, str) and not market_id.startswith("0x"):
            return [market_id], {}

        market = self._find_market_by_condition_id(market_id)
        token_ids = (
            market.get("clobTokenIds")
            or market.get("clob_token_ids")
            or market.get("clobTokenIDs")
            or market.get("clob_token_IDs")
        )
        token_ids = TradesCollector._coerce_list(token_ids)
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError("Gamma market payload did not include clobTokenIds/clob_token_ids.")

        clean_token_ids: list[str] = []
        seen_token_ids: set[str] = set()
        for t in token_ids:
            tid = str(t).strip()
            if not tid or tid in seen_token_ids:
                continue
            seen_token_ids.add(tid)
            clean_token_ids.append(tid)

        if not clean_token_ids:
            raise ValueError("Gamma market payload included clobTokenIds but no usable token ids were found.")

        # Map token id -> outcome name when possible (often aligns with clobTokenIds order).
        outcomes = market.get("outcomes") or market.get("outcomeNames") or market.get("outcome_names")
        outcomes = TradesCollector._coerce_list(outcomes)
        outcome_names: list[str] = []
        if isinstance(outcomes, list):
            for o in outcomes:
                if isinstance(o, str):
                    name = o.strip()
                    if name:
                        outcome_names.append(name)
                elif isinstance(o, Mapping):
                    name = o.get("name") or o.get("title") or o.get("outcome")
                    if name is not None:
                        outcome_names.append(str(name))

        token_outcomes: dict[str, str] = {}
        if outcome_names and len(outcome_names) == len(clean_token_ids):
            token_outcomes = {tid: outcome_names[i] for i, tid in enumerate(clean_token_ids)}

        return clean_token_ids, token_outcomes

    def _find_market_by_condition_id(self, condition_id: str) -> Mapping[str, Any]:
        def _extract_markets(payload: Any) -> list[Mapping[str, Any]]:
            if isinstance(payload, list):
                return [m for m in payload if isinstance(m, Mapping)]
            if isinstance(payload, Mapping):
                maybe = payload.get("markets") or payload.get("data") or payload.get("results") or payload.get("items")
                if isinstance(maybe, list):
                    return [m for m in maybe if isinstance(m, Mapping)]
            return []

        def _get_cid(m: Mapping[str, Any]) -> str | None:
            cid = m.get("conditionId") or m.get("condition_id")
            return None if cid is None else str(cid)

        # Try common filter parameter names and validate exact match.
        filter_variants: list[dict[str, Any]] = [
            {"condition_id": condition_id},
            {"conditionId": condition_id},
            {"condition_ids": [condition_id]},
            {"conditionIds": [condition_id]},
        ]

        for active in (True, False):
            for params in filter_variants:
                try:
                    payload = self._client.get_markets(active=active, limit=25, offset=0, **params)
                except Exception:
                    continue
                for m in _extract_markets(payload):
                    if _get_cid(m) == condition_id:
                        return m

        # Fallback: scan pages (bounded) until we find an exact match.
        scan_limit = 200
        max_pages = 200

        for active in (True, False):
            offset = 0
            for _ in range(max_pages):
                payload = self._client.get_markets(active=active, limit=scan_limit, offset=offset)
                batch = _extract_markets(payload)
                if not batch:
                    break
                for m in batch:
                    if _get_cid(m) == condition_id:
                        return m
                offset += len(batch)

        raise ValueError(f"Could not find Gamma market for condition id: {condition_id}")

    @staticmethod
    def _to_frame(rows: list[dict[str, Any]], *, frame_type: FrameType | None) -> Any:
        frame = frame_type or _default_frame_type()
        if frame == "pandas":
            import pandas as pd

            if not rows:
                return pd.DataFrame(columns=["timestamp_utc", "price", "size", "outcome", "transaction_hash"])
            return pd.DataFrame(rows)
        if frame == "polars":
            import polars as pl

            if not rows:
                return pl.DataFrame(
                    schema=[
                        ("timestamp_utc", pl.Datetime(time_zone="UTC")),
                        ("price", pl.Float64),
                        ("size", pl.Float64),
                        ("outcome", pl.Utf8),
                        ("transaction_hash", pl.Utf8),
                    ]
                )
            return pl.DataFrame(rows)
        raise ValueError(f"Unknown frame_type: {frame!r}")

    @staticmethod
    def _finalize(df: Any, *, start_dt: datetime | None, frame_type: FrameType | None) -> Any:
        frame = frame_type or _default_frame_type()
        if frame == "pandas":
            import pandas as pd

            # Ensure types.
            if "timestamp_utc" in df.columns:
                df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
            for col in ("price", "size"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Filter
            if "timestamp_utc" not in df.columns:
                return df
            if start_dt is not None:
                df = df[df["timestamp_utc"] >= start_dt]

            # Subgraph fill amounts are often raw token base units.
            # Detect and normalize to contract-size units for research use.
            df = TradesCollector._normalize_and_validate_size_pandas(df)

            # Dedup + order
            subset = ["timestamp_utc", "price", "size", "outcome", "transaction_hash"]
            subset = [c for c in subset if c in df.columns]
            df = df.drop_duplicates(subset=subset, keep="first")
            df = df.sort_values("timestamp_utc", ascending=True, kind="stable").reset_index(drop=True)

            # Minimal schema + stable column order
            cols = ["timestamp_utc", "price", "size", "outcome", "transaction_hash"]
            return df[[c for c in cols if c in df.columns]]

        if frame == "polars":
            import polars as pl

            if "timestamp_utc" in df.columns:
                df = df.with_columns(
                    pl.col("timestamp_utc")
                    .cast(pl.Datetime, strict=False)
                    .dt.replace_time_zone("UTC")
                    .alias("timestamp_utc")
                )
            for col in ("price", "size"):
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False).alias(col))

            if start_dt is not None and "timestamp_utc" in df.columns:
                df = df.filter(pl.col("timestamp_utc") >= start_dt)

            df = TradesCollector._normalize_and_validate_size_polars(df)

            subset = ["timestamp_utc", "price", "size", "outcome", "transaction_hash"]
            subset = [c for c in subset if c in df.columns]
            if subset:
                df = df.unique(subset=subset, keep="first")
            if "timestamp_utc" in df.columns:
                df = df.sort("timestamp_utc")

            cols = ["timestamp_utc", "price", "size", "outcome", "transaction_hash"]
            return df.select([c for c in cols if c in df.columns])

        raise ValueError(f"Unknown frame_type: {frame!r}")

    @staticmethod
    def _normalize_and_validate_size_pandas(df: Any) -> Any:
        import pandas as pd

        if "size" not in df.columns:
            return df

        size_series = pd.to_numeric(df["size"], errors="coerce")
        price_series = (
            pd.to_numeric(df["price"], errors="coerce") if "price" in df.columns else pd.Series(dtype="float64")
        )

        scale, reason = TradesCollector._infer_size_scale(
            size_values=size_series.tolist(),
            price_values=price_series.tolist() if len(price_series) else None,
        )
        if scale != 1.0:
            logger.warning("Normalizing trade size by /%s (%s).", scale, reason)
            df = df.copy()
            df["size"] = size_series / float(scale)
        else:
            df = df.copy()
            df["size"] = size_series

        TradesCollector._warn_if_implausible_size(
            df["size"].tolist(),
            scale_used=scale,
            context="pandas_finalize",
        )
        return df

    @staticmethod
    def _normalize_and_validate_size_polars(df: Any) -> Any:
        try:
            import polars as pl
        except Exception:
            return df

        if "size" not in df.columns:
            return df

        size_values = [float(v) for v in df.get_column("size").drop_nulls().to_list()]
        price_values: list[float] | None = None
        if "price" in df.columns:
            price_values = [float(v) for v in df.get_column("price").drop_nulls().to_list()]

        scale, reason = TradesCollector._infer_size_scale(size_values=size_values, price_values=price_values)
        if scale != 1.0:
            logger.warning("Normalizing trade size by /%s (%s).", scale, reason)
            df = df.with_columns((pl.col("size") / float(scale)).alias("size"))

        norm_values = [float(v) for v in df.get_column("size").drop_nulls().to_list()]
        TradesCollector._warn_if_implausible_size(
            norm_values,
            scale_used=scale,
            context="polars_finalize",
        )
        return df

    @staticmethod
    def _infer_size_scale(
        *,
        size_values: list[float],
        price_values: list[float] | None,
    ) -> tuple[float, str]:
        import math

        vals = [float(v) for v in size_values if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0]
        if not vals:
            return 1.0, "no positive size values"
        vals.sort()

        q50 = TradesCollector._quantile_sorted(vals, 0.50)
        q99 = TradesCollector._quantile_sorted(vals, 0.99)
        int_like_frac = sum(1 for v in vals if abs(v - round(v)) <= 1e-9) / float(len(vals))

        price_prob_like = False
        if price_values:
            p = [float(v) for v in price_values if isinstance(v, (int, float)) and math.isfinite(float(v))]
            if p:
                in_prob = sum(1 for x in p if 0.0 <= x <= 1.05)
                price_prob_like = (in_prob / float(len(p))) >= 0.90

        # Strong signal for 1e18-like ERC20 base units.
        if q50 >= 1e15 or q99 >= 1e17:
            return 1e18, "size magnitudes are consistent with 1e18 base units"

        # Common Polymarket CLOB case: integer-like raw units with probability-like prices.
        if q50 >= 1e5 and q99 >= 1e8 and int_like_frac >= 0.90 and price_prob_like:
            return 1e6, "size appears to be raw base units; applying 1e6 normalization"

        return 1.0, "size appears already normalized"

    @staticmethod
    def _warn_if_implausible_size(size_values: list[float], *, scale_used: float, context: str) -> None:
        import math

        vals = [float(v) for v in size_values if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0]
        if not vals:
            return
        vals.sort()
        q50 = TradesCollector._quantile_sorted(vals, 0.50)
        q99 = TradesCollector._quantile_sorted(vals, 0.99)
        vmin = vals[0]

        if scale_used == 1.0 and q99 >= 1e8:
            logger.warning(
                "Trade size looks implausibly large without scaling (context=%s, q50=%.6g, q99=%.6g). "
                "Check whether sizes are raw base units.",
                context,
                q50,
                q99,
            )
        if q99 > 1e7 or q50 > 1e5 or vmin < 1e-9:
            logger.warning(
                "Trade size magnitude may be implausible after normalization (context=%s, scale=%s, min=%.6g, q50=%.6g, q99=%.6g).",
                context,
                scale_used,
                vmin,
                q50,
                q99,
            )

    @staticmethod
    def _quantile_sorted(values: list[float], q: float) -> float:
        import math

        if not values:
            return float("nan")
        if len(values) == 1:
            return float(values[0])
        qq = min(max(float(q), 0.0), 1.0)
        pos = (len(values) - 1) * qq
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(values[lo])
        frac = pos - lo
        return float(values[lo] * (1.0 - frac) + values[hi] * frac)
