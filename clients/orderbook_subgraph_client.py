from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import random
import time
from typing import Any, Iterable

from config import HttpConfig, OrderbookSubgraphConfig

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class TradeCursor:
    timestamp: int
    trade_id: str  # Bytes (0x...)


@dataclass
class FillQuerySpec:
    entity: str
    maker_asset_field: str
    taker_asset_field: str
    maker_amount_field: str
    taker_amount_field: str
    timestamp_field: str
    tx_hash_field: str
    asset_id_gql_type: str
    timestamp_gql_type: str
    id_gql_type: str
    connection_entity: str | None = None


class OrderbookSubgraphClient:
    """
    Minimal GraphQL client for Polymarket's orderbook subgraph.

    Intended usage:
      - page fills backward by (timestamp, id) cursor
    """

    def __init__(
        self,
        *,
        subgraph: OrderbookSubgraphConfig | None = None,
        http: HttpConfig | None = None,
        session: "requests.Session | None" = None,
    ) -> None:
        if requests is None:
            raise ImportError("Missing dependency: requests. Install with `pip install requests`.")

        self._subgraph = subgraph or OrderbookSubgraphConfig()
        self._http = http or HttpConfig()
        self._session = session or requests.Session()
        self._session.headers.setdefault("Accept", "application/json")
        self._session.headers.setdefault("Content-Type", "application/json")
        self._session.headers.setdefault("User-Agent", self._http.user_agent)

    @property
    def endpoints(self) -> tuple[str, ...]:
        return tuple(self._subgraph.endpoints)

    def introspect_query_fields(self) -> list[str]:
        query = """
        query Introspect {
          __schema {
            queryType { fields { name } }
          }
        }
        """
        data = self._query_first_ok(query, {})
        schema = data.get("__schema") if isinstance(data, dict) else None
        qt = schema.get("queryType") if isinstance(schema, dict) else None
        fields = qt.get("fields") if isinstance(qt, dict) else None
        if not isinstance(fields, list):
            return []
        out: list[str] = []
        for f in fields:
            if isinstance(f, dict) and isinstance(f.get("name"), str):
                out.append(f["name"])
        return out

    def introspect_query_field_types(self) -> dict[str, str]:
        query = """
        query Introspect {
          __schema {
            queryType {
              fields {
                name
                type { kind name ofType { kind name ofType { kind name ofType { kind name ofType { kind name } } } } }
              }
            }
          }
        }
        """
        data = self._query_first_ok(query, {})
        schema = data.get("__schema") if isinstance(data, dict) else None
        qt = schema.get("queryType") if isinstance(schema, dict) else None
        fields = qt.get("fields") if isinstance(qt, dict) else None
        if not isinstance(fields, list):
            return {}
        out: dict[str, str] = {}
        for f in fields:
            if not isinstance(f, dict):
                continue
            name = f.get("name")
            if not isinstance(name, str):
                continue
            t = f.get("type")
            out[name] = _render_gql_type(t)
        return out

    def introspect_type_fields(self, type_name: str) -> dict[str, str]:
        query = """
        query TypeFields($name: String!) {
          __type(name: $name) {
            fields { name type { kind name ofType { kind name ofType { kind name ofType { kind name ofType { kind name } } } } } }
          }
        }
        """
        data = self._query_first_ok(query, {"name": type_name})
        t = data.get("__type") if isinstance(data, dict) else None
        fields = t.get("fields") if isinstance(t, dict) else None
        if not isinstance(fields, list):
            return {}
        out: dict[str, str] = {}
        for f in fields:
            if not isinstance(f, dict):
                continue
            name = f.get("name")
            if not isinstance(name, str):
                continue
            out[name] = _render_gql_type(f.get("type"))
        return out

    def get_market_datas_by_condition_id(self, condition_id: str, *, first: int = 5) -> list[dict[str, Any]]:
        """
        Best-effort helper to fetch MarketData rows for a given condition id.

        Useful when Gamma-provided token ids don't match the subgraph's asset ids.
        """

        fields = self.introspect_type_fields("MarketData")
        if not fields:
            return []

        condition_field = None
        for cand in ("conditionId", "conditionID", "condition_id"):
            if cand in fields:
                condition_field = cand
                break
        if condition_field is None:
            return []

        condition_type = _extract_base_gql_named_type(fields.get(condition_field, "")) or "Bytes"

        # Select a conservative set of scalar-ish fields that might contain asset ids.
        select_fields: list[str] = ["id", condition_field]
        for name, t in fields.items():
            if name in select_fields:
                continue
            base = _extract_base_gql_named_type(t) or ""
            is_scalarish = base in {"String", "Bytes", "BigInt", "Int", "ID", "Boolean"}
            is_list = t.strip().startswith("[")
            if not (is_scalarish or is_list):
                continue
            lname = name.lower()
            if any(k in lname for k in ("asset", "token", "clob", "outcome", "condition")):
                select_fields.append(name)
            if len(select_fields) >= 35:
                break

        selection = "\n            ".join(select_fields)
        query = f"""
        query MarketDatas($cid: {condition_type}!, $first: Int!) {{
          marketDatas(first: $first, where: {{ {condition_field}: $cid }}) {{
            {selection}
          }}
        }}
        """

        data = self._query_first_ok(query, {"cid": condition_id, "first": int(first)})
        out = data.get("marketDatas", []) if isinstance(data, dict) else []
        return [m for m in out if isinstance(m, dict)]

    @staticmethod
    def extract_asset_ids(obj: Any) -> list[str]:
        """
        Extract plausible asset id strings from a MarketData-like object.
        """

        out: list[str] = []

        def _add(v: Any) -> None:
            if v is None:
                return
            if isinstance(v, str):
                s = v.strip()
                if s:
                    out.append(s)
            elif isinstance(v, (int, float)):
                out.append(str(int(v)))

        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if not any(x in lk for x in ("asset", "token", "clob")):
                    continue
                if isinstance(v, list):
                    for x in v:
                        _add(x)
                else:
                    _add(v)

        # Unique, stable
        seen: set[str] = set()
        uniq: list[str] = []
        for s in out:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq

    def get_order_filled_events_page_for_asset(
        self,
        asset_id: str,
        *,
        first: int = 1000,
        cursor: TradeCursor | None = None,
        start_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Page `orderFilledEvents` for a single asset id.

        This matches the most common public-query examples for the orderbook subgraph and avoids
        reliance on `trades` / `tokens` fields (which may not exist).
        """

        if not asset_id:
            return []

        # Try multiple event entities (filled vs matched) depending on deployment.
        for spec in self._discover_event_query_specs():
            try:
                page = self._order_filled_events_page_for_asset(
                    asset_id=str(asset_id),
                    first=int(first),
                    cursor=cursor,
                    start_ts=start_ts,
                    entity=spec.entity,
                    maker_asset=spec.maker_asset_field,
                    taker_asset=spec.taker_asset_field,
                    maker_amt=spec.maker_amount_field,
                    taker_amt=spec.taker_amount_field,
                    timestamp_field=spec.timestamp_field,
                    tx_hash_field=spec.tx_hash_field,
                    asset_id_gql_type=spec.asset_id_gql_type,
                    timestamp_gql_type=spec.timestamp_gql_type,
                    id_gql_type=spec.id_gql_type,
                )
            except Exception:
                continue
            if page:
                return page
        return []

    def peek_latest_order_filled_events(self, *, first: int = 5) -> list[dict[str, Any]]:
        # Peek first non-empty among candidate entities.
        for spec in self._discover_event_query_specs():
            var_defs = "$first: Int!"
            query = f"""
            query Peek({var_defs}) {{
              {spec.entity}(first: $first, orderBy: {spec.timestamp_field}, orderDirection: desc) {{
                id
                timestamp: {spec.timestamp_field}
                transactionHash: {spec.tx_hash_field}
                makerAssetId: {spec.maker_asset_field}
                takerAssetId: {spec.taker_asset_field}
                makerAmountFilled: {spec.maker_amount_field}
                takerAmountFilled: {spec.taker_amount_field}
              }}
            }}
            """
            try:
                data = self._query_first_ok(query, {"first": int(first)})
            except Exception:
                continue
            out = data.get(spec.entity, []) if isinstance(data, dict) else []
            items = [e for e in out if isinstance(e, dict)]
            if items:
                return items
        return []

    def try_count_order_filled_events_for_asset(self, asset_id: str, *, start_ts: int | None = None) -> int | None:
        """
        Best-effort count estimate for progress bars.

        Many Graph subgraphs do not expose counts. If the deployment exposes
        `orderFilledEventsConnection.totalCount`, this returns it; otherwise None.
        """

        specs = self._discover_event_query_specs()
        spec = next((s for s in specs if s.connection_entity is not None), None)
        if spec is None or spec.connection_entity is None:
            return None

        where_parts = [
            "{ or: ["
            f"{{ {spec.maker_asset_field}: $assetID }},"
            f"{{ {spec.taker_asset_field}: $assetID }}"
            "] }"
        ]
        if start_ts is not None:
            where_parts.append("{ " + spec.timestamp_field + "_gte: $startTs }")

        where_block = "and: [" + ", ".join(where_parts) + "]"
        var_defs = [f"$assetID: {spec.asset_id_gql_type}!", "$first: Int!"]
        variables: dict[str, Any] = {"assetID": str(asset_id), "first": 1}
        if start_ts is not None:
            var_defs.append(f"$startTs: {spec.timestamp_gql_type}!")
            variables["startTs"] = str(int(start_ts))
        var_defs_s = ", ".join(var_defs)

        query = f"""
        query Count({var_defs_s}) {{
          {spec.connection_entity}(where: {{ {where_block} }}) {{
            totalCount
          }}
        }}
        """

        try:
            data = self._query_first_ok(query, variables)
        except Exception:
            return None
        conn = data.get(spec.connection_entity) if isinstance(data, dict) else None
        if isinstance(conn, dict):
            return _to_int(conn.get("totalCount"))
        return None

    def try_count_orderbook_trades_for_asset(self, asset_id: str) -> int | None:
        """
        Best-effort count via `orderbook(id: ...) { tradesQuantity }`.

        Useful fallback when `orderFilledEventsConnection.totalCount` is unavailable.
        This count is all-time for the asset (cannot apply `start_ts` filtering).
        """

        if not asset_id:
            return None

        query_id = """
        query CountOrderbook($id: ID!) {
          orderbook(id: $id) {
            tradesQuantity
          }
        }
        """
        try:
            data = self._query_first_ok(query_id, {"id": str(asset_id)})
        except Exception:
            data = None

        ob = data.get("orderbook") if isinstance(data, dict) else None
        if isinstance(ob, dict):
            n = _to_int(ob.get("tradesQuantity"))
            if n is not None:
                return n

        # Fallback for deployments that only expose the plural query.
        query_plural = """
        query CountOrderbookPlural($id: ID!, $first: Int!) {
          orderbooks(first: $first, where: { id: $id }) {
            tradesQuantity
          }
        }
        """
        try:
            data = self._query_first_ok(query_plural, {"id": str(asset_id), "first": 1})
        except Exception:
            return None

        rows = data.get("orderbooks") if isinstance(data, dict) else None
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            return _to_int(rows[0].get("tradesQuantity"))
        return None

    def iter_trades(
        self,
        token_ids: list[str],
        *,
        first: int = 1000,
        start_ts: int | None = None,
        max_pages: int | None = None,
    ) -> Iterable[dict[str, Any]]:
        # Kept for backwards-compat (iterates per-asset sequentially).
        for asset_id in token_ids:
            cursor: TradeCursor | None = None
            pages = 0
            while True:
                if max_pages is not None and pages >= max_pages:
                    break

                page = self.get_order_filled_events_page_for_asset(asset_id, first=first, cursor=cursor, start_ts=start_ts)
                if not page:
                    break

                for t in page:
                    yield t

                pages += 1
                last = page[-1]
                ts = _to_int(last.get("timestamp"))
                tid = last.get("id")
                if ts is None or not isinstance(tid, str):
                    logger.warning("Could not read cursor fields from last fill; stopping.")
                    break

                next_cursor = TradeCursor(timestamp=ts, trade_id=tid)
                if cursor is not None and next_cursor == cursor:
                    logger.warning("Cursor did not advance (timestamp=%s id=%s); stopping.", ts, tid)
                    break
                cursor = next_cursor

    def _order_filled_events_page_for_asset(
        self,
        *,
        asset_id: str,
        first: int,
        cursor: TradeCursor | None,
        start_ts: int | None,
        entity: str,
        maker_asset: str,
        taker_asset: str,
        maker_amt: str,
        taker_amt: str,
        timestamp_field: str,
        tx_hash_field: str,
        asset_id_gql_type: str,
        timestamp_gql_type: str,
        id_gql_type: str,
    ) -> list[dict[str, Any]]:
        # Try a single combined OR query first.
        try:
            return self._order_filled_events_page_for_asset_or(
                asset_id=asset_id,
                first=first,
                cursor=cursor,
                start_ts=start_ts,
                entity=entity,
                maker_asset=maker_asset,
                taker_asset=taker_asset,
                maker_amt=maker_amt,
                taker_amt=taker_amt,
                timestamp_field=timestamp_field,
                tx_hash_field=tx_hash_field,
                asset_id_gql_type=asset_id_gql_type,
                timestamp_gql_type=timestamp_gql_type,
                id_gql_type=id_gql_type,
            )
        except Exception:
            # Fall through to side-specific queries.
            pass

        # Fallback: query maker-side and taker-side separately and merge (more compatible with some filter schemas).
        maker_page = self._order_filled_events_page_for_asset_side(
            asset_id=asset_id,
            first=first,
            cursor=cursor,
            start_ts=start_ts,
            entity=entity,
            side_field=maker_asset,
            maker_asset=maker_asset,
            taker_asset=taker_asset,
            maker_amt=maker_amt,
            taker_amt=taker_amt,
            timestamp_field=timestamp_field,
            tx_hash_field=tx_hash_field,
            asset_id_gql_type=asset_id_gql_type,
            timestamp_gql_type=timestamp_gql_type,
            id_gql_type=id_gql_type,
        )
        taker_page = self._order_filled_events_page_for_asset_side(
            asset_id=asset_id,
            first=first,
            cursor=cursor,
            start_ts=start_ts,
            entity=entity,
            side_field=taker_asset,
            maker_asset=maker_asset,
            taker_asset=taker_asset,
            maker_amt=maker_amt,
            taker_amt=taker_amt,
            timestamp_field=timestamp_field,
            tx_hash_field=tx_hash_field,
            asset_id_gql_type=asset_id_gql_type,
            timestamp_gql_type=timestamp_gql_type,
            id_gql_type=id_gql_type,
        )

        merged: dict[str, dict[str, Any]] = {}
        for e in maker_page + taker_page:
            if isinstance(e, dict) and isinstance(e.get("id"), str):
                merged[e["id"]] = e
        items = list(merged.values())

        def _key(ev: dict[str, Any]) -> tuple[int, str]:
            ts = _to_int(ev.get("timestamp")) or 0
            return (ts, str(ev.get("id") or ""))

        items.sort(key=_key, reverse=True)
        return items[: int(first)]

    def _order_filled_events_page_for_asset_or(
        self,
        *,
        asset_id: str,
        first: int,
        cursor: TradeCursor | None,
        start_ts: int | None,
        entity: str,
        maker_asset: str,
        taker_asset: str,
        maker_amt: str,
        taker_amt: str,
        timestamp_field: str,
        tx_hash_field: str,
        asset_id_gql_type: str,
        timestamp_gql_type: str,
        id_gql_type: str,
    ) -> list[dict[str, Any]]:
        variables: dict[str, Any] = {"assetID": asset_id, "first": int(first)}

        and_parts: list[str] = [
            "{ or: ["
            f"{{ {maker_asset}: $assetID }},"
            f"{{ {taker_asset}: $assetID }}"
            "] }"
        ]

        # Additional constraints are expressed explicitly via AND to avoid duplicate keys.
        if start_ts is not None:
            and_parts.append("{ " + timestamp_field + "_gte: $startTs }")
            variables["startTs"] = str(int(start_ts))

        if cursor is not None:
            and_parts.append(
                "{ or: ["
                "{ " + timestamp_field + "_lt: $cursorTs },"
                "{ " + timestamp_field + ": $cursorTs, id_lt: $cursorId }"
                "] }"
            )
            variables["cursorTs"] = str(int(cursor.timestamp))
            variables["cursorId"] = cursor.trade_id

        where_block = "and: [" + ", ".join(and_parts) + "]"
        var_defs = [f"$assetID: {asset_id_gql_type}!", "$first: Int!"]
        if start_ts is not None:
            var_defs.append(f"$startTs: {timestamp_gql_type}!")
        if cursor is not None:
            var_defs.extend([f"$cursorTs: {timestamp_gql_type}!", f"$cursorId: {id_gql_type}!"])
        var_defs_s = ", ".join(var_defs)

        query = f"""
        query Fills({var_defs_s}) {{
          {entity}(
            first: $first,
            orderBy: {timestamp_field},
            orderDirection: desc,
            where: {{ {where_block} }}
          ) {{
            id
            timestamp: {timestamp_field}
            transactionHash: {tx_hash_field}
            makerAssetId: {maker_asset}
            takerAssetId: {taker_asset}
            makerAmountFilled: {maker_amt}
            takerAmountFilled: {taker_amt}
          }}
        }}
        """

        data = self._query_first_ok(query, variables)
        events = data.get(entity, []) if isinstance(data, dict) else []
        return [e for e in events if isinstance(e, dict)]

    def _order_filled_events_page_for_asset_side(
        self,
        *,
        asset_id: str,
        first: int,
        cursor: TradeCursor | None,
        start_ts: int | None,
        entity: str,
        side_field: str,
        maker_asset: str,
        taker_asset: str,
        maker_amt: str,
        taker_amt: str,
        timestamp_field: str,
        tx_hash_field: str,
        asset_id_gql_type: str,
        timestamp_gql_type: str,
        id_gql_type: str,
    ) -> list[dict[str, Any]]:
        variables: dict[str, Any] = {"assetID": asset_id, "first": int(first)}

        and_parts: list[str] = ["{ " + side_field + ": $assetID }"]

        if start_ts is not None:
            and_parts.append("{ " + timestamp_field + "_gte: $startTs }")
            variables["startTs"] = str(int(start_ts))

        if cursor is not None:
            and_parts.append(
                "{ or: ["
                "{ " + timestamp_field + "_lt: $cursorTs },"
                "{ " + timestamp_field + ": $cursorTs, id_lt: $cursorId }"
                "] }"
            )
            variables["cursorTs"] = str(int(cursor.timestamp))
            variables["cursorId"] = cursor.trade_id

        where_block = "and: [" + ", ".join(and_parts) + "]"
        var_defs = [f"$assetID: {asset_id_gql_type}!", "$first: Int!"]
        if start_ts is not None:
            var_defs.append(f"$startTs: {timestamp_gql_type}!")
        if cursor is not None:
            var_defs.extend([f"$cursorTs: {timestamp_gql_type}!", f"$cursorId: {id_gql_type}!"])
        var_defs_s = ", ".join(var_defs)

        query = f"""
        query Fills({var_defs_s}) {{
          {entity}(
            first: $first,
            orderBy: {timestamp_field},
            orderDirection: desc,
            where: {{ {where_block} }}
          ) {{
            id
            timestamp: {timestamp_field}
            transactionHash: {tx_hash_field}
            makerAssetId: {maker_asset}
            takerAssetId: {taker_asset}
            makerAmountFilled: {maker_amt}
            takerAmountFilled: {taker_amt}
          }}
        }}
        """

        data = self._query_first_ok(query, variables)
        events = data.get(entity, []) if isinstance(data, dict) else []
        return [e for e in events if isinstance(e, dict)]

    def _order_filled_events_page_with_fallbacks(
        self,
        *,
        token_ids: list[str],
        first: int,
        cursor: TradeCursor | None,
        start_ts: int | None,
    ) -> list[dict[str, Any]]:
        """
        Goldsky subgraph schemas have changed over time; try multiple field-name variants.
        """

        variants = [
            # Common GraphQL field casing.
            {
                "entity": "orderFilledEvents",
                "maker_asset": "makerAssetId",
                "taker_asset": "takerAssetId",
                "maker_amt": "makerAmountFilled",
                "taker_amt": "takerAmountFilled",
            },
            # Observed variant in some examples.
            {
                "entity": "orderFilledEvents",
                "maker_asset": "makerAssetID",
                "taker_asset": "takerAssetID",
                "maker_amt": "makerAmountFilled",
                "taker_amt": "takerAmountFilled",
            },
        ]

        last_err: Exception | None = None
        for v in variants:
            try:
                return self._order_filled_events_page(
                    token_ids=token_ids,
                    first=first,
                    cursor=cursor,
                    start_ts=start_ts,
                    entity=v["entity"],
                    maker_asset=v["maker_asset"],
                    taker_asset=v["taker_asset"],
                    maker_amt=v["maker_amt"],
                    taker_amt=v["taker_amt"],
                )
            except Exception as e:
                last_err = e
                continue

        if last_err is not None:
            raise last_err
        return []

    # `trades`-based querying removed: recent orderbook-subgraph deployments do not expose it.

    def _order_filled_events_page(
        self,
        *,
        token_ids: list[str],
        first: int,
        cursor: TradeCursor | None,
        start_ts: int | None,
        entity: str,
        maker_asset: str,
        taker_asset: str,
        maker_amt: str,
        taker_amt: str,
    ) -> list[dict[str, Any]]:
        variables: dict[str, Any] = {"tokenIds": token_ids, "first": int(first)}

        and_parts: list[str] = []

        and_parts.append(
            "{ or: ["
            f"{{ {maker_asset}_in: $tokenIds }},"
            f"{{ {taker_asset}_in: $tokenIds }}"
            "] }"
        )

        if start_ts is not None:
            and_parts.append("{ timestamp_gte: $startTs }")
            variables["startTs"] = str(int(start_ts))

        if cursor is not None:
            and_parts.append(
                "{ or: ["
                "{ timestamp_lt: $cursorTs },"
                "{ timestamp: $cursorTs, id_lt: $cursorId }"
                "] }"
            )
            variables["cursorTs"] = str(int(cursor.timestamp))
            variables["cursorId"] = cursor.trade_id

        where_block = "and: [" + ", ".join(and_parts) + "]"
        var_defs = ["$tokenIds: [String!]!", "$first: Int!"]
        if start_ts is not None:
            var_defs.append("$startTs: BigInt!")
        if cursor is not None:
            var_defs.extend(["$cursorTs: BigInt!", "$cursorId: ID!"])
        var_defs_s = ", ".join(var_defs)

        query = f"""
        query Fills({var_defs_s}) {{
          {entity}(
            first: $first,
            orderBy: timestamp,
            orderDirection: desc,
            where: {{ {where_block} }}
          ) {{
            id
            timestamp
            transactionHash
            {maker_asset}
            {taker_asset}
            {maker_amt}
            {taker_amt}
          }}
        }}
        """

        data = self._query_first_ok(query, variables)
        events = data.get(entity, []) if isinstance(data, dict) else []
        return [e for e in events if isinstance(e, dict)]

    def _discover_event_query_specs(self) -> list[FillQuerySpec]:
        cached = getattr(self, "_event_specs", None)
        if isinstance(cached, list) and cached and all(isinstance(x, FillQuerySpec) for x in cached):
            return cached

        qfields = set(self.introspect_query_fields())
        qtypes = self.introspect_query_field_types()

        # Priority: filled events first, then matched events.
        candidates: list[tuple[str, str]] = []
        if "orderFilledEvents" in qfields and "orderFilledEvent" in qfields:
            candidates.append(("orderFilledEvents", "OrderFilledEvent"))
        if "ordersMatchedEvents" in qfields and "ordersMatchedEvent" in qfields:
            candidates.append(("ordersMatchedEvents", "OrdersMatchedEvent"))

        # If introspection fails, keep a safe default.
        if not candidates:
            candidates = [("orderFilledEvents", "OrderFilledEvent")]

        specs: list[FillQuerySpec] = []
        for entity, item_type in candidates:
            entity_fields = self.introspect_type_fields(item_type)
            if not entity_fields:
                continue

            def pick_field(opts: list[str]) -> str | None:
                for o in opts:
                    if o in entity_fields:
                        return o
                return None

            maker_asset = pick_field(["makerAssetId", "makerAssetID", "makerAsset"])
            taker_asset = pick_field(["takerAssetId", "takerAssetID", "takerAsset"])
            maker_amt = pick_field(["makerAmountFilled", "makerAmount"])
            taker_amt = pick_field(["takerAmountFilled", "takerAmount"])
            timestamp_field = pick_field(["timestamp", "createdAt", "time"]) or "timestamp"
            tx_hash_field = pick_field(["transactionHash", "txHash", "hash"]) or "transactionHash"

            if not maker_asset or not taker_asset or not maker_amt or not taker_amt:
                continue

            asset_type = _extract_base_gql_named_type(entity_fields.get(maker_asset, "")) or "String"
            ts_type = _extract_base_gql_named_type(entity_fields.get(timestamp_field, "")) or "BigInt"
            id_type = _extract_base_gql_named_type(entity_fields.get("id", "")) or "ID"

            connection_entity = None
            if f"{entity}Connection" in qfields:
                connection_entity = f"{entity}Connection"
            elif f"{entity}Connection" in qtypes:
                connection_entity = f"{entity}Connection"

            specs.append(
                FillQuerySpec(
                    entity=entity,
                    maker_asset_field=maker_asset,
                    taker_asset_field=taker_asset,
                    maker_amount_field=maker_amt,
                    taker_amount_field=taker_amt,
                    timestamp_field=timestamp_field,
                    tx_hash_field=tx_hash_field,
                    asset_id_gql_type=asset_type,
                    timestamp_gql_type=ts_type,
                    id_gql_type=id_type,
                    connection_entity=connection_entity,
                )
            )

        if not specs:
            # Last-resort defaults (matches the user's introspected schema).
            specs = [
                FillQuerySpec(
                    entity="orderFilledEvents",
                    maker_asset_field="makerAssetId",
                    taker_asset_field="takerAssetId",
                    maker_amount_field="makerAmountFilled",
                    taker_amount_field="takerAmountFilled",
                    timestamp_field="timestamp",
                    tx_hash_field="transactionHash",
                    asset_id_gql_type="String",
                    timestamp_gql_type="BigInt",
                    id_gql_type="ID",
                    connection_entity=None,
                )
            ]

        setattr(self, "_event_specs", specs)
        return specs

    # -----------------------
    # Internals
    # -----------------------
    def _query_first_ok(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        endpoints = list(self._subgraph.endpoints) or []
        if not endpoints:
            raise ValueError("No subgraph endpoints configured.")

        errors: list[str] = []
        for url in endpoints:
            try:
                return self._query(url, query, variables)
            except Exception as e:
                msg = f"{url}: {type(e).__name__}: {e}"
                errors.append(msg)
                continue

        raise RuntimeError("All subgraph endpoints failed:\n" + "\n".join(errors))

    def _query(self, url: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        payload = {"query": query, "variables": variables}
        resp = self._request("POST", url, json_body=payload)
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            snippet = (resp.text or "")[:500]
            raise ValueError(f"Non-JSON response from subgraph {url}: {snippet}") from e

        if not isinstance(data, dict):
            raise ValueError(f"Unexpected GraphQL response type: {type(data)!r}")

        errs = data.get("errors")
        if errs:
            raise ValueError(f"GraphQL errors: {errs}")

        out = data.get("data")
        if not isinstance(out, dict):
            raise ValueError("GraphQL response did not include a data object")
        return out

    def _request(self, method: str, url: str, *, json_body: dict[str, Any]) -> "requests.Response":
        assert requests is not None

        last_exc: Exception | None = None
        attempts = max(1, int(self._http.max_retries) + 1)

        for attempt in range(1, attempts + 1):
            try:
                resp = self._session.request(
                    method=method,
                    url=url,
                    json=json_body,
                    timeout=self._http.timeout_seconds,
                )
                if resp.status_code in _RETRY_STATUS_CODES and attempt < attempts:
                    self._sleep_backoff(attempt, resp)
                    continue
                resp.raise_for_status()
                return resp
            except Exception as e:
                last_exc = e
                if attempt >= attempts:
                    raise
                self._sleep_backoff(attempt, None)

        raise RuntimeError("Unreachable") from last_exc

    def _sleep_backoff(self, attempt: int, resp: "requests.Response | None") -> None:
        retry_after = None
        if resp is not None:
            ra = resp.headers.get("Retry-After")
            if ra:
                try:
                    retry_after = float(ra)
                except Exception:
                    retry_after = None

        base = float(self._http.backoff_base_seconds) * (2 ** max(0, attempt - 1))
        jitter = random.random() * base * 0.25
        delay = base + jitter
        if retry_after is not None:
            delay = max(delay, retry_after)
        delay = min(delay, float(self._http.backoff_max_seconds))
        time.sleep(delay)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("0x"):
            return None
        try:
            return int(s)
        except Exception:
            return None
    return None


def _render_gql_type(type_obj: Any) -> str:
    """
    Render GraphQL type object as a string like `[OrderFilledEvent!]!`.
    """

    def _walk(t: Any) -> str:
        if not isinstance(t, dict):
            return "Unknown"
        kind = t.get("kind")
        name = t.get("name")
        of_type = t.get("ofType")

        if kind == "NON_NULL":
            return _walk(of_type) + "!"
        if kind == "LIST":
            return "[" + _walk(of_type) + "]"
        if isinstance(name, str) and name:
            return name
        return "Unknown"

    return _walk(type_obj)


def _extract_base_gql_named_type(rendered: str) -> str | None:
    # Strip wrappers like [X!]! -> X
    s = (rendered or "").replace("[", "").replace("]", "").replace("!", "").strip()
    return s or None


def _extract_list_item_type_name_from_rendered(rendered: str) -> str | None:
    # For e.g. [OrderFilledEvent!]! -> OrderFilledEvent
    s = _extract_base_gql_named_type(rendered)
    return s
