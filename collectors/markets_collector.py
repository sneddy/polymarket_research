from __future__ import annotations

from datetime import UTC, datetime
import logging
from typing import Any, Literal

from clients.gamma_client import GammaClient
from storage.parquet_store import ParquetStore
from utils import to_snake_case_keys


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


class MarketsCollector:
    def __init__(self, client: GammaClient, *, store: ParquetStore | None = None) -> None:
        self._client = client
        self._store = store or ParquetStore()

    def download_markets(
        self,
        *,
        active: bool = True,
        limit: int = 100,
        max_pages: int | None = None,
        frame_type: FrameType | None = None,
        normalize_keys: bool = True,
        show_progress: bool = True,
        estimate_total: bool = True,
        **params: Any,
    ) -> Any:
        tqdm = _resolve_tqdm(show_progress=show_progress)
        if show_progress and tqdm is None:
            logger.warning("show_progress=True but tqdm is unavailable in the active environment.")

        total = None
        if show_progress and estimate_total:
            total = self._client.estimate_markets_count(active=active, **params)
            if total is None:
                logger.info("Could not estimate total market count; showing open-ended progress.")

        pbar = (
            tqdm(
                total=total,
                disable=False,
                unit="market",
                desc=f"Downloading markets ({'active' if active else 'inactive'})",
                leave=True,
            )
            if tqdm is not None
            else None
        )

        rows: list[dict[str, Any]] = []
        progress_every = max(25, int(limit))
        try:
            for item in self._client.iter_markets(active=active, limit=limit, max_pages=max_pages, **params):
                if not isinstance(item, dict):
                    continue
                rows.append(to_snake_case_keys(item) if normalize_keys else item)
                if pbar is not None:
                    pbar.update(1)
                    if len(rows) % progress_every == 0:
                        pbar.set_postfix({"fetched": len(rows), "state": "active" if active else "inactive"})
            if pbar is not None:
                pbar.set_postfix({"fetched": len(rows), "state": "active" if active else "inactive"})
        finally:
            if pbar is not None:
                pbar.close()

        return self._to_frame(rows, frame_type=frame_type)

    def download_market_universe(
        self,
        *,
        include_active: bool = True,
        include_inactive: bool = True,
        limit: int = 100,
        max_pages: int | None = None,
        frame_type: FrameType | None = None,
        normalize_keys: bool = True,
        dedupe: bool = True,
        show_progress: bool = True,
        estimate_total: bool = True,
        **params: Any,
    ) -> Any:
        """
        Download a broad market universe from Gamma.

        By default this pulls both active and inactive markets and deduplicates them by condition id / id / slug.
        """

        states: list[bool] = []
        if include_active:
            states.append(True)
        if include_inactive:
            states.append(False)
        if not states:
            raise ValueError("At least one of include_active/include_inactive must be True.")

        tqdm = _resolve_tqdm(show_progress=show_progress)
        if show_progress and tqdm is None:
            logger.warning("show_progress=True but tqdm is unavailable in the active environment.")

        total = None
        if show_progress and estimate_total:
            estimates: list[int] = []
            for active in states:
                est = self._client.estimate_markets_count(active=active, **params)
                if est is None:
                    estimates = []
                    break
                estimates.append(int(est))
            if estimates:
                total = int(sum(estimates))
            else:
                logger.info("Could not estimate market-universe total; showing open-ended progress.")

        pbar = (
            tqdm(
                total=total,
                disable=False,
                unit="market",
                desc="Downloading market metadata",
                leave=True,
            )
            if tqdm is not None
            else None
        )

        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        progress_every = max(25, int(limit))
        try:
            for active in states:
                state_label = "active" if active else "inactive"
                if pbar is not None:
                    pbar.set_postfix({"state": state_label, "fetched": len(rows)})
                try:
                    for item in self._client.iter_markets(active=active, limit=limit, max_pages=max_pages, **params):
                        if not isinstance(item, dict):
                            continue
                        rows.append(to_snake_case_keys(item) if normalize_keys else item)
                        if pbar is not None:
                            pbar.update(1)
                            if len(rows) % progress_every == 0:
                                pbar.set_postfix({"state": state_label, "fetched": len(rows)})
                except Exception as e:
                    errors.append(f"active={active}: {type(e).__name__}: {e}")
                    logger.warning("Market universe fetch failed for active=%s: %s", active, e)
                    continue
            if pbar is not None:
                pbar.set_postfix({"state": "done", "fetched": len(rows)})
        finally:
            if pbar is not None:
                pbar.close()

        if not rows:
            detail = "; ".join(errors) if errors else "no rows returned"
            raise RuntimeError(f"Failed to download market universe: {detail}")

        if dedupe:
            rows = self._dedupe_market_rows(rows, normalized=normalize_keys)

        return self._to_frame(rows, frame_type=frame_type)

    def summarize_markets(self, df: Any, *, frame_type: FrameType | None = None) -> dict[str, Any]:
        """
        Build high-level statistics over a market DataFrame.
        """

        pdf = self._to_pandas(df)
        out: dict[str, Any] = {
            "generated_at_utc": datetime.now(tz=UTC).isoformat(),
            "markets_total": int(len(pdf)),
        }
        if pdf.empty:
            return out

        active = self._coerce_bool_series(pdf, ["active"])
        closed = self._coerce_bool_series(pdf, ["closed"])
        archived = self._coerce_bool_series(pdf, ["archived"])

        out.update(
            {
                "markets_active": int(active.sum()),
                "markets_inactive": int((~active).sum()),
                "markets_closed": int(closed.sum()),
                "markets_archived": int(archived.sum()),
                "markets_open": int((~closed).sum()),
            }
        )

        liquidity = self._coalesce_numeric_series(pdf, ["liquidity_clob", "liquidity_num", "liquidity"])
        volume_total = self._coalesce_numeric_series(pdf, ["volume_clob", "volume_num", "volume"])
        volume_24h = self._coalesce_numeric_series(pdf, ["volume24hr_clob", "volume24hr"])
        volume_1w = self._coalesce_numeric_series(pdf, ["volume1wk_clob", "volume1wk"])
        spread = self._coalesce_numeric_series(pdf, ["spread"])

        out.update(
            {
                "liquidity_total": float(liquidity.sum()),
                "liquidity_median": float(liquidity.median()),
                "volume_total": float(volume_total.sum()),
                "volume_24h_total": float(volume_24h.sum()),
                "volume_1w_total": float(volume_1w.sum()),
                "spread_median": float(spread.median()),
            }
        )
        return out

    def rank_markets(
        self,
        df: Any,
        *,
        top_n: int = 25,
        min_liquidity: float | None = None,
        min_volume_24h: float | None = None,
        frame_type: FrameType | None = None,
    ) -> Any:
        """
        Rank markets for discovery using a blended score from liquidity/volume/spread.
        """

        import pandas as pd

        pdf = self._to_pandas(df).copy()
        if pdf.empty:
            return self._to_frame([], frame_type=frame_type)

        work = pd.DataFrame(index=pdf.index)
        work["condition_id"] = self._coalesce_text_series(pdf, ["condition_id", "conditionId"])
        work["slug"] = self._coalesce_text_series(pdf, ["slug"])
        work["question"] = self._coalesce_text_series(pdf, ["question"])
        work["active"] = self._coerce_bool_series(pdf, ["active"])
        work["closed"] = self._coerce_bool_series(pdf, ["closed"])
        work["liquidity"] = self._coalesce_numeric_series(pdf, ["liquidity_clob", "liquidity_num", "liquidity"])
        work["volume_total"] = self._coalesce_numeric_series(pdf, ["volume_clob", "volume_num", "volume"])
        work["volume_24h"] = self._coalesce_numeric_series(pdf, ["volume24hr_clob", "volume24hr"])
        work["volume_1w"] = self._coalesce_numeric_series(pdf, ["volume1wk_clob", "volume1wk"])
        work["spread"] = self._coalesce_numeric_series(pdf, ["spread"])
        work["end_date"] = self._coalesce_datetime_series(pdf, ["end_date", "endDate"])

        if min_liquidity is not None:
            work = work[work["liquidity"] >= float(min_liquidity)]
        if min_volume_24h is not None:
            work = work[work["volume_24h"] >= float(min_volume_24h)]

        if work.empty:
            return self._to_frame([], frame_type=frame_type)

        for col in ("liquidity", "volume_total", "volume_24h", "volume_1w"):
            work[f"{col}_pct"] = work[col].rank(pct=True, method="average").fillna(0.0)

        spread_pct = work["spread"].rank(pct=True, method="average").fillna(0.0)
        work["spread_pct"] = spread_pct
        work["spread_quality_pct"] = 1.0 - spread_pct

        # Weighted preference: recent activity + liquidity first; tighter spreads are better.
        work["market_score"] = (
            0.45 * work["volume_24h_pct"]
            + 0.25 * work["liquidity_pct"]
            + 0.20 * work["volume_1w_pct"]
            + 0.05 * work["volume_total_pct"]
            + 0.05 * work["spread_quality_pct"]
        )

        ranked = work.sort_values(["market_score", "volume_24h", "liquidity"], ascending=[False, False, False])
        if top_n > 0:
            ranked = ranked.head(int(top_n))

        cols = [
            "condition_id",
            "slug",
            "question",
            "active",
            "closed",
            "liquidity",
            "volume_24h",
            "volume_1w",
            "volume_total",
            "spread",
            "market_score",
            "end_date",
        ]
        ranked = ranked[[c for c in cols if c in ranked.columns]].reset_index(drop=True)
        return self._to_frame(ranked.to_dict(orient="records"), frame_type=frame_type)

    def download_market_meta(
        self,
        *,
        include_active: bool = True,
        include_inactive: bool = True,
        limit: int = 100,
        max_pages: int | None = None,
        top_n: int = 25,
        min_liquidity: float | None = None,
        min_volume_24h: float | None = None,
        frame_type: FrameType | None = None,
        normalize_keys: bool = True,
        dedupe: bool = True,
        show_progress: bool = True,
        estimate_total: bool = True,
        **params: Any,
    ) -> dict[str, Any]:
        """
        Fetch the market universe and produce:
          - `markets`: full market frame
          - `summary`: aggregate statistics
          - `top_markets`: ranked subset for discovery

        Note:
          `top_n` limits only the final ranked output. The universe download still
          fetches all matching markets (unless you bound it with `max_pages`/filters).
        """

        markets = self.download_market_universe(
            include_active=include_active,
            include_inactive=include_inactive,
            limit=limit,
            max_pages=max_pages,
            frame_type=frame_type,
            normalize_keys=normalize_keys,
            dedupe=dedupe,
            show_progress=show_progress,
            estimate_total=estimate_total,
            **params,
        )
        summary = self.summarize_markets(markets, frame_type=frame_type)
        top_markets = self.rank_markets(
            markets,
            top_n=top_n,
            min_liquidity=min_liquidity,
            min_volume_24h=min_volume_24h,
            frame_type=frame_type,
        )
        return {"markets": markets, "summary": summary, "top_markets": top_markets}

    def save_to_parquet(self, df: Any, path: str, *, partition_cols: list[str] | None = None) -> str:
        out = self._store.save(df, path, partition_cols=partition_cols)
        return str(out)

    @staticmethod
    def _dedupe_market_rows(rows: list[dict[str, Any]], *, normalized: bool) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        key_candidates = (
            ("condition_id", "id", "slug") if normalized else ("conditionId", "condition_id", "id", "slug")
        )
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = None
            for cand in key_candidates:
                value = row.get(cand)
                if value is not None and str(value).strip():
                    key = str(value)
                    break
            if key is None:
                out.append(row)
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    @staticmethod
    def _to_pandas(df: Any) -> Any:
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            return df
        try:
            import polars as pl

            if isinstance(df, pl.DataFrame):
                return df.to_pandas()
        except Exception:
            pass
        if isinstance(df, list):
            return pd.DataFrame(df)
        return pd.DataFrame()

    @staticmethod
    def _coalesce_text_series(pdf: Any, candidates: list[str]) -> Any:
        import pandas as pd

        for c in candidates:
            if c in pdf.columns:
                return pdf[c].astype("string")
        return pd.Series(index=pdf.index, dtype="string")

    @staticmethod
    def _coalesce_numeric_series(pdf: Any, candidates: list[str]) -> Any:
        import pandas as pd

        for c in candidates:
            if c in pdf.columns:
                return pd.to_numeric(pdf[c], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=pdf.index, dtype="float64")

    @staticmethod
    def _coerce_bool_series(pdf: Any, candidates: list[str]) -> Any:
        import pandas as pd

        raw = None
        for c in candidates:
            if c in pdf.columns:
                raw = pdf[c]
                break
        if raw is None:
            return pd.Series(False, index=pdf.index, dtype="bool")

        def _one(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                s = v.strip().lower()
                if s in {"true", "1", "yes", "y"}:
                    return True
                if s in {"false", "0", "no", "n", ""}:
                    return False
            return False

        return raw.map(_one).astype("bool")

    @staticmethod
    def _coalesce_datetime_series(pdf: Any, candidates: list[str]) -> Any:
        import pandas as pd

        for c in candidates:
            if c in pdf.columns:
                return pd.to_datetime(pdf[c], utc=True, errors="coerce")
        return pd.Series(index=pdf.index, dtype="datetime64[ns, UTC]")

    @staticmethod
    def _to_frame(rows: list[dict[str, Any]], *, frame_type: FrameType | None) -> Any:
        frame = frame_type or _default_frame_type()
        if frame == "pandas":
            import pandas as pd

            df = pd.DataFrame(rows)
            for col in df.columns:
                if any(s in col for s in ("date", "_at", "timestamp")):
                    if df[col].dtype == object:
                        df[col] = pd.to_datetime(df[col].astype("string").str.replace("Z", "+00:00", regex=False), errors="ignore", utc=True)
            return df

        if frame == "polars":
            import polars as pl

            df = pl.DataFrame(rows)
            for col in df.columns:
                if any(s in col for s in ("date", "_at", "timestamp")):
                    if df.schema[col] == pl.Utf8:
                        df = df.with_columns(
                            pl.col(col)
                            .str.replace("Z$", "+00:00")
                            .str.strptime(pl.Datetime, strict=False)
                            .dt.replace_time_zone("UTC")
                            .alias(col)
                        )
            return df

        raise ValueError(f"Unknown frame_type: {frame!r}")
