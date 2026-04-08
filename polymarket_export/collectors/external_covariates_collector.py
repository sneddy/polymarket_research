from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Literal, Sequence

from clients.binance_archive_client import BinanceArchiveClient
from clients.binance_client import BinanceClient
from clients.edgar_client import EdgarClient
from clients.fred_client import FredClient
from configs.external_covariates_config import DEFAULT_EXTERNAL_SERIES
from configs.external_covariates_config import EXTERNAL_COVARIATE_REGISTRY
from configs.external_covariates_config import ExternalCovariateSpec
from storage.parquet_store import ParquetStore


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


class ExternalCovariatesCollector:
    """Download and normalize external market covariates from multiple providers."""

    def __init__(
        self,
        *,
        binance_client: BinanceClient | None = None,
        binance_archive_client: BinanceArchiveClient | None = None,
        edgar_client: EdgarClient | None = None,
        fred_client: FredClient | None = None,
        store: ParquetStore | None = None,
    ) -> None:
        self._binance = binance_client
        self._binance_archive = binance_archive_client
        self._edgar = edgar_client
        self._fred = fred_client
        self._store = store or ParquetStore()

    def download_series(
        self,
        series_id: str,
        *,
        start_date: datetime | str,
        end_date: datetime | str,
        frame_type: FrameType | None = None,
        binance_source: str = "api",
        archive_tail_days: int = 45,
        show_progress: bool = True,
    ) -> Any:
        spec = self._resolve_spec(series_id)
        rows = self._download_rows_for_spec(
            spec,
            start_date=start_date,
            end_date=end_date,
            binance_source=binance_source,
            archive_tail_days=archive_tail_days,
            show_progress=show_progress,
        )
        return self._to_frame(rows, frame_type=frame_type)

    def download_many(
        self,
        series_ids: Sequence[str] | None = None,
        *,
        start_date: datetime | str,
        end_date: datetime | str,
        frame_type: FrameType | None = None,
        binance_source: str = "api",
        archive_tail_days: int = 45,
        show_progress: bool = True,
    ) -> Any:
        selected = list(series_ids) if series_ids is not None else list(DEFAULT_EXTERNAL_SERIES)
        rows: list[dict[str, Any]] = []
        tqdm = _resolve_tqdm(show_progress=show_progress)
        if show_progress and tqdm is None:
            logger.warning("show_progress=True but tqdm is unavailable in the active environment.")
        pbar = (
            tqdm(
                total=len(selected),
                disable=False,
                unit="series",
                desc="External covariates",
                leave=True,
            )
            if tqdm is not None
            else None
        )

        for series_id in selected:
            spec = self._resolve_spec(series_id)
            logger.info(
                "Downloading external covariate | series_id=%s provider=%s symbol=%s interval=%s binance_source=%s",
                spec.series_id,
                spec.provider,
                spec.provider_symbol,
                spec.interval,
                binance_source if spec.provider == "binance" else "n/a",
            )
            rows.extend(
                self._download_rows_for_spec(
                    spec,
                    start_date=start_date,
                    end_date=end_date,
                    binance_source=binance_source,
                    archive_tail_days=archive_tail_days,
                    show_progress=show_progress,
                )
            )
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "series_id": spec.series_id,
                        "provider": spec.provider,
                        "rows": len(rows),
                    }
                )
        if pbar is not None:
            pbar.close()
        return self._to_frame(rows, frame_type=frame_type)

    def save_to_parquet(self, df: Any, path: str, *, partition_cols: list[str] | None = None) -> str:
        out = self._store.save(df, path, partition_cols=partition_cols)
        return str(out)

    def _download_rows_for_spec(
        self,
        spec: ExternalCovariateSpec,
        *,
        start_date: datetime | str,
        end_date: datetime | str,
        binance_source: str,
        archive_tail_days: int,
        show_progress: bool,
    ) -> list[dict[str, Any]]:
        if spec.provider == "binance":
            if self._binance is None:
                self._binance = BinanceClient()
            if self._binance_archive is None:
                self._binance_archive = BinanceArchiveClient()
            if str(binance_source).lower() == "archive":
                raw = [
                    self._normalize_row(spec, self._binance.normalize_kline_row(row))
                    for row in self._binance_archive.download_klines(
                        spec.provider_symbol,
                        interval=spec.interval,
                        start_date=start_date,
                        end_date=end_date,
                        tail_days=archive_tail_days,
                        show_progress=show_progress,
                    )
                ]
            else:
                raw = [
                    self._normalize_row(spec, self._binance.normalize_kline_row(row))
                    for row in self._binance.iter_klines(
                        spec.provider_symbol,
                        interval=spec.interval,
                        start_date=start_date,
                        end_date=end_date,
                    )
                ]
            return raw

        if spec.provider == "fred":
            if self._fred is None:
                self._fred = FredClient()
            raw = [
                self._normalize_row(spec, row)
                for row in self._fred.download_series_csv(
                    spec.provider_symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
            ]
            return raw

        if spec.provider == "sec_edgar":
            if self._edgar is None:
                self._edgar = EdgarClient()
            raw = [
                self._normalize_row(spec, row)
                for row in self._edgar.download_daily_form_counts(
                    spec.provider_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    show_progress=show_progress,
                )
            ]
            return raw

        raise ValueError(f"Unsupported provider: {spec.provider!r}")

    @staticmethod
    def _normalize_row(spec: ExternalCovariateSpec, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "series_id": spec.series_id,
            "provider": spec.provider,
            "provider_symbol": spec.provider_symbol,
            "category": spec.category,
            "units": spec.units,
            "interval": spec.interval,
            "value_field": spec.value_field,
            "timestamp_utc": row.get("timestamp_utc"),
            "close_timestamp_utc": row.get("close_timestamp_utc"),
            "value": row.get(spec.value_field) if spec.value_field in row else row.get("value"),
            "open": row.get("open"),
            "high": row.get("high"),
            "low": row.get("low"),
            "close": row.get("close"),
            "volume": row.get("volume"),
            "quote_asset_volume": row.get("quote_asset_volume"),
            "trade_count": row.get("trade_count"),
            "taker_buy_base_volume": row.get("taker_buy_base_volume"),
            "taker_buy_quote_volume": row.get("taker_buy_quote_volume"),
        }

    @staticmethod
    def _resolve_spec(series_id: str) -> ExternalCovariateSpec:
        try:
            return EXTERNAL_COVARIATE_REGISTRY[str(series_id)]
        except KeyError as exc:
            known = ", ".join(sorted(EXTERNAL_COVARIATE_REGISTRY))
            raise ValueError(f"Unknown series_id={series_id!r}. Known ids: {known}") from exc

    @staticmethod
    def _to_frame(rows: list[dict[str, Any]], *, frame_type: FrameType | None) -> Any:
        frame = frame_type or _default_frame_type()
        columns = [
            "series_id",
            "provider",
            "provider_symbol",
            "category",
            "units",
            "interval",
            "value_field",
            "timestamp_utc",
            "close_timestamp_utc",
            "value",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]
        if frame == "pandas":
            import pandas as pd

            df = pd.DataFrame(rows, columns=columns)
            if "timestamp_utc" in df.columns:
                df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
            if "close_timestamp_utc" in df.columns:
                df["close_timestamp_utc"] = pd.to_datetime(df["close_timestamp_utc"], utc=True, errors="coerce")
            for col in (
                "value",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_asset_volume",
                "trade_count",
                "taker_buy_base_volume",
                "taker_buy_quote_volume",
            ):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.sort_values(["series_id", "timestamp_utc"], kind="stable").reset_index(drop=True)

        if frame == "polars":
            import polars as pl

            df = pl.DataFrame(rows, schema=columns if not rows else None)
            if "timestamp_utc" in df.columns:
                df = df.with_columns(pl.col("timestamp_utc").cast(pl.Datetime(time_zone="UTC"), strict=False))
            if "close_timestamp_utc" in df.columns:
                df = df.with_columns(
                    pl.col("close_timestamp_utc").cast(pl.Datetime(time_zone="UTC"), strict=False)
                )
            return df.sort(["series_id", "timestamp_utc"])

        raise ValueError(f"Unknown frame_type: {frame!r}")
