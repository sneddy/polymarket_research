from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path
import shutil
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.benchmark_window_config import DEFAULT_BENCHMARK_END_DATE
from configs.benchmark_window_config import DEFAULT_BENCHMARK_START_DATE
from configs.benchmark_window_config import DEFAULT_EXTERNAL_EVENTS_OUT
from collectors.external_covariates_collector import ExternalCovariatesCollector
from storage.parquet_store import _is_pandas_df
from storage.parquet_store import _is_polars_df


logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = Path("logs")
DEFAULT_EVENT_SERIES = (
    "edgar_total_filings",
    "edgar_8k_filings",
    "edgar_10q_filings",
    "edgar_10k_filings",
)


def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _resolve_tqdm(show_progress: bool) -> object | None:
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


def setup_logging(level: str, *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper()))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def _next_start_date(default_start: str, last_timestamp: datetime | None) -> str:
    if last_timestamp is None:
        return default_start
    next_dt = last_timestamp.astimezone(UTC) + timedelta(days=1)
    return next_dt.replace(hour=0, minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "Z")


def _dataset_exists(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    return any(path.rglob("*.parquet"))


def _series_last_timestamps(df: object) -> dict[str, datetime]:
    if _is_pandas_df(df):
        import pandas as pd

        work = df.copy()
        work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="coerce")
        work = work.dropna(subset=["series_id", "timestamp_utc"])
        grouped = work.groupby("series_id", dropna=True)["timestamp_utc"].max()
        return {
            str(series_id): ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            for series_id, ts in grouped.items()
        }

    if _is_polars_df(df):
        import polars as pl

        work = df.with_columns(
            pl.col("timestamp_utc")
            .cast(pl.Datetime(time_zone="UTC"), strict=False)
            .alias("timestamp_utc")
        ).drop_nulls(["series_id", "timestamp_utc"])
        grouped = work.group_by("series_id").agg(pl.col("timestamp_utc").max().alias("timestamp_utc"))
        return {
            str(row["series_id"]): row["timestamp_utc"]
            for row in grouped.to_dicts()
            if row.get("series_id") is not None and row.get("timestamp_utc") is not None
        }

    raise TypeError(f"Unsupported dataframe type: {type(df)!r}")


def _concat_frames(existing_df: object | None, new_frames: list[object]) -> object | None:
    frames = []
    if existing_df is not None:
        frames.append(existing_df)
    frames.extend(new_frames)
    if not frames:
        return None

    first = frames[0]
    if _is_pandas_df(first):
        import pandas as pd

        return pd.concat(frames, ignore_index=True)

    if _is_polars_df(first):
        import polars as pl

        return pl.concat(frames, how="vertical_relaxed")

    raise TypeError(f"Unsupported dataframe type: {type(first)!r}")


def _dedupe_frame(df: object) -> object:
    if _is_pandas_df(df):
        work = df.copy()
        work = work.sort_values(["series_id", "timestamp_utc"], kind="stable")
        work = work.drop_duplicates(subset=["series_id", "timestamp_utc"], keep="last")
        return work.reset_index(drop=True)

    if _is_polars_df(df):
        return (
            df.sort(["series_id", "timestamp_utc"], descending=[False, False], nulls_last=True)
            .unique(subset=["series_id", "timestamp_utc"], keep="last")
            .sort(["series_id", "timestamp_utc"], descending=[False, False], nulls_last=True)
        )

    raise TypeError(f"Unsupported dataframe type: {type(df)!r}")


def _frame_len(df: object | None) -> int:
    return 0 if df is None else int(len(df))


def _replace_output_atomically(df: object, collector: ExternalCovariatesCollector, out_path: Path, partition_cols: list[str] | None) -> Path:
    tmp_path = out_path.parent / f"{out_path.name}.tmp_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}"
    if tmp_path.exists():
        if tmp_path.is_dir():
            shutil.rmtree(tmp_path)
        else:
            tmp_path.unlink()

    saved_tmp = Path(collector.save_to_parquet(df, tmp_path, partition_cols=partition_cols))
    if out_path.exists():
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()
    shutil.move(str(saved_tmp), str(out_path))
    return out_path


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download normalized SEC EDGAR event-count series for the benchmark date window."
    )
    p.add_argument(
        "--start-date",
        default=DEFAULT_BENCHMARK_START_DATE,
        help=f"ISO start date. Default: {DEFAULT_BENCHMARK_START_DATE}.",
    )
    p.add_argument(
        "--end-date",
        default=DEFAULT_BENCHMARK_END_DATE,
        help=f"ISO end date. Default: {DEFAULT_BENCHMARK_END_DATE}.",
    )
    p.add_argument(
        "--series-id",
        action="append",
        dest="series_ids",
        default=None,
        help="EDGAR series id to download. Repeatable. Defaults to the standard EDGAR benchmark event set.",
    )
    p.add_argument(
        "--out",
        default=str(DEFAULT_EXTERNAL_EVENTS_OUT),
        help="Parquet path or partitioned dataset directory.",
    )
    p.add_argument("--frame-type", choices=["pandas", "polars"], default=None)
    p.add_argument(
        "--partition-cols",
        nargs="*",
        default=["series_id"],
        help="Optional parquet partition columns. Default: series_id.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    p.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help="Directory for per-run log files.",
    )
    p.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    return p.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_path = log_dir / f"get_events_{timestamp}.log"
    setup_logging(args.log_level, log_path=log_path)
    logger.info("run logging initialized | log_path=%s", log_path)

    selected = list(args.series_ids) if args.series_ids is not None else list(DEFAULT_EVENT_SERIES)
    out_path = Path(args.out).expanduser().resolve()
    logger.info(
        "event download started | start_date=%s end_date=%s series_count=%s out=%s",
        args.start_date,
        args.end_date,
        len(selected),
        out_path,
    )
    logger.info(
        "sec user-agent reminder | source=config.py field=SecEdgarConfig.user_agent requirement=descriptive-contact-info",
    )

    collector = ExternalCovariatesCollector()
    existing_df = None
    last_seen_by_series: dict[str, datetime] = {}
    if _dataset_exists(out_path):
        logger.info("existing event dataset detected | out=%s", out_path)
        existing_df = collector._store.load(out_path, frame_type=args.frame_type)
        last_seen_by_series = _series_last_timestamps(existing_df)
        logger.info(
            "existing event dataset loaded | rows=%s tracked_series=%s",
            _frame_len(existing_df),
            len(last_seen_by_series),
        )

    downloaded_frames: list[object] = []
    tqdm = _resolve_tqdm(show_progress=not args.no_progress)
    if not args.no_progress and tqdm is None:
        logger.warning("show_progress=True but tqdm is unavailable in the active environment.")
    pbar = (
        tqdm(
            total=len(selected),
            disable=False,
            unit="series",
            desc="EDGAR event series",
            leave=True,
        )
        if tqdm is not None
        else None
    )
    try:
        for series_id in selected:
            effective_start = _next_start_date(args.start_date, last_seen_by_series.get(series_id))
            if effective_start > args.end_date:
                logger.info(
                    "series already up to date | series_id=%s last_timestamp_utc=%s effective_start=%s end_date=%s",
                    series_id,
                    last_seen_by_series.get(series_id),
                    effective_start,
                    args.end_date,
                )
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"series_id": series_id, "status": "up_to_date"})
                continue

            logger.info(
                "downloading event series | series_id=%s effective_start=%s end_date=%s",
                series_id,
                effective_start,
                args.end_date,
            )
            frame = collector.download_series(
                series_id,
                start_date=effective_start,
                end_date=args.end_date,
                frame_type=args.frame_type,
                show_progress=not args.no_progress,
            )
            downloaded_frames.append(frame)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"series_id": series_id, "rows": len(frame)})
    finally:
        if pbar is not None:
            pbar.close()

    combined = _concat_frames(existing_df, downloaded_frames)
    if combined is None:
        combined = collector.download_many(
            [],
            start_date=args.start_date,
            end_date=args.end_date,
            frame_type=args.frame_type,
            show_progress=False,
        )
    df = _dedupe_frame(combined)
    out = _replace_output_atomically(df, collector, out_path, args.partition_cols)
    logger.info(
        "event download finished | rows=%s series_count=%s downloaded_series=%s out=%s",
        len(df),
        len(selected),
        len(downloaded_frames),
        out,
    )
    print(f"Wrote {len(df)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
