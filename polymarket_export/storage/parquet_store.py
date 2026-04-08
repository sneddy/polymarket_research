from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence


FrameType = Literal["pandas", "polars"]


def _default_frame_type() -> FrameType:
    try:
        import pandas as _  # noqa: F401

        return "pandas"
    except Exception:
        return "polars"


def _is_pandas_df(obj: Any) -> bool:
    try:
        import pandas as pd  # type: ignore

        return isinstance(obj, pd.DataFrame)
    except Exception:
        mod = getattr(obj.__class__, "__module__", "")
        return mod == "pandas" or str(mod).startswith("pandas.")


def _is_polars_df(obj: Any) -> bool:
    try:
        import polars as pl  # type: ignore

        return isinstance(obj, pl.DataFrame)
    except Exception:
        mod = getattr(obj.__class__, "__module__", "")
        return mod == "polars" or str(mod).startswith("polars.")


@dataclass(frozen=True)
class ParquetStore:
    """Small parquet helper to keep I/O separate from collection logic."""

    frame_type: FrameType = "pandas"

    def save(self, df: Any, path: str | Path, partition_cols: Sequence[str] | None = None) -> Path:
        out_path = Path(path)
        if partition_cols:
            out_path.mkdir(parents=True, exist_ok=True)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)

        if _is_pandas_df(df):
            return self._save_pandas(df, out_path, partition_cols)
        if _is_polars_df(df):
            return self._save_polars(df, out_path, partition_cols)

        raise TypeError(f"Unsupported dataframe type: {type(df)!r}")

    def load(self, path: str | Path, frame_type: FrameType | None = None) -> Any:
        frame = frame_type or self.frame_type or _default_frame_type()
        in_path = Path(path)

        if frame == "pandas":
            import pandas as pd

            return pd.read_parquet(in_path)

        if frame == "polars":
            import polars as pl

            if in_path.is_dir():
                # Read partitioned dataset / directory of parquet files.
                return pl.scan_parquet(str(in_path / "**" / "*.parquet")).collect()
            return pl.read_parquet(in_path)

        raise ValueError(f"Unknown frame_type: {frame!r}")

    @staticmethod
    def _save_pandas(df: Any, out_path: Path, partition_cols: Sequence[str] | None) -> Path:
        if partition_cols:
            df.to_parquet(out_path, partition_cols=list(partition_cols), index=False)
            return out_path

        df.to_parquet(out_path, index=False)
        return out_path

    @staticmethod
    def _save_polars(df: Any, out_path: Path, partition_cols: Sequence[str] | None) -> Path:
        if partition_cols:
            raise NotImplementedError("polars partitioned parquet writing is not implemented here.")

        df.write_parquet(out_path)
        return out_path
