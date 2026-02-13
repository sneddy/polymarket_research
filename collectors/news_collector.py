from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from clients.news_client import NewsClient
from storage.parquet_store import ParquetStore


FrameType = Literal["pandas", "polars"]


class NewsCollector:
    def __init__(self, client: NewsClient, *, store: ParquetStore | None = None) -> None:
        self._client = client
        self._store = store or ParquetStore()

    def search(
        self,
        query: str,
        start_date: datetime | str,
        end_date: datetime | str,
        *,
        max_records: int = 250,
        frame_type: FrameType | None = None,
    ) -> Any:
        return self._client.search_news(query, start_date, end_date, max_records=max_records, frame_type=frame_type)

    def save_to_parquet(self, df: Any, path: str, *, partition_cols: list[str] | None = None) -> str:
        out = self._store.save(df, path, partition_cols=partition_cols)
        return str(out)

