from __future__ import annotations

from datetime import UTC, datetime
import logging
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from clients.gdelt_client import GDELTClient
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


class NewsCollector:
    def __init__(self, client: GDELTClient | None = None, *, store: ParquetStore | None = None) -> None:
        self._client = client or GDELTClient()
        self._store = store or ParquetStore()

    def search(
        self,
        query: str | None,
        start_date: datetime | str,
        end_date: datetime | str,
        *,
        max_records: int = 250,
        language: str | None = None,
        dedupe: bool = False,
        frame_type: FrameType | None = None,
    ) -> Any:
        return self.search_gdelt(
            query=query,
            start_date=start_date,
            end_date=end_date,
            max_records=max_records,
            language=language,
            dedupe=dedupe,
            frame_type=frame_type,
        )

    def search_gdelt(
        self,
        query: str | None = None,
        *,
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
        timespan: str | None = None,
        max_records: int = 250,
        page_size: int = 250,
        show_progress: bool = True,
        dedupe: bool = False,
        language: str | None = None,
        frame_type: FrameType | None = None,
    ) -> Any:
        """
        Query GDELT DOC API and return normalized rows:
          timestamp_utc, title, url, source, language, domain
        """

        q = str(query).strip() if query is not None else ""
        if not q:
            q = "the"

        target = int(max_records)
        if target <= 0:
            return self._to_frame([], frame_type=frame_type)

        tqdm = _resolve_tqdm(show_progress=show_progress)
        if show_progress and tqdm is None:
            logger.warning("show_progress=True but tqdm is unavailable in the active environment.")

        pbar = (
            tqdm(
                total=target,
                disable=False,
                unit="article",
                desc="Downloading GDELT news",
                leave=True,
            )
            if tqdm is not None
            else None
        )

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        pages = 0
        raw_seen = 0
        language_norm = self._normalize_text_for_key(language) if language is not None else None

        try:
            for payload in self._client.iter_doc_pages(
                query=q,
                start_date=start_date,
                end_date=end_date,
                timespan=timespan,
                max_records=target,
                page_size=page_size,
                sort="datedesc",
            ):
                pages += 1
                articles = self._client.extract_articles(payload)
                raw_seen += len(articles)

                before = len(rows)
                for article in articles:
                    row = self._normalize_gdelt_article(article)
                    if row is None:
                        continue

                    if language_norm is not None:
                        row_lang = self._normalize_text_for_key(row.get("language"))
                        if not self._language_match(row_lang, language_norm):
                            continue

                    if dedupe:
                        key = self._article_dedupe_key(row)
                        if key is not None:
                            if key in seen:
                                continue
                            seen.add(key)

                    rows.append(row)
                    if len(rows) >= target:
                        break

                if pbar is not None:
                    delta = len(rows) - before
                    if delta > 0:
                        pbar.update(min(delta, max(0, target - pbar.n)))
                    pbar.set_postfix({"pages": pages, "kept": len(rows), "raw": raw_seen})

                if len(rows) >= target:
                    break

            if pbar is not None:
                pbar.set_postfix({"pages": pages, "kept": len(rows), "raw": raw_seen})
        finally:
            if pbar is not None:
                pbar.close()

        df = self._to_frame(rows, frame_type=frame_type)
        return self._finalize(df, frame_type=frame_type, dedupe=dedupe)

    def save_to_parquet(self, df: Any, path: str, *, partition_cols: list[str] | None = None) -> str:
        out = self._store.save(df, path, partition_cols=partition_cols)
        return str(out)

    @staticmethod
    def _normalize_gdelt_article(article: dict[str, Any]) -> dict[str, Any] | None:
        dt = GDELTClient.parse_article_datetime(article)
        if dt is None:
            return None
        dt = dt.astimezone(UTC)

        title = NewsCollector._clean_text(article.get("title"))

        url = NewsCollector._clean_text(
            article.get("url")
            or article.get("sourceurl")
            or article.get("urlmobile")
            or article.get("urlMobile")
        )

        domain = NewsCollector._clean_text(article.get("domain"))
        if domain is None:
            domain = NewsCollector._domain_from_url(url)

        source = NewsCollector._clean_text(
            article.get("source")
            or article.get("sourcename")
            or article.get("sourceName")
            or article.get("sourcecountry")
            or article.get("sourceCountry")
        )
        if source is None:
            source = domain

        language = NewsCollector._clean_text(
            article.get("language")
            or article.get("lang")
            or article.get("sourceLanguage")
            or article.get("sourcelang")
        )

        return {
            "timestamp_utc": dt,
            "title": title,
            "url": url,
            "source": source,
            "language": language,
            "domain": domain,
        }

    @staticmethod
    def _clean_text(value: Any) -> str | None:
        if value is None:
            return None
        # pandas.NA / NaN handling without importing pandas globally
        try:
            if value != value:  # noqa: PLR0124
                return None
        except Exception:
            pass
        if str(value) in {"<NA>", "nan", "None"}:
            return None
        s = str(value).strip()
        return s if s else None

    @staticmethod
    def _domain_from_url(url: str | None) -> str | None:
        if url is None:
            return None
        try:
            parsed = urlparse(url)
        except Exception:
            return None
        host = (parsed.netloc or "").strip().lower()
        if not host:
            return None
        if host.startswith("www."):
            host = host[4:]
        return host or None

    @staticmethod
    def _normalize_text_for_key(value: Any) -> str | None:
        v = NewsCollector._clean_text(value)
        if v is None:
            return None
        return " ".join(v.lower().split())

    @staticmethod
    def _canonical_url_for_key(url: str | None) -> str | None:
        raw = NewsCollector._clean_text(url)
        if raw is None:
            return None
        try:
            p = urlparse(raw)
        except Exception:
            return None
        host = (p.netloc or "").strip().lower()
        if not host:
            return None
        if host.startswith("www."):
            host = host[4:]
        path = (p.path or "").strip() or "/"
        if path != "/" and path.endswith("/"):
            path = path[:-1]

        # Remove common tracking query params; keep deterministic sorted residual.
        drop_prefixes = ("utm_",)
        drop_exact = {"fbclid", "gclid", "igshid", "mc_cid", "mc_eid", "ref", "ref_src"}
        kept: list[tuple[str, str]] = []
        for k, v in parse_qsl(p.query or "", keep_blank_values=False):
            kl = k.strip().lower()
            if not kl:
                continue
            if kl in drop_exact or any(kl.startswith(pref) for pref in drop_prefixes):
                continue
            kept.append((kl, v))
        kept.sort(key=lambda x: (x[0], x[1]))
        query = urlencode(kept, doseq=True)
        canonical = urlunparse(("https", host, path, "", query, ""))
        return canonical

    @staticmethod
    def _article_dedupe_key(row: dict[str, Any]) -> str | None:
        canonical_url = NewsCollector._canonical_url_for_key(row.get("url"))
        if canonical_url:
            return f"url::{canonical_url}"

        title = NewsCollector._normalize_text_for_key(row.get("title"))
        domain = NewsCollector._normalize_text_for_key(row.get("domain"))
        source = NewsCollector._normalize_text_for_key(row.get("source"))

        if title and domain:
            return f"domain_title::{domain}::{title}"
        if title and source:
            return f"source_title::{source}::{title}"
        if title:
            return f"title::{title}"
        return None

    @staticmethod
    def _language_match(actual: str | None, expected: str) -> bool:
        a = (actual or "").strip().lower()
        e = expected.strip().lower()
        if not a or not e:
            return False
        if a == e:
            return True
        if e == "english" and a in {"en", "eng"}:
            return True
        if e in {"en", "eng"} and a == "english":
            return True
        return False

    @staticmethod
    def _to_frame(rows: list[dict[str, Any]], *, frame_type: FrameType | None) -> Any:
        frame = frame_type or _default_frame_type()

        if frame == "pandas":
            import pandas as pd

            cols = ["timestamp_utc", "title", "url", "source", "language", "domain"]
            if not rows:
                return pd.DataFrame(columns=cols)
            return pd.DataFrame(rows)

        if frame == "polars":
            import polars as pl

            if not rows:
                return pl.DataFrame(
                    schema=[
                        ("timestamp_utc", pl.Datetime(time_zone="UTC")),
                        ("title", pl.Utf8),
                        ("url", pl.Utf8),
                        ("source", pl.Utf8),
                        ("language", pl.Utf8),
                        ("domain", pl.Utf8),
                    ]
                )
            return pl.DataFrame(rows)

        raise ValueError(f"Unknown frame_type: {frame!r}")

    @staticmethod
    def _finalize(df: Any, *, frame_type: FrameType | None, dedupe: bool = True) -> Any:
        frame = frame_type or _default_frame_type()

        cols = ["timestamp_utc", "title", "url", "source", "language", "domain"]

        if frame == "pandas":
            import pandas as pd

            for col in cols:
                if col not in df.columns:
                    df[col] = None

            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp_utc"])

            for col in ("title", "url", "source", "language", "domain"):
                df[col] = df[col].astype("string")

            df = df.sort_values(["timestamp_utc", "url", "title"], ascending=[True, True, True], kind="stable")
            if dedupe:
                keys = []
                for row in df[["title", "url", "source", "domain"]].to_dict(orient="records"):
                    key = NewsCollector._article_dedupe_key(row)
                    keys.append(key)
                df["_dedupe_key"] = keys
                # Keep all rows without a reliable key; dedupe only keyed rows.
                keyed = df[df["_dedupe_key"].notna()].drop_duplicates(subset=["_dedupe_key"], keep="first")
                unkeyed = df[df["_dedupe_key"].isna()]
                df = (
                    pd.concat([keyed, unkeyed], ignore_index=True)
                    .sort_values(["timestamp_utc", "url", "title"], ascending=[True, True, True], kind="stable")
                    .drop(columns=["_dedupe_key"], errors="ignore")
                )
            df = df.reset_index(drop=True)
            return df[cols]

        if frame == "polars":
            import polars as pl

            for col in cols:
                if col not in df.columns:
                    df = df.with_columns(pl.lit(None).alias(col))

            df = df.with_columns(
                pl.col("timestamp_utc")
                .cast(pl.Datetime(time_zone="UTC"), strict=False)
                .alias("timestamp_utc")
            )

            for col in ("title", "url", "source", "language", "domain"):
                df = df.with_columns(pl.col(col).cast(pl.Utf8, strict=False).alias(col))

            df = df.filter(pl.col("timestamp_utc").is_not_null())
            df = df.sort(["timestamp_utc", "url", "title"], descending=[False, False, False], nulls_last=True)
            if dedupe:
                dedupe_keys = [
                    NewsCollector._article_dedupe_key(
                        {
                            "title": r.get("title"),
                            "url": r.get("url"),
                            "source": r.get("source"),
                            "domain": r.get("domain"),
                        }
                    )
                    for r in df.select(["title", "url", "source", "domain"]).to_dicts()
                ]
                df = df.with_columns(pl.Series("_dedupe_key", dedupe_keys))
                keyed = df.filter(pl.col("_dedupe_key").is_not_null()).unique(subset=["_dedupe_key"], keep="first")
                unkeyed = df.filter(pl.col("_dedupe_key").is_null())
                df = pl.concat([keyed, unkeyed], how="vertical").drop("_dedupe_key")
            df = df.sort(["timestamp_utc", "url", "title"], descending=[False, False, False], nulls_last=True)
            return df.select(cols)

        raise ValueError(f"Unknown frame_type: {frame!r}")
