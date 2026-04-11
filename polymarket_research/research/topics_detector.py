"""Topic-model comparison helpers for canonical Polymarket market text."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Iterable, Sequence
import warnings

from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from turftopic import FASTopic, S3
from turftopic.encoders.base import ExternalEncoder
from umap import UMAP



TEXT_MODE_ALIASES = {
    "question": "question",
    "full_description": "full_description",
    "description": "full_description",
    "combined": "question_plus_full_description",
    "both": "question_plus_full_description",
    "question_plus_full_description": "question_plus_full_description",
}


def _softmax(matrix: np.ndarray) -> np.ndarray:
    """Return a row-wise softmax for a score matrix."""
    shifted = matrix - matrix.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = exp.sum(axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return exp / denom


def _normalize_topic_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert a non-negative document-topic matrix into row-normalized scores."""
    clipped = np.clip(np.asarray(matrix, dtype=float), 0.0, None)
    row_sums = clipped.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return clipped / row_sums


def _join_text_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Join a subset of text columns into one normalized text field."""
    present = [column for column in columns if column in frame.columns]
    if not present:
        return pd.Series("", index=frame.index, dtype="string")

    parts = []
    for column in present:
        values = frame[column].fillna("").astype(str).str.strip()
        parts.append(values.where(values.ne(""), ""))

    out = parts[0]
    for values in parts[1:]:
        out = (out + " " + values).str.strip()
    return out.str.replace(r"\s+", " ", regex=True).astype("string")


def build_topic_input_frame(markets: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical topic-model input text columns to a market frame."""
    out = markets.copy()
    if "full_description" not in out.columns:
        out["full_description"] = _join_text_columns(
            out,
            ["description", "event_description", "resolution_source", "event_title"],
        )
    else:
        out["full_description"] = out["full_description"].fillna("").astype(str).str.strip()

    if "question" not in out.columns:
        out["question"] = ""
    out["question"] = out["question"].fillna("").astype(str).str.strip()

    out["full_description"] = out["full_description"].where(out["full_description"].ne(""), out["question"])
    out["question_plus_full_description"] = (
        out["question"].fillna("").astype(str).str.strip() + " " + out["full_description"].fillna("").astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    return out


def select_topic_text(markets: pd.DataFrame, text_mode: str = "full_description") -> pd.Series:
    """Select one text view from the prepared market frame."""
    prepared = build_topic_input_frame(markets)
    canonical_mode = TEXT_MODE_ALIASES.get(str(text_mode).strip().lower(), str(text_mode).strip().lower())
    if canonical_mode not in prepared.columns:
        raise ValueError(f"Unsupported text_mode={text_mode!r}.")
    return prepared[canonical_mode].fillna("").astype(str).str.strip()


class CachedSentenceTransformerEncoder(ExternalEncoder):
    """Small `turftopic` encoder wrapper with per-string embedding cache."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", *, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, sentences: Iterable[str]) -> np.ndarray:
        ordered = [str(sentence) for sentence in sentences]
        missing = [sentence for sentence in ordered if sentence not in self._cache]
        if missing:
            encoded = self.model.encode(missing, show_progress_bar=False)
            for sentence, vector in zip(missing, encoded, strict=False):
                self._cache[sentence] = np.asarray(vector, dtype=np.float32)
        return np.vstack([self._cache[sentence] for sentence in ordered])


@dataclass
class TopicModelResult:
    """Normalized outputs for one topic-model run."""

    model_name: str
    text_mode: str
    documents: pd.DataFrame
    topics: pd.DataFrame
    doc_topic_matrix: np.ndarray
    runtime_seconds: float
    fitted_model: object | None = None
    _projection_cache: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def representative_documents(self, top_n: int = 5) -> pd.DataFrame:
        """Return top-confidence documents per topic."""
        rows: list[pd.DataFrame] = []
        for topic_id in self.topics["topic_id"].tolist():
            subset = (
                self.documents.loc[self.documents["topic_id"] == topic_id]
                .sort_values("topic_confidence", ascending=False)
                .head(top_n)
            )
            rows.append(subset)
        if not rows:
            return self.documents.head(0).copy()
        return pd.concat(rows, ignore_index=True)

    def topic_summary(self) -> pd.DataFrame:
        """Return one row per topic with labels and sizes."""
        return self.topics.sort_values(["topic_size", "topic_id"], ascending=[False, True]).reset_index(drop=True)

    def project_2d(self, reducer: str = "umap", *, random_state: int = 0) -> pd.DataFrame:
        """Project document-topic scores into 2D for plotting."""
        reducer_key = str(reducer).strip().lower()
        if reducer_key in self._projection_cache:
            return self._projection_cache[reducer_key].copy()

        matrix = np.asarray(self.doc_topic_matrix, dtype=float)
        if matrix.ndim != 2 or len(matrix) != len(self.documents):
            raise ValueError("Document-topic matrix shape does not match documents.")

        if len(matrix) < 3:
            coords = np.column_stack([np.arange(len(matrix), dtype=float), np.zeros(len(matrix), dtype=float)])
        elif reducer_key == "umap":
            if UMAP is None:
                raise RuntimeError("UMAP is not installed in the active environment.")
            n_neighbors = min(15, max(2, len(matrix) - 1))
            reducer_model = UMAP(
                n_components=2,
                metric="cosine",
                random_state=random_state,
                n_neighbors=n_neighbors,
                init="random",
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"n_jobs value .* overridden .* random_state",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r"Spectral initialisation failed!.*",
                    category=UserWarning,
                )
                coords = reducer_model.fit_transform(matrix)
        elif reducer_key in {"tsne", "t-sne"}:
            perplexity = min(30, max(2, len(matrix) // 4))
            perplexity = min(perplexity, len(matrix) - 1)
            reducer_model = TSNE(n_components=2, random_state=random_state, init="pca", perplexity=perplexity)
            coords = reducer_model.fit_transform(matrix)
        else:
            raise ValueError(f"Unsupported reducer={reducer!r}.")

        projected = self.documents.copy()
        projected["x"] = coords[:, 0]
        projected["y"] = coords[:, 1]
        self._projection_cache[reducer_key] = projected
        return projected.copy()

    def plot_2d(
        self,
        reducer: str = "umap",
        *,
        random_state: int = 0,
        figsize: tuple[float, float] = (12.0, 9.0),
        alpha: float = 0.72,
    ):
        """Plot the document cloud with topic-label annotations."""
        projected = self.project_2d(reducer=reducer, random_state=random_state)
        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(
            data=projected,
            x="x",
            y="y",
            hue="topic_label",
            size="topic_confidence",
            palette="tab20",
            alpha=alpha,
            linewidth=0.0,
            ax=ax,
            legend=False,
        )
        centers = projected.groupby("topic_id", as_index=False)[["x", "y"]].mean()
        label_lookup = self.topics.set_index("topic_id")["topic_label"].to_dict()
        for row in centers.itertuples(index=False):
            ax.text(
                float(row.x),
                float(row.y),
                str(label_lookup.get(int(row.topic_id), f"topic_{int(row.topic_id)}")),
                fontsize=9,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "black"},
            )
        ax.set_title(f"{self.model_name} on {self.text_mode} ({reducer.upper()})")
        ax.set_xlabel("component_1")
        ax.set_ylabel("component_2")
        fig.tight_layout()
        return fig, ax, projected

    def plot_2d_with_topic_map(
        self,
        reducer: str = "umap",
        *,
        random_state: int = 0,
        figsize: tuple[float, float] = (15.0, 10.0),
        alpha: float = 0.62,
        point_size_range: tuple[float, float] = (18.0, 90.0),
        table_position: str = "bottom",
        show_table: bool = True,
        legend_mode: str = "compact",
        max_label_words: int = 3,
        legend_max_words: int = 4,
    ):
        """Plot one model with a compact topic-id legend and a topic mapping table."""
        projected = self.project_2d(reducer=reducer, random_state=random_state)
        topics = self.topic_summary().copy()
        topics["topic_short_label"] = topics["topic_terms"].map(
            lambda text: ", ".join(str(text).split(", ")[:max(1, int(max_label_words))])
        )
        topics["legend_label"] = topics.apply(
            lambda row: (
                f"T{int(row['topic_id'])} - [{', '.join(str(row['topic_terms']).split(', ')[:max(1, int(legend_max_words))])}] "
                f"(size: {int(row['topic_size'])})"
            ),
            axis=1,
        )

        topic_ids = topics["topic_id"].tolist()
        palette_colors = sns.color_palette("tab20", n_colors=max(20, len(topic_ids)))
        color_lookup = {
            int(topic_id): palette_colors[idx % len(palette_colors)]
            for idx, topic_id in enumerate(sorted(topic_ids))
        }

        projected["color"] = projected["topic_id"].map(color_lookup)
        projected["point_size"] = np.interp(
            projected["topic_confidence"].fillna(projected["topic_confidence"].mean()).to_numpy(),
            (
                float(projected["topic_confidence"].min()),
                float(projected["topic_confidence"].max()) if len(projected) else 1.0,
            ),
            point_size_range,
        )
        if projected["topic_confidence"].min() == projected["topic_confidence"].max():
            projected["point_size"] = float(sum(point_size_range) / 2.0)

        table_position_key = str(table_position).strip().lower()
        if not show_table:
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(1, 2, figure=fig, width_ratios=[0.34, 0.66])
            legend_ax = fig.add_subplot(gs[0, 0])
            scatter_ax = fig.add_subplot(gs[0, 1])
            table_ax = None
        elif table_position_key == "right":
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(1, 3, figure=fig, width_ratios=[0.18, 0.54, 0.28])
            legend_ax = fig.add_subplot(gs[0, 0])
            scatter_ax = fig.add_subplot(gs[0, 1])
            table_ax = fig.add_subplot(gs[0, 2])
        else:
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(2, 2, figure=fig, height_ratios=[0.72, 0.28], width_ratios=[0.18, 0.82])
            legend_ax = fig.add_subplot(gs[0, 0])
            scatter_ax = fig.add_subplot(gs[0, 1])
            table_ax = fig.add_subplot(gs[1, :])

        scatter_ax.scatter(
            projected["x"],
            projected["y"],
            c=projected["color"].tolist(),
            s=projected["point_size"].tolist(),
            alpha=alpha,
            linewidths=0.0,
        )
        scatter_ax.set_title(f"{self.model_name} on {self.text_mode} ({reducer.upper()})")
        scatter_ax.set_xlabel("component_1")
        scatter_ax.set_ylabel("component_2")

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                label=(
                    f"T{int(topic_id)}"
                    if str(legend_mode).strip().lower() == "compact"
                    else str(topics.loc[topics["topic_id"].eq(int(topic_id)), "legend_label"].iloc[0])
                ),
                markerfacecolor=color_lookup[int(topic_id)],
                markersize=8,
            )
            for topic_id in sorted(topic_ids)
        ]
        legend_ax.legend(
            handles=handles,
            loc="upper left",
            frameon=False,
            title="Topic Id" if str(legend_mode).strip().lower() == "compact" else "Topics",
            fontsize=9,
            title_fontsize=10,
            handletextpad=0.8,
            labelspacing=0.8,
            borderaxespad=0.0,
        )
        legend_ax.axis("off")

        if table_ax is not None:
            table_rows = [
                [
                    f"T{int(row.topic_id)}",
                    str(row.topic_short_label),
                    int(row.topic_size),
                ]
                for row in topics.itertuples(index=False)
            ]
            table = table_ax.table(
                cellText=table_rows,
                colLabels=["topic_id", "signature_words", "size"],
                loc="center",
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.2)
            table_ax.axis("off")

        fig.tight_layout()
        return fig, {"legend_ax": legend_ax, "scatter_ax": scatter_ax, "table_ax": table_ax}, projected


@dataclass
class BaseTopicsDetector:
    """Common interface for topic-model wrappers."""

    n_topics: int = 12
    text_mode: str = "full_description"
    random_state: int = 0
    min_df: int = 5
    max_df: float = 0.5
    max_features: int = 5_000
    top_terms: int = 8

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    def fit_transform(self, markets: pd.DataFrame) -> TopicModelResult:
        raise NotImplementedError

    def _prepare_markets(self, markets: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        prepared = build_topic_input_frame(markets)
        text = select_topic_text(prepared, self.text_mode)
        mask = text.str.strip().ne("")
        prepared = prepared.loc[mask].copy().reset_index(drop=True)
        text = text.loc[mask].reset_index(drop=True)
        return prepared, text

    def _effective_topic_count(self, n_docs: int) -> int:
        return max(2, min(int(self.n_topics), max(2, n_docs // 3), n_docs - 1))

    def _topic_rows_from_pairs(self, topic_pairs, assignments: np.ndarray) -> pd.DataFrame:
        topic_sizes = pd.Series(assignments).value_counts().sort_index()
        rows = []
        for topic_id, weighted_terms in topic_pairs:
            terms = [str(term) for term, _weight in weighted_terms[: self.top_terms]]
            rows.append(
                {
                    "topic_id": int(topic_id),
                    "topic_terms": ", ".join(terms),
                    "topic_label": f"topic_{int(topic_id)}: {', '.join(terms[:4])}",
                    "topic_size": int(topic_sizes.get(int(topic_id), 0)),
                }
            )
        return pd.DataFrame(rows).sort_values("topic_id").reset_index(drop=True)

    def _document_frame(
        self,
        prepared_markets: pd.DataFrame,
        text: pd.Series,
        doc_topic_matrix: np.ndarray,
        topics: pd.DataFrame,
    ) -> pd.DataFrame:
        topic_id = doc_topic_matrix.argmax(axis=1).astype(int)
        confidence = doc_topic_matrix.max(axis=1).astype(float)
        out = prepared_markets.copy()
        out["text_used"] = text
        out["topic_id"] = topic_id
        out["topic_confidence"] = confidence
        topic_label = topics.set_index("topic_id")["topic_label"].to_dict()
        topic_terms = topics.set_index("topic_id")["topic_terms"].to_dict()
        out["topic_label"] = out["topic_id"].map(topic_label)
        out["topic_terms"] = out["topic_id"].map(topic_terms)
        return out


@dataclass
class S3TopicsDetector(BaseTopicsDetector):
    """Semantic Signal Separation wrapper."""

    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_device: str = "cpu"
    feature_importance: str = "combined"
    max_iter: int = 200
    ngram_range: tuple[int, int] = (1, 2)

    @property
    def model_name(self) -> str:
        return "S3"

    def fit_transform(self, markets: pd.DataFrame) -> TopicModelResult:
        prepared_markets, text = self._prepare_markets(markets)
        if len(prepared_markets) < 3:
            raise ValueError("Need at least 3 non-empty documents for topic modeling.")

        n_topics = self._effective_topic_count(len(prepared_markets))
        encoder = CachedSentenceTransformerEncoder(self.encoder_name, device=self.encoder_device)
        vectorizer = CountVectorizer(
            stop_words="english",
            min_df=1 if len(prepared_markets) < self.min_df else self.min_df,
            max_df=1.0 if len(prepared_markets) < 10 else self.max_df,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )
        model = S3(
            n_components=n_topics,
            encoder=encoder,
            vectorizer=vectorizer,
            feature_importance=self.feature_importance,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

        started = time.perf_counter()
        doc_topic = _normalize_topic_matrix(model.fit_transform(text.tolist()))
        runtime_seconds = time.perf_counter() - started
        documents = self._document_frame(
            prepared_markets,
            text,
            doc_topic,
            topics=self._topic_rows_from_pairs(model.get_topics(), doc_topic.argmax(axis=1)),
        )
        topics = self._topic_rows_from_pairs(model.get_topics(), documents["topic_id"].to_numpy())
        documents["topic_label"] = documents["topic_id"].map(topics.set_index("topic_id")["topic_label"])
        documents["topic_terms"] = documents["topic_id"].map(topics.set_index("topic_id")["topic_terms"])
        return TopicModelResult(
            model_name=self.model_name,
            text_mode=self.text_mode,
            documents=documents,
            topics=topics,
            doc_topic_matrix=doc_topic,
            runtime_seconds=runtime_seconds,
            fitted_model=model,
        )


@dataclass
class FASTopicDetector(BaseTopicsDetector):
    """FASTopic wrapper."""

    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_device: str = "cpu"
    n_epochs: int = 100
    learning_rate: float = 0.002
    batch_size: int | None = None
    theta_temp: float = 1.0
    dt_alpha: float = 3.0
    tw_alpha: float = 2.0
    ngram_range: tuple[int, int] = (1, 2)

    @property
    def model_name(self) -> str:
        return "FASTopic"

    def fit_transform(self, markets: pd.DataFrame) -> TopicModelResult:
        prepared_markets, text = self._prepare_markets(markets)
        if len(prepared_markets) < 3:
            raise ValueError("Need at least 3 non-empty documents for topic modeling.")

        n_topics = self._effective_topic_count(len(prepared_markets))
        encoder = CachedSentenceTransformerEncoder(self.encoder_name, device=self.encoder_device)
        vectorizer = CountVectorizer(
            stop_words="english",
            min_df=1 if len(prepared_markets) < self.min_df else self.min_df,
            max_df=1.0 if len(prepared_markets) < 10 else self.max_df,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )
        model = FASTopic(
            n_components=n_topics,
            encoder=encoder,
            vectorizer=vectorizer,
            random_state=self.random_state,
            batch_size=self.batch_size,
            DT_alpha=self.dt_alpha,
            TW_alpha=self.tw_alpha,
            theta_temp=self.theta_temp,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            device=self.encoder_device,
        )

        started = time.perf_counter()
        doc_topic = _normalize_topic_matrix(model.fit_transform(text.tolist()))
        runtime_seconds = time.perf_counter() - started
        documents = self._document_frame(
            prepared_markets,
            text,
            doc_topic,
            topics=self._topic_rows_from_pairs(model.get_topics(), doc_topic.argmax(axis=1)),
        )
        topics = self._topic_rows_from_pairs(model.get_topics(), documents["topic_id"].to_numpy())
        documents["topic_label"] = documents["topic_id"].map(topics.set_index("topic_id")["topic_label"])
        documents["topic_terms"] = documents["topic_id"].map(topics.set_index("topic_id")["topic_terms"])
        return TopicModelResult(
            model_name=self.model_name,
            text_mode=self.text_mode,
            documents=documents,
            topics=topics,
            doc_topic_matrix=doc_topic,
            runtime_seconds=runtime_seconds,
            fitted_model=model,
        )


@dataclass
class TFIDFTopicBaseline(BaseTopicsDetector):
    """Simple TF-IDF + KMeans baseline with signature words from cluster centroids."""

    ngram_range: tuple[int, int] = (1, 2)
    n_init: int | str = "auto"

    @property
    def model_name(self) -> str:
        return "TFIDF"

    def fit_transform(self, markets: pd.DataFrame) -> TopicModelResult:
        prepared_markets, text = self._prepare_markets(markets)
        if len(prepared_markets) < 3:
            raise ValueError("Need at least 3 non-empty documents for topic modeling.")

        n_topics = self._effective_topic_count(len(prepared_markets))
        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1 if len(prepared_markets) < self.min_df else self.min_df,
            max_df=1.0 if len(prepared_markets) < 10 else self.max_df,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )

        started = time.perf_counter()
        tfidf = vectorizer.fit_transform(text.tolist())
        kmeans = KMeans(n_clusters=n_topics, random_state=self.random_state, n_init=self.n_init)
        assignments = kmeans.fit_predict(tfidf)
        similarity = cosine_similarity(tfidf, kmeans.cluster_centers_)
        doc_topic = _softmax(similarity)
        runtime_seconds = time.perf_counter() - started

        feature_names = vectorizer.get_feature_names_out()
        topic_rows = []
        topic_sizes = pd.Series(assignments).value_counts().sort_index()
        for topic_id, center in enumerate(kmeans.cluster_centers_):
            top_indices = np.argsort(center)[::-1][: self.top_terms]
            terms = [str(feature_names[idx]) for idx in top_indices]
            topic_rows.append(
                {
                    "topic_id": int(topic_id),
                    "topic_terms": ", ".join(terms),
                    "topic_label": f"topic_{int(topic_id)}: {', '.join(terms[:4])}",
                    "topic_size": int(topic_sizes.get(topic_id, 0)),
                }
            )
        topics = pd.DataFrame(topic_rows).sort_values("topic_id").reset_index(drop=True)
        documents = self._document_frame(prepared_markets, text, doc_topic, topics=topics)
        return TopicModelResult(
            model_name=self.model_name,
            text_mode=self.text_mode,
            documents=documents,
            topics=topics,
            doc_topic_matrix=doc_topic,
            runtime_seconds=runtime_seconds,
            fitted_model={"vectorizer": vectorizer, "kmeans": kmeans},
        )


def compare_topic_models(
    markets: pd.DataFrame,
    detectors: Sequence[BaseTopicsDetector],
) -> tuple[dict[str, TopicModelResult], pd.DataFrame]:
    """Run multiple detectors on the same market frame and return their results plus a summary table."""
    results: dict[str, TopicModelResult] = {}
    rows: list[dict[str, object]] = []
    for detector in detectors:
        result = detector.fit_transform(markets)
        results[result.model_name] = result
        rows.append(
            {
                "model_name": result.model_name,
                "text_mode": result.text_mode,
                "n_documents": int(len(result.documents)),
                "n_topics": int(len(result.topics)),
                "runtime_seconds": float(result.runtime_seconds),
                "mean_topic_confidence": float(result.documents["topic_confidence"].mean()),
            }
        )
    return results, pd.DataFrame(rows).sort_values("runtime_seconds").reset_index(drop=True)


def plot_topic_model_grid(
    results: Sequence[TopicModelResult],
    *,
    reducer: str = "umap",
    random_state: int = 0,
    figsize_per_plot: tuple[float, float] = (7.0, 5.5),
):
    """Plot multiple topic-model clouds side by side."""
    n_results = len(results)
    fig, axes = plt.subplots(1, n_results, figsize=(figsize_per_plot[0] * n_results, figsize_per_plot[1]))
    if n_results == 1:
        axes = [axes]
    for ax, result in zip(axes, results, strict=False):
        projected = result.project_2d(reducer=reducer, random_state=random_state)
        sns.scatterplot(
            data=projected,
            x="x",
            y="y",
            hue="topic_label",
            size="topic_confidence",
            palette="tab20",
            alpha=0.72,
            linewidth=0.0,
            ax=ax,
            legend=False,
        )
        centers = projected.groupby("topic_id", as_index=False)[["x", "y"]].mean()
        label_lookup = result.topics.set_index("topic_id")["topic_label"].to_dict()
        for row in centers.itertuples(index=False):
            ax.text(
                float(row.x),
                float(row.y),
                str(label_lookup.get(int(row.topic_id), f"topic_{int(row.topic_id)}")),
                fontsize=8,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "black"},
            )
        ax.set_title(f"{result.model_name} ({reducer.upper()})")
        ax.set_xlabel("component_1")
        ax.set_ylabel("component_2")
    fig.tight_layout()
    return fig, axes


__all__ = [
    "FASTopicDetector",
    "S3TopicsDetector",
    "TFIDFTopicBaseline",
    "TopicModelResult",
    "build_topic_input_frame",
    "compare_topic_models",
    "plot_topic_model_grid",
    "select_topic_text",
]
