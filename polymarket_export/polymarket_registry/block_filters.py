"""Block-level market selection filters shared by notebooks and scripts."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import logging
import re

import pandas as pd


logger = logging.getLogger(__name__)
DEFAULT_MIN_RESIDUAL_VOLUME = 20_000.0


@dataclass
class BlockFilter:
    """Base class for block-level exclusion filters."""

    name: str
    category: str | None = None

    def normalize_text(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Return a normalized lowercase text series for a column."""
        if col not in df.columns:
            return pd.Series("", index=df.index)
        return df[col].fillna("").astype(str).str.lower().str.strip()

    def unassigned_mask(self, df: pd.DataFrame) -> pd.Series:
        """Return the rows that do not yet have an assigned category."""
        if "category" not in df.columns:
            return pd.Series(True, index=df.index)
        return df["category"].isna() | df["category"].astype(str).str.strip().eq("")

    def build_mask(self, df: pd.DataFrame) -> pd.Series:
        """Build the raw filter mask for the block."""
        raise NotImplementedError

    def candidate_mask(self, df: pd.DataFrame) -> pd.Series:
        """Restrict the block mask to rows that are still unassigned."""
        return self.unassigned_mask(df) & self.build_mask(df)

    def apply_to_remaining(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign this block's category to all currently matching rows."""
        if self.category is None:
            raise ValueError(f"{self.name} does not define a category")
        out = df.copy()
        out.loc[self.candidate_mask(out), "category"] = self.category
        return out

    def summarize_categories(self, df: pd.DataFrame) -> pd.Series:
        """Summarize the current category distribution."""
        return df["category"].fillna("unknown").value_counts()

    def show_flagged_series(self, df: pd.DataFrame, limit: int = 25) -> pd.Series:
        """Show the top event series flagged by this block."""
        series_slug = self.normalize_text(df, "event_series_slug").replace("", "<NA>")
        return series_slug.loc[self.candidate_mask(df)].value_counts().head(limit)

    def preview_flagged_rows(self, df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
        """Preview representative rows flagged by this block."""
        columns = [
            column
            for column in ["market_id", "market_slug", "event_slug", "event_series_slug", "event_title", "question", "volume_num"]
            if column in df.columns
        ]
        return df.loc[self.candidate_mask(df), columns].head(limit)


def initialize_work_df(df: pd.DataFrame) -> pd.DataFrame:
    """Create a mutable work dataframe with an empty category column."""
    work_df = df.copy()
    work_df["category"] = None
    return work_df


class UltraShortRecurringTemplateFilter(BlockFilter):
    """Exclude ultra-short recurring template markets."""

    def __init__(self) -> None:
        """Initialize the recurring-template filter."""
        super().__init__(name="ultra_short_recurring_templates", category="short_recurrence")

    def show_recurrence_distribution(self, df: pd.DataFrame) -> pd.Series:
        """Show the recurrence distribution of the market universe."""
        return df["event_recurrence"].value_counts(dropna=False)

    def show_top_series_for_recurrence(self, df: pd.DataFrame, recurrence: str, limit: int = 20) -> pd.Series:
        """Show the most common series for a given recurrence bucket."""
        mask = df["event_recurrence"].astype(str).eq(recurrence)
        return df.loc[mask, "event_series_slug"].fillna("<NA>").value_counts().head(limit)

    def detect_intraday_recurrence(self, df: pd.DataFrame) -> pd.Series:
        """Detect markets with 5m, 15m, or hourly recurrence."""
        return df["event_recurrence"].isin(["5m", "15m", "hourly"])

    def detect_up_down_template_markets(self, df: pd.DataFrame) -> pd.Series:
        """Detect mechanical Up/Down template markets."""
        return df["outcomes"].astype(str).str.strip().eq('["Up", "Down"]')

    def build_mask(self, df: pd.DataFrame) -> pd.Series:
        """Combine recurring-template substeps into one mask."""
        return self.detect_intraday_recurrence(df) | self.detect_up_down_template_markets(df)


class FinancialPriceDerivedFilter(BlockFilter):
    """Exclude price-derived and benchmark-derived financial markets."""

    def __init__(self) -> None:
        """Initialize the financial filter."""
        super().__init__(name="financial_price_derived", category="quant_price_structures")

    def series_slug(self, df: pd.DataFrame) -> pd.Series:
        """Return normalized event series slugs."""
        return self.normalize_text(df, "event_series_slug")

    def detect_structured_price_constructions(self, df: pd.DataFrame) -> pd.Series:
        """Detect structured price-construction market families."""
        series_slug = self.series_slug(df)
        return series_slug.str.contains(
            r"(?:multi-strikes|neg-risk|hit-price|weekly-brackets|monthly-prices)",
            regex=True,
            na=False,
        )

    def detect_benchmark_target_markets(self, df: pd.DataFrame) -> pd.Series:
        """Detect benchmark-target and target-price market families."""
        series_slug = self.series_slug(df)
        return series_slug.str.contains(
            r"(?:eth-weeklies|btc-weeklies|sol-weeklies|doge-monthly|mstr-weeklies|"
            r"ethereum-etf-flows-daily|bitcoin-etf-flows-daily|"
            r"crude-oil-cl-hit|will-silver-si-hit|what-will-gold-gc-hit|"
            r"mag-7-weekly|largest-company|second-largest-company|ipo-closing-market-cap)",
            regex=True,
            na=False,
        )

    def build_mask(self, df: pd.DataFrame) -> pd.Series:
        """Combine financial substeps into one mask."""
        return self.detect_structured_price_constructions(df) | self.detect_benchmark_target_markets(df)


class AttentionSpeechMetricsFilter(BlockFilter):
    """Exclude attention, speech, and mention-count markets."""

    def __init__(self) -> None:
        """Initialize the attention-and-speech filter."""
        super().__init__(name="attention_speech_metrics", category="attention_social_metrics")

    def series_slug(self, df: pd.DataFrame) -> pd.Series:
        """Return normalized event series slugs."""
        return self.normalize_text(df, "event_series_slug")

    def detect_attention_and_mention_markets(self, df: pd.DataFrame) -> pd.Series:
        """Detect attention, speech, and mention-count families."""
        series_slug = self.series_slug(df)
        return series_slug.str.contains(
            r"(?:tweets?|mentions?|truth-social|pmqs|mrbeast-views|trump-truths|trump-post-weekly|trump-talk-monthly|all-in-podcast|rogan-mentions|andrew-tate-tweets)",
            regex=True,
            na=False,
        )

    def build_mask(self, df: pd.DataFrame) -> pd.Series:
        """Build the attention-and-speech exclusion mask."""
        return self.detect_attention_and_mention_markets(df)


class WeatherMarketsFilter(BlockFilter):
    """Exclude weather market families."""

    def __init__(self) -> None:
        """Initialize the weather filter."""
        super().__init__(name="weather_markets", category="weather")

    def detect_weather_markets(self, df: pd.DataFrame) -> pd.Series:
        """Detect markets with weather-like series slugs."""
        series_slug = self.normalize_text(df, "event_series_slug")
        return series_slug.str.contains("weather", regex=False, na=False)

    def build_mask(self, df: pd.DataFrame) -> pd.Series:
        """Build the weather exclusion mask."""
        return self.detect_weather_markets(df)


class SportsAndEsportsFilter(BlockFilter):
    """Exclude sports and esports markets."""

    CYBER_RE = (
        r"(?:counter-strike|cs2|csgo|dota|league-of-legends|valorant|honor-of-kings|overwatch|"
        r"starcraft-2|mobile-legends-bang-bang|call-of-duty|esports|lck|lpl)"
    )

    SPORT_RE = (
        r"(?:nba|nfl|mlb|nhl|ncaa|cbb|cwbb|cfb|atp|wta|efl-championship|fa-cup|ucl-|uel-|efl-|"
        r"europa-conference-league|coupe-de-france|khl|ahl|shl|dehl|cehl|"
        r"japan-j2-league|japan-j-league|serie-b|ligue-2|primeira-liga|saudi-professional-league|"
        r"fifa-friendly|primera-a|primera-division|primera-divisin-argentina|"
        r"scottish-premiership|ukraine-premier-liha|womens-t20-world-cup-qualifier|"
        r"liga-1|egypt-1|romania-1|czechia-1|"
        r"(?:mls|la-liga|ligue-1|bundesliga|premier-league|serie-a)(?:-\d{4})?|"
        r"(?:mex|tur|ere)-\d{4}|cricket|soccer|basketball)"
    )

    def __init__(self) -> None:
        """Initialize the sports-and-esports filter."""
        super().__init__(name="sports_and_esports")

    def parse_outcomes(self, value) -> list[str] | None:
        """Parse outcome labels stored as list-like payloads."""
        if isinstance(value, list):
            return [str(x).strip().lower() for x in value]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                return None
            if isinstance(parsed, list):
                return [str(x).strip().lower() for x in parsed]
        return None

    def build_slug_signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Build combined slug signals from series and event slugs."""
        series_slug = self.normalize_text(df, "event_series_slug")
        event_slug = self.normalize_text(df, "event_slug")
        primary_slug = series_slug.where(series_slug.ne(""), event_slug)
        slug_blob = (series_slug + " " + event_slug).str.strip()
        return primary_slug, event_slug, slug_blob

    def build_text_blob(self, df: pd.DataFrame) -> pd.Series:
        """Build a normalized text blob from slug, title, and question fields."""
        return (
            self.normalize_text(df, "market_slug")
            + " "
            + self.normalize_text(df, "event_title")
            + " "
            + self.normalize_text(df, "question")
        )

    def has_exact_slug_token(self, series_slug: pd.Series, token: str) -> pd.Series:
        """Detect an exact hyphen-delimited token inside a slug series."""
        normalized_token = str(token).strip().lower()
        return series_slug.str.contains(rf"(?:^|-){re.escape(normalized_token)}(?:-|$)", regex=True, na=False)

    def detect_cybersport_by_slug_or_text(self, df: pd.DataFrame) -> pd.Series:
        """Detect esports markets from slugs and free text."""
        primary_slug, event_slug, slug_blob = self.build_slug_signals(df)
        text_blob = self.build_text_blob(df)
        return (
            primary_slug.str.contains(self.CYBER_RE, regex=True, na=False)
            | event_slug.str.contains(self.CYBER_RE, regex=True, na=False)
            | slug_blob.str.contains(self.CYBER_RE, regex=True, na=False)
            | text_blob.str.contains(self.CYBER_RE, regex=True, na=False)
        )

    def detect_cybersport_best_of_formats(self, df: pd.DataFrame) -> pd.Series:
        """Detect esports markets through best-of formatting fields."""
        event_score = self.normalize_text(df, "event_score")
        event_period = self.normalize_text(df, "event_period")
        return event_score.str.contains(r"\bbo\d+\b", regex=True, na=False) | event_period.str.contains(
            r"\bbo\d+\b",
            regex=True,
            na=False,
        )

    def detect_sport_by_slug_or_text(self, df: pd.DataFrame) -> pd.Series:
        """Detect sports markets from slugs and free text."""
        primary_slug, event_slug, slug_blob = self.build_slug_signals(df)
        text_blob = self.build_text_blob(df)
        return (
            primary_slug.str.contains(self.SPORT_RE, regex=True, na=False)
            | event_slug.str.contains(self.SPORT_RE, regex=True, na=False)
            | slug_blob.str.contains(self.SPORT_RE, regex=True, na=False)
            | text_blob.str.contains(self.SPORT_RE, regex=True, na=False)
        )

    def detect_exact_competition_tokens(self, df: pd.DataFrame) -> pd.Series:
        """Detect sports through exact competition tokens in slugs."""
        primary_slug, event_slug, _ = self.build_slug_signals(df)
        return (
            self.has_exact_slug_token(primary_slug, "uel")
            | self.has_exact_slug_token(primary_slug, "ucl")
            | self.has_exact_slug_token(primary_slug, "efl")
            | self.has_exact_slug_token(event_slug, "uel")
            | self.has_exact_slug_token(event_slug, "ucl")
            | self.has_exact_slug_token(event_slug, "efl")
        )

    def detect_sport_by_score_fields(self, df: pd.DataFrame) -> pd.Series:
        """Detect sports through non-empty score fields."""
        return self.normalize_text(df, "event_score").ne("")

    def detect_sport_by_outcome_schema(self, df: pd.DataFrame) -> pd.Series:
        """Detect sports through typical over-under outcome schemas."""
        if "outcomes" not in df.columns:
            return pd.Series(False, index=df.index)
        outcomes_norm = df["outcomes"].map(self.parse_outcomes)
        return outcomes_norm.map(
            lambda value: isinstance(value, list)
            and value in (["over", "under"], ["odd", "even"], ["favorite", "underdog"])
        )

    def cybersport_mask(self, df: pd.DataFrame) -> pd.Series:
        """Build the esports-only exclusion mask."""
        return self.detect_cybersport_by_slug_or_text(df) | self.detect_cybersport_best_of_formats(df)

    def sport_mask(self, df: pd.DataFrame) -> pd.Series:
        """Build the sports-only exclusion mask."""
        return (
            self.detect_sport_by_slug_or_text(df)
            | self.detect_exact_competition_tokens(df)
            | self.detect_sport_by_score_fields(df)
            | self.detect_sport_by_outcome_schema(df)
        )

    def build_mask(self, df: pd.DataFrame) -> pd.Series:
        """Build the combined sports-and-esports exclusion mask."""
        return self.cybersport_mask(df) | self.sport_mask(df)

    def show_flagged_split(self, df: pd.DataFrame) -> pd.Series:
        """Show the split between esports and sport matches."""
        unassigned = self.unassigned_mask(df)
        cyber_mask = self.cybersport_mask(df)
        return pd.Series(
            {
                "cybersport": int((unassigned & cyber_mask).sum()),
                "sport": int((unassigned & self.sport_mask(df) & ~cyber_mask).sum()),
            }
        )

    def apply_to_remaining(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign esports first, then sports, to remaining rows."""
        out = df.copy()
        missing_mask = self.unassigned_mask(out)
        cyber_mask = self.cybersport_mask(out)
        out.loc[missing_mask & cyber_mask, "category"] = "cybersport"

        missing_mask = self.unassigned_mask(out)
        sport_mask = self.sport_mask(out)
        out.loc[missing_mask & sport_mask, "category"] = "sport"
        return out


class FinalVolumeScreen(BlockFilter):
    """Exclude low-volume residual markets at the end of the pipeline."""

    def __init__(self, min_volume: float) -> None:
        """Initialize the final residual volume screen."""
        super().__init__(name="final_volume_screen", category="low_volume")
        self.min_volume = float(min_volume)

    def normalized_volume(self, df: pd.DataFrame) -> pd.Series:
        """Return a numeric volume series with missing values filled."""
        return pd.to_numeric(df.get("volume_num"), errors="coerce").fillna(0.0)

    def detect_low_volume_remaining_markets(self, df: pd.DataFrame) -> pd.Series:
        """Detect residual markets at or below the minimum volume threshold."""
        return self.normalized_volume(df) <= self.min_volume

    def build_mask(self, df: pd.DataFrame) -> pd.Series:
        """Build the low-volume exclusion mask."""
        return self.detect_low_volume_remaining_markets(df)

    def show_low_volume_examples(self, df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
        """Show the highest-volume examples that still fail the final screen."""
        columns = [
            column
            for column in ["market_id", "event_series_slug", "event_title", "question", "volume_num"]
            if column in df.columns
        ]
        return df.loc[self.candidate_mask(df), columns].sort_values("volume_num", ascending=False).head(limit)


def default_block_filters(*, min_volume: float = DEFAULT_MIN_RESIDUAL_VOLUME) -> list[BlockFilter]:
    """Build the default sequence of block filters for selection."""
    return [
        UltraShortRecurringTemplateFilter(),
        FinancialPriceDerivedFilter(),
        AttentionSpeechMetricsFilter(),
        WeatherMarketsFilter(),
        SportsAndEsportsFilter(),
        FinalVolumeScreen(min_volume),
    ]


def apply_default_block_filters(
    df: pd.DataFrame,
    *,
    min_volume: float = DEFAULT_MIN_RESIDUAL_VOLUME,
) -> pd.DataFrame:
    """Apply the canonical block-filter pipeline to a market dataframe."""
    work_df = initialize_work_df(df)
    for block in default_block_filters(min_volume=min_volume):
        candidate_mask = block.candidate_mask(work_df)
        before_unassigned = int(block.unassigned_mask(work_df).sum())
        matched_rows = int(candidate_mask.sum())
        logger.info(
            "block filter stage | block=%s category=%s matched_rows=%s remaining_before=%s",
            block.name,
            block.category,
            matched_rows,
            before_unassigned,
        )
        work_df = block.apply_to_remaining(work_df)
        after_unassigned = int(block.unassigned_mask(work_df).sum())
        logger.info(
            "block filter summary | block=%s remaining_after=%s category_counts=%s",
            block.name,
            after_unassigned,
            block.summarize_categories(work_df).to_dict(),
        )
    return work_df


__all__ = [
    "BlockFilter",
    "DEFAULT_MIN_RESIDUAL_VOLUME",
    "FinalVolumeScreen",
    "FinancialPriceDerivedFilter",
    "SportsAndEsportsFilter",
    "AttentionSpeechMetricsFilter",
    "UltraShortRecurringTemplateFilter",
    "WeatherMarketsFilter",
    "apply_default_block_filters",
    "default_block_filters",
    "initialize_work_df",
]
