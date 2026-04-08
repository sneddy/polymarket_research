"""Utility helpers for the core Polymarket dataset package."""

from polymarket_research.utils.filesystem import setup_root
from polymarket_research.utils.metrics import (
    clipped_probabilities,
    safe_auc,
    safe_average_precision,
)
from polymarket_research.utils.text import (
    build_family_id,
    normalize_text,
    parse_listish,
    tag_jaccard,
)

__all__ = [
    "build_family_id",
    "clipped_probabilities",
    "normalize_text",
    "parse_listish",
    "safe_auc",
    "safe_average_precision",
    "setup_root",
    "tag_jaccard",
]
