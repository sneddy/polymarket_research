"""Text and tag helpers shared across research notebooks."""

from __future__ import annotations

import ast
import re

import numpy as np


STOPWORD_TOKENS = {"will", "the", "a", "an", "be", "is", "are", "to", "of", "by", "in"}


def parse_listish(value) -> list[str]:
    """Parse a stored tag/list field into a clean list of strings."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = None

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]

    if "|" in text:
        parts = text.split("|")
    elif "," in text:
        parts = text.split(",")
    else:
        parts = [text]
    return [part.strip() for part in parts if part.strip()]


def normalize_text(value: str) -> str:
    """Normalize free text for simple lexical matching."""
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_family_id(question: str, domain: str, tags) -> str:
    """Build a weak family identifier from market text and tags."""
    norm_question = normalize_text(question)
    norm_tags = [normalize_text(tag) for tag in parse_listish(tags)]
    tokens = [token for token in norm_question.split() if token not in STOPWORD_TOKENS]
    key = " ".join(tokens[:6]) if tokens else norm_question[:48]
    tag_key = "|".join(sorted(norm_tags[:3]))
    return f"{domain}::{tag_key}::{key}".strip(":")


def tag_jaccard(tags_a, tags_b) -> float:
    """Compute Jaccard overlap between two tag sets."""
    left = set(parse_listish(tags_a))
    right = set(parse_listish(tags_b))
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)
