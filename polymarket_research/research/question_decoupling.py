"""Question-decoupling helpers for Polymarket market text.

This module isolates the text decomposition step used in exploratory
closeness notebooks:

- qualitative core: the underlying event/entity/question
- quantitative trigger: temporal or numeric resolution condition
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import pandas as pd

_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DOW = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_QUARTER = r"(?:q[1-4](?:\s+\d{4})?|quarter\s+[1-4](?:\s+\d{4})?|h[12]\s+\d{4})"
_MEETING = rf"(?:the\s+)?{_MONTH}\s+\d{{4}}\s+meeting"
_DATE_POINT = rf"(?:{_MEETING}|(?:the\s+end\s+of\s+)?{_MONTH}(?:\s+\d{{1,2}}(?:,?\s+\d{{4}})?|\s+\d{{4}})?|(?:the\s+end\s+of\s+)?{_QUARTER}|(?:the\s+end\s+of\s+)?\d{{4}}|{_DOW})"
_TRIGGER_PATTERNS = [
    rf"\b(?:by|before|after|on)\s+{_DATE_POINT}\b",
    rf"\b{_DOW}\b",
    r"\bwithin\s+\d+\s+(?:day|week|month|hour|year)s?\b",
    r"\bin\s+\d{4}\b",
    r"\bby\s+(?:the\s+)?end\s+of\s+\d{4}\b",
    r"\bfirst\s+\d+\s+(?:day|week|month|hour)s?\b",
    r"\bnext\s+\d+\s+(?:day|week|month|hour)s?\b",
]
_TRIGGER_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in _TRIGGER_PATTERNS]


def _normalize(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s\$%,.+]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _select_non_overlapping_matches(
    patterns: list[re.Pattern[str]],
    text: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for order, pattern in enumerate(patterns):
        for match in pattern.finditer(text):
            candidates.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group().strip(),
                    "order": order,
                }
            )

    candidates.sort(key=lambda item: (item["start"], -(item["end"] - item["start"]), item["order"]))

    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        overlaps_existing = any(
            not (candidate["end"] <= kept["start"] or candidate["start"] >= kept["end"])
            for kept in selected
        )
        if overlaps_existing:
            continue
        selected.append(candidate)

    selected.sort(key=lambda item: item["start"])
    return selected


def _remove_spans(text: str, spans: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        pieces.append(text[cursor : span["start"]])
        cursor = span["end"]
    pieces.append(text[cursor:])
    cleaned = " ".join(pieces)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.strip(" ,")


@dataclass(frozen=True)
class DecomposedQuestion:
    """Normalized decomposition of one market question."""

    original: str
    decoupled_event: str
    decoupled_trigger: str
    trigger_candidates: list[str]
    has_trigger: bool
    qual_core: str
    quant_spans: list[str]
    has_quant: bool


class QuestionDecoupler:
    """Split market questions into qualitative core and quantitative trigger."""

    def decompose_question(self, question: str) -> DecomposedQuestion:
        normalized = _normalize(question)
        trigger_matches = _select_non_overlapping_matches(_TRIGGER_COMPILED, normalized)
        trigger_candidates = list(dict.fromkeys(match["text"] for match in trigger_matches))

        # When several temporal candidates exist, the last one is usually the
        # explicit market-resolution clause rather than part of the event text.
        decoupled_trigger = trigger_candidates[-1] if trigger_candidates else ""
        chosen_matches = [trigger_matches[-1]] if trigger_matches else []
        decoupled_event = _remove_spans(normalized, chosen_matches)

        return DecomposedQuestion(
            original=question,
            decoupled_event=decoupled_event,
            decoupled_trigger=decoupled_trigger,
            trigger_candidates=trigger_candidates,
            has_trigger=bool(decoupled_trigger),
            qual_core=decoupled_event,
            quant_spans=[decoupled_trigger] if decoupled_trigger else [],
            has_quant=bool(decoupled_trigger),
        )

    def decompose_markets(
        self,
        markets: pd.DataFrame,
        *,
        question_col: str = "question",
    ) -> pd.DataFrame:
        """Attach decomposition columns to a market frame."""
        if question_col not in markets.columns:
            raise KeyError(f"Expected column '{question_col}' in markets frame.")

        out = markets.copy()
        decomposed = out[question_col].apply(self.decompose_question)
        out["decoupled_event"] = [item.decoupled_event for item in decomposed]
        out["decoupled_trigger"] = [item.decoupled_trigger for item in decomposed]
        out["trigger_candidates"] = [item.trigger_candidates for item in decomposed]
        out["has_trigger"] = [item.has_trigger for item in decomposed]
        out["qual_core"] = [item.qual_core for item in decomposed]
        out["quant_spans"] = [item.quant_spans for item in decomposed]
        out["has_quant"] = [item.has_quant for item in decomposed]
        return out


__all__ = [
    "DecomposedQuestion",
    "QuestionDecoupler",
]
