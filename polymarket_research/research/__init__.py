"""Research-layer modules with lazy exports for optional heavy dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "DecomposedQuestion": ("polymarket_research.research.question_decoupling", "DecomposedQuestion"),
    "QuestionDecoupler": ("polymarket_research.research.question_decoupling", "QuestionDecoupler"),
    "FASTopicFactory": ("polymarket_research.research.topic_models", "FASTopicFactory"),
    "S3TopicFactory": ("polymarket_research.research.topic_models", "S3TopicFactory"),
    "TFIDFTopicFactory": ("polymarket_research.research.topic_models", "TFIDFTopicFactory"),
    "TopicFactory": ("polymarket_research.research.topic_models", "TopicFactory"),
    "TopicModel": ("polymarket_research.research.topic_models", "TopicModel"),
    "compare_topic_factories": ("polymarket_research.research.topic_models", "compare_topic_factories"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
