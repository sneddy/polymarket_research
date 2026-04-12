"""Research-layer modules.

This package intentionally avoids eager imports because some legacy research
helpers depend on unfinished benchmark code that is not part of the stable
package surface.
"""

from .question_decoupling import DecomposedQuestion, QuestionDecoupler
from .topic_models import (
    FASTopicFactory,
    S3TopicFactory,
    TFIDFTopicFactory,
    TopicFactory,
    TopicModel,
    compare_topic_factories,
)

__all__ = [
    "DecomposedQuestion",
    "FASTopicFactory",
    "QuestionDecoupler",
    "S3TopicFactory",
    "TFIDFTopicFactory",
    "TopicFactory",
    "TopicModel",
    "compare_topic_factories",
]
