"""
Veil - Modular toolkit for masking sensitive entities (PII) with a regex engine for Spanish identifiers.
"""

__version__: str = "0.1.0"

from . import entity_detectors as entity_detectors
from . import evaluator as evaluator
from . import metric_store as metric_store
from . import pipeline as pipeline

__all__ = [
    "entity_detectors",
    "metric_store",
    "evaluator",
    "pipeline",
]
