"""
Base classes and utilities for pipeline evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from veil.core.base_entity_detector import Span


@dataclass
class EvaluationResult:
    """Basic evaluation metrics."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    details: Dict[str, Any] | None = None


class BaseEvaluator(ABC):
    """
    Base class for Veil evaluators.
    """

    def __init__(self, name: str = "base_evaluator") -> None:
        self.name = name

    @abstractmethod
    def evaluate(
        self,
        predicted: List[Span],
        ground_truth: List[Span],
    ) -> EvaluationResult:
        """Compute evaluation metrics."""
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class SimpleExactMatchEvaluator(BaseEvaluator):
    """
    Simple evaluator based on exact match (span + type).
    """

    def __init__(self) -> None:
        super().__init__("exact_match")

    def evaluate(
        self,
        predicted: List[Span],
        ground_truth: List[Span],
    ) -> EvaluationResult:
        gt_set = {(e.start, e.end, e.entity_type) for e in ground_truth}
        pred_set = {(e.start, e.end, e.entity_type) for e in predicted}

        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )

        return EvaluationResult(
            precision=precision,
            recall=recall,
            f1=f1,
            tp=tp,
            fp=fp,
            fn=fn,
            details={},
        )
