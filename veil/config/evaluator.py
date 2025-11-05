from dataclasses import field
from typing import List

from veil.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass
class EvaluatorConfig:
    """Configuration for the evaluator."""

    enabled: bool = field(
        default=True,
        metadata={
            "help": "Enable evaluation. Ground truth must be provided in the datahandler. Results will be handled by the metric store."
        },
    )

    strict_type: bool = field(
        default=True,
        metadata={
            "help": "If True, a predicted span only matches a ground-truth span when their entity types are equal."
        },
    )

    # Reporting controls (does not change matching used for the primary/legacy fields)
    report_exact: bool = field(
        default=True,
        metadata={
            "help": "If True, always include an 'exact' variant in the evaluation report and metrics."
        },
    )

    report_exact_by_id: bool = field(
        default=True,
        metadata={
            "help": "If True, include an 'exact_by_id' variant where matches also require identical non-null span IDs."
        },
    )

    report_iou_thresholds: List[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9],
        metadata={
            "help": "List of IoU thresholds to compute and report in addition to the primary mode."
        },
    )

    use_hungarian_id_mapping: bool = field(
        default=False,
        metadata={
            "help": "If True, use optimal (Hungarian-like) assignment for ID cluster mapping instead of greedy."
        },
    )
