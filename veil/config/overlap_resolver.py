from dataclasses import field

from veil.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass
class OverlapResolverConfig:
    """Configuration for overlap resolution between spans from different detectors.

    IoU threshold controls when two spans are considered conflicting:
    - If IoU > threshold, they conflict and the lower-priority span is dropped.
    - With threshold=0.0, any real overlap is considered a conflict.
    """

    iou_threshold: float = field(
        default=0.75,
        metadata={
            "help": "Intersection-over-Union threshold in [0,1). IoU > threshold triggers conflict.",
        },
    )

    overlap_policy: str = field(
        default="same_type_only",
        metadata={
            "help": (
                "Overlap policy. 'same_type_only' = conflicts only within same canonical type; "
                "'cross_type' = treat cross-type overlaps as conflicts using the same selection rules."
            )
        },
    )

    def __post_init__(self):
        if not (0.0 <= self.iou_threshold < 1.0):
            raise ValueError("iou_threshold must be in [0.0, 1.0)")
        try:
            policy = str(self.overlap_policy).lower()
        except Exception:
            policy = "same_type_only"
        if policy not in {"same_type_only", "cross_type"}:
            raise ValueError(
                "overlap_policy must be one of {'same_type_only', 'cross_type'}"
            )
