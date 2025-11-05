from dataclasses import field
from typing import Optional

from veil.config.core.dataclass import dataclass
from veil.core.base_entity_type import EntityTypeBase


@dataclass(slots=True)
class Span:
    start: int = field(
        default=0,
        metadata={"help": "Byte or char offset in `text`"},
    )
    end: int = field(
        default=0,
        metadata={"help": "Exclusive"},
    )
    entity_type: Optional[EntityTypeBase] = field(
        default=None,
        metadata={
            "help": "Span entity type, must be a member of any subclass of EntityTypeBase."
        },
    )
    id: Optional[int] = field(
        default=None,
        metadata={"help": "Unique identifier for the span."},
    )
    replacement: Optional[str] = field(
        default=None,
        metadata={
            "help": "Replacement text for the span. That is, the original text that will be replaced by the mask."
        },
    )
    confidence: Optional[float] = field(
        default=None,
        metadata={"help": "Confidence score for the span."},
    )
