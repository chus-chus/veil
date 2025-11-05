from dataclasses import field
from typing import Any, Dict, List, Optional

from veil.config.core.frozen_dataclass import frozen_dataclass
from veil.core.span import Span


@frozen_dataclass(slots=True)
class MaskResult:
    """Result of applying masking to a document.

    Contains the masked text and the spans that were masked.
    """

    doc_id: Optional[str] = field(default=None)
    original_text: str = field(default="")
    masked_text: str = field(default="")
    entities: List[Span] = field(default_factory=list)
    evaluation: Optional[Dict[str, Any]] = field(default=None)
