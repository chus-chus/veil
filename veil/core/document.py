from dataclasses import field
from typing import Any, Dict, List, Optional

from veil.config.core.frozen_dataclass import frozen_dataclass
from veil.core.span import Span


@frozen_dataclass(slots=True)
class Document:
    text: str = field(
        default="",
        metadata={"help": "The raw text that should be masked."},
    )
    doc_id: Optional[str] = field(
        default=None,
        metadata={"help": "Unique identifier."},
    )
    ground_truth: Optional[List[Span]] = field(
        default=None,
        metadata={"help": "Optional gold spans"},
    )
    metadata: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Arbitrary extra info (timestamps, user id, etc.)"},
    )
