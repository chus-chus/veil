from typing import List

from veil.config.masker import MaskerConfig
from veil.core.document import Document
from veil.core.enums.mask_type import MaskType
from veil.core.mask_result import MaskResult
from veil.core.span import Span


class Masker:
    def __init__(self, config: MaskerConfig):
        self.config = config

    def mask(self, doc: Document, spans: List[Span]) -> MaskResult:
        """
        Mask the text in document given a list of spans.
        """
        if not spans:
            return MaskResult(
                doc_id=doc.doc_id,
                original_text=doc.text,
                masked_text=doc.text,
                entities=[],
            )

        spans.sort(key=lambda x: x.start)

        masked_chunks: List[str] = []
        cursor = 0
        text = doc.text
        for span in spans:
            start = max(0, min(len(text), int(span.start)))
            end = max(0, min(len(text), int(span.end)))
            if end <= start:
                continue
            # append unmasked region
            if cursor < start:
                masked_chunks.append(text[cursor:start])
            # compute mask token
            original_substring = text[start:end]
            # Do not mutate span (frozen dataclass). Use local original substring.
            masked_chunks.append(self._mask_for_span(span, original_substring))
            cursor = end

        # tail
        if cursor < len(text):
            masked_chunks.append(text[cursor:])

        masked_text = "".join(masked_chunks)

        return MaskResult(
            doc_id=doc.doc_id,
            original_text=text,
            masked_text=masked_text,
            entities=spans,
        )

    def _mask_for_span(self, span: Span, original: str) -> str:
        mask_type = MaskType.from_str(self.config.method)
        if mask_type == MaskType.ASTERISK:
            return "*" * len(original)
        # Default ENTITY_TAG: [ENTITY] or [MASK]
        try:
            label = span.entity_type.name
        except AttributeError:
            raise ValueError(f"ISpan {span} has no entity type")
        span_id = getattr(span, "id", None)
        if span_id is not None:
            token = f"{label}{span_id}"
        else:
            token = label
        return f"[{token}]"
