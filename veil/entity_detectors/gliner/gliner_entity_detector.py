from __future__ import annotations

from typing import List, Optional, Tuple

from veil.config.entity_detectors import GlinerEntityDetectorConfig
from veil.core.base_entity_detector import BaseEntityDetector
from veil.core.document import Document
from veil.core.span import Span
from veil.logger import init_logger

from .gliner_entity_type import GlinerEntityType

logger = init_logger(__name__)


class GlinerEntityDetector(BaseEntityDetector[GlinerEntityType]):
    """Adapter over Gliner entity detection models with support for chunking long documents."""

    ENTITY_TYPES = {e for e in GlinerEntityType}

    def __init__(self, config: GlinerEntityDetectorConfig):
        super().__init__(config)
        try:
            from gliner import GLiNER
        except ImportError as e:
            raise RuntimeError(
                "GLiNER library not available. Please install it using: pip install gliner"
            ) from e

        # Load the model
        device = "cuda" if config.cuda_device >= 0 else "cpu"

        logger.info(f"Loading Gliner model from {config.model} on {device}")
        self.model = GLiNER.from_pretrained(config.model, map_location=device)

        # Get the tokenizer for chunking
        self.tokenizer = self.model.data_processor.transformer_tokenizer

        # Build alias mapping for label conversion
        self._alias_map = GlinerEntityType._build_alias_map_for_subclass()

    def _map_label_to_entity(self, label: str) -> Optional[GlinerEntityType]:
        """Map a GLiNER output label to a GlinerEntityType using the alias system."""
        key = label.upper()
        canonical_name = self._alias_map.get(key)
        if canonical_name:
            try:
                return GlinerEntityType[canonical_name]
            except KeyError:
                return None
        return None

    def _create_chunks(self, text: str) -> List[Tuple[str, int, int]]:
        """Create overlapping chunks of text with character offsets.

        Args:
            text: Input text to chunk

        Returns:
            List of tuples (chunk_text, start_char, end_char) where start_char and end_char
            are the character positions in the original text
        """
        # Tokenize the full text
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )

        tokens = encoding.tokens()
        offsets = encoding["offset_mapping"]

        # If text fits within max_length, return it as a single chunk
        if len(tokens) <= self.config.max_length:
            return [(text, 0, len(text))]

        chunks = []
        chunk_size = self.config.max_length
        overlap = self.config.chunk_overlap

        i = 0
        while i < len(tokens):
            # Determine chunk end
            chunk_end = min(i + chunk_size, len(tokens))

            # Get character positions for this chunk
            start_char = offsets[i][0]
            end_char = offsets[chunk_end - 1][1]

            # Extract chunk text
            chunk_text = text[start_char:end_char]
            chunks.append((chunk_text, start_char, end_char))

            # Move to next chunk with overlap
            if chunk_end >= len(tokens):
                break
            i = chunk_end - overlap

        return chunks

    def _merge_overlapping_entities(self, spans: List[Span]) -> List[Span]:
        """Deduplicate entities that appear in overlapping chunks.

        Keeps the entity with the highest confidence score when duplicates are found.
        """
        if not spans:
            return []

        # Sort by start position
        sorted_spans = sorted(spans, key=lambda s: (s.start, s.end))

        merged = []
        i = 0
        while i < len(sorted_spans):
            current = sorted_spans[i]
            j = i + 1

            # Find all overlapping spans with the same entity type
            duplicates = [current]
            while j < len(sorted_spans):
                next_span = sorted_spans[j]

                # Check if spans overlap significantly (IoU > 0.5) and same type
                if (
                    next_span.entity_type == current.entity_type
                    and self._calculate_iou(current, next_span)
                    > self.config.nms_iou_threshold
                ):
                    duplicates.append(next_span)
                    j += 1
                else:
                    break

            # Keep the one with highest confidence
            best = max(duplicates, key=lambda s: s.confidence or 0.0)
            merged.append(best)

            i = j

        return merged

    def _calculate_iou(self, span1: Span, span2: Span) -> float:
        """Calculate Intersection over Union for two spans."""
        # Calculate intersection
        start = max(span1.start, span2.start)
        end = min(span1.end, span2.end)
        intersection = max(0, end - start)

        # Calculate union
        union = (span1.end - span1.start) + (span2.end - span2.start) - intersection

        if union == 0:
            return 0.0
        return intersection / union

    def detect_entities(self, doc: Document) -> List[Span]:
        """Detect entities in the document using GLiNER model with chunking support.

        Long documents are automatically split into overlapping chunks to handle
        the model's maximum sequence length limitation.

        Args:
            doc: Document to process

        Returns:
            List of Span objects with detected entities
        """
        text = doc.text or ""
        if not text:
            return []

        # Get labels from config
        labels = self.config.labels

        # Create chunks
        chunks = self._create_chunks(text)

        # Process each chunk
        all_spans: List[Span] = []

        for chunk_text, start_offset, _ in chunks:
            # Run GLiNER inference on this chunk
            entities = self.model.predict_entities(
                chunk_text,
                labels,
                threshold=self.config.threshold,
            )

            # Convert GLiNER entities to Span objects
            filtered = []
            for entity in entities:
                # Map the label to our entity type
                entity_type = self._map_label_to_entity(entity["label"])

                # Only include entities with recognized types
                if entity_type is not None:
                    # Adjust offsets to match original document
                    start = entity["start"] + start_offset
                    end = entity["end"] + start_offset
                    replacement = entity["text"]
                    # Basic length filters
                    if end <= start:
                        continue
                    span_len = end - start
                    if (
                        span_len < self.config.min_span_chars
                        or span_len > self.config.max_span_chars
                    ):
                        continue

                    filtered.append(
                        Span(
                            start=start,
                            end=end,
                            entity_type=entity_type,
                            replacement=replacement,
                            confidence=entity.get("score"),
                        )
                    )
            # Keep top-K per chunk to limit over-detection
            if (
                self.config.top_k_per_chunk
                and len(filtered) > self.config.top_k_per_chunk
            ):
                filtered = sorted(
                    filtered, key=lambda s: s.confidence or 0.0, reverse=True
                )[: self.config.top_k_per_chunk]
            all_spans.extend(filtered)

        # Merge overlapping entities from different chunks
        merged_spans = self._merge_overlapping_entities(all_spans)

        return merged_spans
