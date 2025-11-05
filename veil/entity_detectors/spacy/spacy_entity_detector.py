from __future__ import annotations

from typing import List, Optional

from veil.config.entity_detectors import SpacyEntityDetectorConfig
from veil.core.base_entity_detector import BaseEntityDetector
from veil.core.document import Document
from veil.core.span import Span
from veil.logger import init_logger

from .spacy_entity_type import SpacyEntityType

logger = init_logger(__name__)


class SpacyEntityDetector(BaseEntityDetector[SpacyEntityType]):
    """Adapter over spaCy NER models."""

    ENTITY_TYPES = {e for e in SpacyEntityType}

    def __init__(self, config: SpacyEntityDetectorConfig):
        super().__init__(config)
        try:
            import spacy
        except ImportError as e:
            raise RuntimeError(
                "Spacy library not available. Please install it using: pip install spacy"
            ) from e

        # Configure device
        use_gpu = config.cuda_device >= 0
        if use_gpu:
            try:
                spacy.require_gpu(config.cuda_device)
                logger.info(f"spaCy set to use GPU device {config.cuda_device}")
            except Exception as gpu_err:
                logger.warning(
                    f"Failed to enable spaCy GPU on device {config.cuda_device}: {gpu_err}. Falling back to CPU."
                )
                try:
                    spacy.require_cpu()
                except Exception:
                    pass
        else:
            try:
                spacy.require_cpu()
            except Exception:
                pass

        # Load the spaCy pipeline
        logger.info(f"Loading spaCy model '{config.model}'")
        self._nlp = spacy.load(config.model)

        # Build alias mapping for label conversion
        self._alias_map = SpacyEntityType._build_alias_map_for_subclass()

    def _map_label_to_entity(self, label: str) -> Optional[SpacyEntityType]:
        key = label.upper()
        canonical_name = self._alias_map.get(key)
        if canonical_name:
            try:
                return SpacyEntityType[canonical_name]
            except KeyError:
                return None
        return None

    def detect_entities(self, doc: Document) -> List[Span]:
        text = doc.text or ""
        if not text:
            return []

        spacy_doc = self._nlp(text)
        spans: List[Span] = []

        for ent in spacy_doc.ents:
            entity_type = self._map_label_to_entity(ent.label_)
            if entity_type is None:
                continue
            spans.append(
                Span(
                    start=ent.start_char,
                    end=ent.end_char,
                    entity_type=entity_type,
                    replacement=ent.text,
                    confidence=None,
                )
            )

        return spans
