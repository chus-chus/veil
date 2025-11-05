"""
Pure regular-expression-based masker.

This masker uses optimized regex patterns to detect Spanish-sensitive
identifiers with checksum validation.
"""

from __future__ import annotations

import logging
from typing import List

from veil.config.entity_detectors import RegexEntityDetectorConfig
from veil.core.base_entity_detector import BaseEntityDetector
from veil.core.document import Document
from veil.core.span import Span
from veil.entity_detectors.regex.regex_entity_type import RegexEntityType

from .patterns import (
    get_pattern_meta,
)
from .patterns import validate_pattern as validate_pattern_exact
from .validators import get_supported_validator_types, validate_identifier

logger = logging.getLogger(__name__)


class RegexEntityDetector(BaseEntityDetector[RegexEntityType]):
    """
    Pure regular-expression-based anonymizer.

    Uses optimized regex patterns to detect sensitive identifiers with optional
    checksum validation for higher precision.
    """

    ENTITY_TYPES = {e for e in RegexEntityType}

    def __init__(self, cfg: RegexEntityDetectorConfig):
        """Initialize the regex entity detector with a configuration."""
        super().__init__(cfg)
        self.enable_validation = cfg.enable_validation
        self.min_confidence = cfg.min_confidence
        self.case_sensitive = cfg.case_sensitive

    def detect_entities(self, doc: Document) -> List[Span]:
        """
        Detect entities in the text using regex patterns.

        Args:
            doc: Document to analyze

        Returns:
            List[Span]: List of detected entities
        """
        text = doc.text
        entities: List[Span] = []

        try:
            # Search each entity type
            for entity_type in RegexEntityType:
                entities.extend(self._find_entities_of_type(text, entity_type))

        except Exception as e:
            logger.error(f"Error detecting entities: {e}")
            raise

        return entities

    def _find_entities_of_type(
        self, text: str, entity_type: RegexEntityType
    ) -> List[Span]:
        """
        Find entities of a specific type in the text.

        Args:
            text: Text to analyze
            entity_type: Entity type to search for
            options: Anonymization options

        Returns:
            List[Span]: List of found entities
        """
        entities = []

        try:
            # Get pattern metadata for this entity type
            pattern_meta = get_pattern_meta(entity_type)

            # Find matches
            for match in pattern_meta["compiled"].finditer(text):
                start = match.start()
                end = match.end()
                matched_text = match.group()

                # Validate entity if validation is enabled
                is_valid = True
                if self.enable_validation:
                    is_valid = self._validate_entity(entity_type, matched_text)

                # Create detected entity (aligned with `Span` fields)
                entity = Span(
                    start=start,
                    end=end,
                    entity_type=entity_type,
                    replacement=matched_text,
                    confidence=(
                        float(pattern_meta["confidence"])  # type: ignore[index]
                        if is_valid
                        else float(pattern_meta["confidence"]) * 0.5  # type: ignore[index]
                    ),
                )

                entities.append(entity)

        except Exception as e:
            logger.error(f"Error finding entities of type {entity_type}: {e}")

        return entities

    def _validate_entity(self, entity_type: RegexEntityType, text: str) -> bool:
        """
        Validate an entity using checksums when possible.

        Args:
            entity_type: Entity type
            text: Entity text

        Returns:
            bool: True if the entity is valid
        """
        try:
            # Only validate types that have checksums
            if entity_type in get_supported_validator_types():
                return validate_identifier(entity_type, text)

            # For other types, assume valid if it matches the pattern
            return True

        except Exception as e:
            logger.warning(f"Error validating entity {entity_type}: {e}")
            return True  # If there's an error, assume valid

    def get_pattern_info(self, entity_type: str) -> dict:
        """
        Get pattern information for an entity type.

        Args:
            entity_type: Entity type

        Returns:
            dict: Pattern information
        """
        try:
            entity_enum = RegexEntityType[entity_type]
            pattern_meta = get_pattern_meta(entity_enum)

            return {
                "pattern": pattern_meta["pattern"],
                "confidence": pattern_meta["confidence"],
                "description": pattern_meta["description"],
                "has_validation": entity_type in get_supported_validator_types(),
            }
        except (ValueError, KeyError) as e:
            logger.error(f"Error getting pattern info for {entity_type}: {e}")
            return {}

    def test_pattern(self, entity_type: str, test_text: str) -> bool:
        """
        Test whether a text matches a specific pattern.

        Args:
            entity_type: Entity type to test
            test_text: Text to test

        Returns:
            bool: True if the text matches the pattern
        """
        try:
            entity_enum = RegexEntityType[entity_type]
            return validate_pattern_exact(entity_enum, test_text)
        except (ValueError, KeyError) as e:
            logger.error(f"Error testing pattern for {entity_type}: {e}")
            return False
