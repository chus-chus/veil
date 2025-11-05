from __future__ import annotations

"""Registry mapping :class:`~veil.types.entity_detector_type.EntityDetectorType` to entity detector classes.

external code can instantiate a entity detector via::

    entity_detector_cls = EntityDetectorRegistry.get(EntityDetectorType.REGEX)
    entity_detector = entity_detector_cls()

Registrations happen at import-time to keep the public API friction-free.
"""

from veil.core.base_registry import BaseRegistry
from veil.core.enums.entity_detector_type import EntityDetectorType
from veil.entity_detectors.api.hosted_masker_api_entity_detector import (
    HostedMaskerApiEntityDetector,
)
from veil.entity_detectors.gliner import GlinerEntityDetector
from veil.entity_detectors.regex import RegexEntityDetector
from veil.entity_detectors.spacy import SpacyEntityDetector


class EntityDetectorRegistry(BaseRegistry):
    """Concrete registry for entity detector implementations."""

    @classmethod
    def get_key_from_str(cls, key_str: str) -> EntityDetectorType:
        """Convert a CLI/YAML string into a :class:`EntityDetectorType`."""
        return EntityDetectorType.from_str(key_str)


EntityDetectorRegistry.register(EntityDetectorType.REGEX, RegexEntityDetector)
EntityDetectorRegistry.register(
    EntityDetectorType.HOSTED_MASKER_API, HostedMaskerApiEntityDetector
)
EntityDetectorRegistry.register(EntityDetectorType.GLINER, GlinerEntityDetector)
EntityDetectorRegistry.register(EntityDetectorType.SPACY, SpacyEntityDetector)
