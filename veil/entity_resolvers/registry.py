from __future__ import annotations

from veil.core.base_registry import BaseRegistry
from veil.core.enums.entity_resolver_type import EntityResolverType

from .embeddings_resolver import EmbeddingsEntityResolver


class EntityResolverRegistry(BaseRegistry):
    """Concrete registry for entity resolver implementations."""

    @classmethod
    def get_key_from_str(cls, key_str: str) -> EntityResolverType:
        return EntityResolverType.from_str(key_str)


EntityResolverRegistry.register(EntityResolverType.EMBEDDINGS, EmbeddingsEntityResolver)
