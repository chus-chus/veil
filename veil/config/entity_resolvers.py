from dataclasses import field

from veil.config.core.base_poly_config import BasePolyConfig
from veil.config.core.frozen_dataclass import frozen_dataclass
from veil.core.enums.entity_resolver_type import EntityResolverType


@frozen_dataclass
class BaseEntityResolverConfig(BasePolyConfig):
    """Marker base for entity resolver configuration objects."""

    @classmethod
    def get_type(cls):
        raise NotImplementedError


@frozen_dataclass
class EmbeddingsEntityResolverConfig(BaseEntityResolverConfig):
    """Configuration for an embeddings-based entity resolver.

    For v1 we keep options minimal and local-only.
    """

    # Similarity threshold to link mentions (0..1)
    threshold: float = field(default=0.82)
    # Maximum left/right context characters to include in representation (0 to disable)
    context_chars: int = field(default=0)

    @classmethod
    def get_type(cls):
        return EntityResolverType.EMBEDDINGS
