from abc import ABC, abstractmethod
from typing import Generic, List, Set, TypeVar

from veil.config.entity_detectors import BaseEntityDetectorConfig
from veil.core.base_entity_type import EntityTypeBase
from veil.core.document import Document
from veil.core.span import Span

E = TypeVar("E", bound=EntityTypeBase)


class BaseEntityDetector(Generic[E], ABC):

    # child classes must define this
    ENTITY_TYPES: Set[E]

    def __init__(
        self,
        config: BaseEntityDetectorConfig,
    ):
        """Initialize base entity detector with basic universal parameters.

        Sub-classes can accept more parameters via **kwargs and must set their
        own attributes (e.g., validation flags).
        """
        self.config = config

    @classmethod
    def get_supported_entities(cls) -> List[str]:
        return [e.name for e in cls.ENTITY_TYPES]

    @abstractmethod
    def detect_entities(self, doc: Document) -> List[Span]:
        """
        Detect entities in the document.
        """
        raise NotImplementedError("Subclasses must implement this method")
