from veil.core.base_int_enum import BaseIntEnum


class EntityDetectorType(BaseIntEnum):
    """Enum for available entity detector types."""

    REGEX = 1
    MASKER_API = 2
    HOSTED_MASKER_API = 3
    GLINER = 4
    SPACY = 5
