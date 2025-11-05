from veil.core.base_int_enum import BaseIntEnum


class MaskType(BaseIntEnum):
    """Available masking methods."""

    ENTITY_TAG = 1  # <MASK>
    ASTERISK = 2  # **********
