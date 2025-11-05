from dataclasses import field

from veil.config.core.frozen_dataclass import frozen_dataclass
from veil.core.enums.mask_type import MaskType


@frozen_dataclass
class MaskerConfig:
    """Configuration for the masker."""

    method: str = field(
        default=MaskType.ENTITY_TAG.name,
        metadata={"help": "Method to use for masking."},
    )

    def __post_init__(self):
        MaskType.from_str(self.method)  # will raise an error if the method is invalid
