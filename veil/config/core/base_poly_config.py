from abc import ABC
from typing import Any

from veil.config.core.frozen_dataclass import frozen_dataclass
from veil.config.utils import get_all_subclasses


@frozen_dataclass
class BasePolyConfig(ABC):
    @classmethod
    def create_from_type(cls, type_: Any) -> Any:
        for subclass in get_all_subclasses(cls):
            if subclass.get_type() == type_:
                return subclass()
        raise ValueError(f"Invalid type: {type_}")

    @classmethod
    def get_type(cls) -> Any:
        raise NotImplementedError(
            f"[{cls.__name__}] get_type() method is not implemented"
        )
