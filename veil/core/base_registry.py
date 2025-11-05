from __future__ import annotations

"""Simplified generic registry following the spec provided by the user."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type

from veil.core.base_int_enum import BaseIntEnum


class BaseRegistry(ABC):
    """Generic enum-key → implementation registry."""

    _key_class: Type[BaseIntEnum] = BaseIntEnum

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry: Dict[BaseIntEnum, Any] = {}

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, key: BaseIntEnum, implementation_class: Any) -> None:
        if key in cls._registry:
            return
        cls._registry[key] = implementation_class

    @classmethod
    def unregister(cls, key: BaseIntEnum) -> None:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")
        del cls._registry[key]

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, key: BaseIntEnum, *args, **kwargs) -> Any:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")
        impl_cls = cls._registry[key]
        try:
            return impl_cls(*args, **kwargs)
        except TypeError:
            # Constructor doesn't accept provided args – fallback to parameter-less.
            return impl_cls()

    @classmethod
    def get_class(cls, key: BaseIntEnum) -> Any:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")
        return cls._registry[key]

    # ------------------------------------------------------------------
    # String helpers – to be implemented by subclasses
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def get_key_from_str(cls, key_str: str) -> BaseIntEnum:
        pass

    @classmethod
    def get_from_str(cls, key_str: str, *args, **kwargs) -> Any:
        return cls.get(cls.get_key_from_str(key_str), *args, **kwargs)
