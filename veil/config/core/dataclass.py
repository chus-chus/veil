import dataclasses
import typing


@typing.dataclass_transform()
def dataclass(_cls=None, **kwargs):
    """
    A decorator that creates a thin wrapper around dataclasses.dataclass,
    allowing allow_from_file to be passed in as a keyword argument.

    Args:
        _cls: The class to decorate (for decorator syntax handling).
        **kwargs: Additional keyword arguments to pass to dataclasses.dataclass.
        allow_from_file (bool): If True, mark the dataclass as allowing configuration
            to be loaded from a file.
    """
    # Extract custom keyword arguments handled by this decorator
    allow_from_file = kwargs.pop("allow_from_file", False)

    def wrap(cls):
        """Apply the standard dataclass decorator while handling extras."""
        # Apply the built-in dataclass decorator with the remaining kwargs
        datacls = dataclasses.dataclass(cls, **kwargs)  # type: ignore[arg-type]

        # Mark as allowing file-based configuration if requested
        if allow_from_file:
            setattr(datacls, "_allow_from_file", True)

        return datacls

    # Support both @dataclass and @dataclass() syntaxes
    if _cls is None:
        return wrap
    return wrap(_cls)
