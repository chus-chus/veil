import hashlib
import logging
import sys
from copy import deepcopy
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Union, get_args, get_origin, get_type_hints

primitive_types = {int, str, float, bool, type(None)}

logger = logging.getLogger(__name__)


def _get_hash(key):
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def get_all_subclasses(cls):
    subclasses = cls.__subclasses__()
    return subclasses + [g for s in subclasses for g in get_all_subclasses(s)]


def is_primitive_type(field_type: type) -> bool:
    # Check if the type is a primitive type
    return field_type in primitive_types


def is_generic_composed_of_primitives(field_type: type) -> bool:
    origin = get_origin(field_type)
    if origin in {list, dict, tuple, Union}:
        # Check all arguments of the generic type
        args = get_args(field_type)
        return all(is_composed_of_primitives(arg) for arg in args)
    return False


def is_composed_of_primitives(field_type: type) -> bool:
    # Check if the type is a primitive type
    if is_primitive_type(field_type):
        return True

    # Check if the type is a generic type composed of primitives
    if is_generic_composed_of_primitives(field_type):
        return True

    return False


def to_snake_case(name: str) -> str:
    return "".join(["_" + i.lower() if i.isupper() else i for i in name]).lstrip("_")


def is_optional(field_type: type) -> bool:
    return get_origin(field_type) is Union and type(None) in get_args(field_type)


def is_list(field_type: type) -> bool:
    # Check if the field type is a List
    return get_origin(field_type) is list


def is_dict(field_type: type) -> bool:
    # Check if the field type is a Dict
    return get_origin(field_type) is dict


def is_bool(field_type: type) -> bool:
    return field_type is bool


def get_inner_type(field_type: type) -> type:
    return next(t for t in get_args(field_type) if t is not type(None))


def is_subclass(cls, parent: type) -> bool:
    return hasattr(cls, "__bases__") and parent in cls.__bases__


from enum import Enum


def dataclass_to_dict(obj):
    """
    Recursively convert a dataclass (or any nested structure containing
    dataclasses) into a JSON-serialisable structure composed only of
    dicts, lists and primitive types.

    Special handling:
    • Enum instances are converted to their ``value`` (or ``name`` when the
      value is not JSON-serialisable).
    • Dictionaries are traversed recursively.
    """
    # lists and tuples
    if isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]

    # dicts
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}

    # enums
    if isinstance(obj, Enum):
        return (
            obj.value
            if isinstance(obj.value, (str, int, float, bool, type(None)))
            else obj.name
        )

    # dataclasses
    if is_dataclass(obj):
        data = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            data[field.name] = dataclass_to_dict(value)

        for key, value in obj.__dict__.items():
            if key not in data:
                data[key] = dataclass_to_dict(value)

        if hasattr(obj, "get_type") and callable(getattr(obj, "get_type", None)):
            data["name"] = str(obj.get_type())  # type: ignore[attr-defined]
        elif hasattr(obj, "get_name") and callable(getattr(obj, "get_name", None)):
            data["name"] = obj.get_name()  # type: ignore[attr-defined]
        return data

    # all other primitives
    return obj


def dict_to_args(class_dict):
    args = []
    for key, value in class_dict.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
                else:
                    args.append(f"--no-{key}")
            else:
                args.append(f"--{key} {value}")
    return " ".join(args)


def expand_dict(d: Dict) -> List[Dict]:
    """
    Recursively expand a configuration dictionary that may contain lists into
    a list of dictionaries representing every combination in the Cartesian
    product of the list elements. Lists may appear at any depth of the
    configuration tree. Nested dictionaries are handled recursively.
    """
    variants: List[Dict] = [dict()]

    for key, value in d.items():
        # Figure out all the possible values for this key
        if isinstance(value, list):
            # Each element in the list may be a dictionary that itself needs
            # expansion. Scalars are taken as-is.
            possible_values = []
            for item in value:
                if isinstance(item, dict):
                    possible_values.extend(expand_dict(item))
                else:
                    possible_values.append(item)
        elif isinstance(value, dict):
            possible_values = expand_dict(value)
        else:
            possible_values = [value]

        # Compose current variants with the new possibilities (cartesian product)
        new_variants: list[dict] = []
        for base in variants:
            for option in possible_values:
                v_copy = deepcopy(base)
                v_copy[key] = option
                new_variants.append(v_copy)
        variants = new_variants

    return variants


def _strip_optional(t: Any) -> Any:
    """Return the inner type if *t* is Optional[T], otherwise return *t* unchanged."""
    if get_origin(t) is Union:
        non_none = [a for a in get_args(t) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return t


def _issubclass_safe(cls: Any, parent: type) -> bool:
    """Safe variant of issubclass that returns False for non-class *cls*."""
    try:
        return isinstance(cls, type) and issubclass(cls, parent)
    except TypeError:
        return False


def _match_subclass_by_type(parent: type, type_val: Any) -> type:
    """Return the subclass of *parent* whose ``get_type()`` equals *type_val*.

    The comparison is performed with a bit of leniency: we allow Enum values,
    strings matching Enum names (case-insensitive) or the string or repr of the
    Enum value itself.
    """
    from veil.config.utils import get_all_subclasses

    for subclass in get_all_subclasses(parent):
        subtype = subclass.get_type()
        if subtype == type_val:
            return subclass
        # Handle Enum → str comparisons in either direction
        if hasattr(subtype, "name") and isinstance(type_val, str):
            if subtype.name.lower() == type_val.lower():
                return subclass
        if isinstance(subtype, str) and isinstance(type_val, str):
            if subtype.lower() == type_val.lower():
                return subclass
    raise ValueError(
        f"No subclass of '{parent.__name__}' matches type value '{type_val}'."
    )


def create_class_from_dict(cls: type, config_dict: dict | None):
    """Recursively instantiate *cls* using values from *config_dict*.

    This utility understands three kinds of fields:
    1. Primitive (or containers of primitives) – their value is taken directly.
    2. Nested dataclasses – they are created recursively.
    3. Polymorphic configs that inherit from ``BasePolyConfig`` – the concrete
       subclass is selected using the ``type`` key in the corresponding dict
       (or by providing the type directly as a string/enum value).

    If there are any keys present in *config_dict* that do not correspond to a dataclass
    field, a TypeError is raised.
    """
    from veil.config.core.base_poly_config import BasePolyConfig

    # Fast path: if cls is not a dataclass return config_dict as is
    if (
        not is_dataclass(cls)
        or config_dict is None
        or not isinstance(config_dict, dict)
    ):
        logger.debug(
            "create_class_from_dict fast path for %s with value %s", cls, config_dict
        )
        # Caller will assign directly.
        return config_dict

    # Handle polymorphic configs (BasePolyConfig) before validating fields so the
    # concrete subclass defines the schema to check against.
    if _issubclass_safe(cls, BasePolyConfig):
        # Dict payload with a type discriminator → pick concrete subclass.
        if isinstance(config_dict, dict) and "type" in config_dict:
            type_val = config_dict["type"]
            subclass = _match_subclass_by_type(cls, type_val)
            sub_dict = {k: v for k, v in config_dict.items() if k != "type"}
            return create_class_from_dict(subclass, sub_dict)
        # Bare discriminator value (e.g. "regex")
        if not isinstance(config_dict, dict):
            subclass = _match_subclass_by_type(cls, config_dict)
            return subclass()

    kwargs: dict[str, Any] = {}

    try:
        module = sys.modules.get(cls.__module__)
        globalns = vars(module) if module else None
        type_hints = get_type_hints(cls, globalns=globalns)
    except Exception:
        type_hints = {}

    # Warn about any unexpected keys in the config dictionary
    known_fields = {f.name for f in fields(cls)}
    extra_keys = set(config_dict) - known_fields
    if extra_keys:
        logger.error(
            "create_class_from_dict: unknown arguments for %s: %s",
            cls.__name__,
            sorted(extra_keys),
        )
        raise TypeError(
            f"Unknown arguments provided for {cls.__name__}: {sorted(extra_keys)}"
        )

    for f in fields(cls):
        if f.name not in config_dict:
            logger.debug(
                "Field '%s' not supplied for %s; using default.", f.name, cls.__name__
            )
            # Not supplied: rely on dataclass default
            continue

        raw_value = config_dict[f.name]
        field_type = _strip_optional(type_hints.get(f.name, f.type))

        # Handle list/tuple/dict containers with potential dataclass items
        origin = get_origin(field_type)
        if origin is list and isinstance(raw_value, list):
            inner_type = _strip_optional(get_args(field_type)[0])
            if is_dataclass(inner_type) or _issubclass_safe(inner_type, BasePolyConfig):
                processed_list = [
                    (
                        create_class_from_dict(inner_type, itm)
                        if isinstance(itm, dict)
                        else itm
                    )
                    for itm in raw_value
                ]
                kwargs[f.name] = processed_list
                continue
        elif origin is dict and isinstance(raw_value, dict):
            key_type, val_type = get_args(field_type)
            if is_dataclass(val_type) or _issubclass_safe(val_type, BasePolyConfig):
                processed_dict = {
                    k: create_class_from_dict(val_type, v) if isinstance(v, dict) else v
                    for k, v in raw_value.items()
                }
                kwargs[f.name] = processed_dict
                continue

        # Polymorphic config: choose subclass based on "type" key
        if _issubclass_safe(field_type, BasePolyConfig):
            if isinstance(raw_value, dict):
                type_val = raw_value.get("type")
                if type_val is not None:
                    subclass = _match_subclass_by_type(field_type, type_val)
                    sub_dict = {k: v for k, v in raw_value.items() if k != "type"}
                else:
                    # Type not specified: assume the parent type itself
                    subclass = field_type
                    sub_dict = raw_value
                kwargs[f.name] = create_class_from_dict(subclass, sub_dict)
            else:
                # raw_value directly specifies the type discriminator
                subclass = _match_subclass_by_type(field_type, raw_value)
                kwargs[f.name] = subclass()
            continue

        # Nested dataclass (non-poly)
        if is_dataclass(field_type):
            if isinstance(raw_value, dict):
                kwargs[f.name] = create_class_from_dict(field_type, raw_value)
            else:
                kwargs[f.name] = raw_value
            continue

        # Primitive or unknown – assign directly.
        kwargs[f.name] = raw_value

    try:
        return cls(**kwargs)
    except TypeError as e:
        logger.error(
            "Failed to instantiate %s with kwargs %s. Error: %s",
            cls.__name__,
            kwargs,
            e,
        )
        raise


def load_yaml_config(file_path: str):
    """Load a YAML configuration file and return its contents.

    The function performs a series of robustness checks and provides
    informative log messages for each failure mode.

    1. Verifies the file exists and is readable.
    2. Attempts to parse using ``yaml.safe_load``.
    3. On YAML parse errors, falls back to ``json.loads`` (helpful when the
       file is actually JSON or a subset thereof).
    4. Returns the parsed content, which can be either a dictionary (mapping)
       or a list. Lists are automatically wrapped in a dictionary with a
       special key to maintain compatibility with the configuration machinery.

    Parameters
    ----------
    file_path : str
        Path to the YAML/JSON configuration file.

    Returns
    -------
    dict
        Parsed configuration. If the top-level is a list, it's wrapped
        as {'_list': <the_list>}.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    PermissionError
        If the file cannot be read.
    yaml.YAMLError | json.JSONDecodeError
        If the file cannot be parsed as YAML or JSON.
    ValueError
        If the top-level parsed object is neither a mapping nor a list.
    """

    import json
    import os

    import yaml

    # check file
    if not os.path.exists(file_path):
        logger.error("Configuration file '%s' does not exist.", file_path)
        raise FileNotFoundError(f"Configuration file '{file_path}' not found.")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
    except Exception as exc:
        logger.error("Failed to read configuration file '%s': %s", file_path, exc)
        raise

    # try yaml first
    try:
        data = yaml.safe_load(raw_content)
    except yaml.YAMLError as yaml_err:
        logger.warning(
            "YAML parsing error in '%s': %s – attempting JSON fallback.",
            file_path,
            yaml_err,
        )
        # json fallback
        try:
            data = json.loads(raw_content)
            logger.info("File '%s' parsed successfully as JSON.", file_path)
        except json.JSONDecodeError as json_err:
            logger.error(
                "Failed to parse '%s' as either YAML or JSON: %s", file_path, json_err
            )
            # original YAML error for clarity
            raise yaml_err from None

    # handle empty file (safe_load returns None)
    if data is None:
        logger.warning(
            "Configuration file '%s' is empty. Treating as an empty mapping.",
            file_path,
        )
        data = {}

    # handle list at top level by wrapping it
    if isinstance(data, list):
        logger.info(
            "Configuration file '%s' contains a list at the top level. "
            "Wrapping it with '_list' key.",
            file_path,
        )
        return {"_list": data}

    # ensure the loaded data is a mapping
    if not isinstance(data, dict):
        logger.error(
            "Configuration file '%s' must contain either a mapping or a list "
            "at the top level (got %s).",
            file_path,
            type(data).__name__,
        )
        raise ValueError(
            f"Configuration file {file_path} must contain either a mapping "
            f"or a list at the top level, got {type(data).__name__}."
        )
    return data


def has_allow_from_file_attribute(cls: type) -> bool:
    """
    Check if a class has the _allow_from_file attribute set to True.
    Only return True if the attribute is defined directly on cls (i.e. not inherited)

    Args:
        cls: The class to check

    Returns:
        True if the class has the _allow_from_file attribute set to True, False otherwise
    """
    return vars(cls).get("_allow_from_file", False)
