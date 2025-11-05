import copy
import json
import sys
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
)
from collections import defaultdict, deque
from dataclasses import MISSING, fields, make_dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, get_args, get_type_hints

from veil.config.core.base_poly_config import BasePolyConfig
from veil.config.utils import (
    get_all_subclasses,
    get_inner_type,
    has_allow_from_file_attribute,
    is_bool,
    is_composed_of_primitives,
    is_dict,
    is_list,
    is_optional,
    is_primitive_type,
    is_subclass,
    load_yaml_config,
    to_snake_case,
)
from veil.logger import init_logger

logger = init_logger(__name__)


def explode_dict(
    cls,
    config: Dict[str, Any],
    prefix: str = "",
    *,
    max_combinations: int = 10_000,
) -> List[Dict[str, Any]]:
    """
    Recursively explode a dictionary containing lists of values into a list of dictionaries
    representing all combinations (cartesian product), with optional prefix applied to keys.

    Args:
        config: Dictionary potentially containing lists to explode
        prefix: Prefix to apply to all top-level keys

    Example:
        Input: {'a': [1, 2], 'b': [3, 4]}, prefix='test_'
        Output: [{'test_a': 1, 'test_b': 3}, {'test_a': 1, 'test_b': 4},
                 {'test_a': 2, 'test_b': 3}, {'test_a': 2, 'test_b': 4}]

    NOTE:
        In deeply–nested configs with many lists, the number of combinations can grow
        exponentially. This method raises a ``ValueError`` if the number of combinations
        exceeds ``max_combinations``. Pass ``float('inf')`` to disable the limit.
    """

    # for guarding against combinatorial explosion
    combination_counter = [0]  # mutable counter in a closure

    def _increment_counter(n: int):
        """Increment the global combination counter and enforce the max limit."""
        if n == 0:
            return
        combination_counter[0] += n
        if combination_counter[0] > max_combinations:
            raise ValueError(
                "The number of generated configuration combinations ("
                f"{combination_counter[0]}) exceeds the allowed maximum of "
                f"{max_combinations}. Reduce list sizes or increase the limit "
                "to avoid combinatorial explosion."
            )

    def _resolve_prefix_for_data(
        cls, current_prefix: str, data: Dict[str, Any], strict: bool = True
    ) -> str:
        """Resolve the effective prefix for a dictionary possibly representing a BasePolyConfig.

        If the dictionary contains a "type" key, resolve the typed child name from
        `base_poly_children_types` using the stripped `current_prefix`. When `strict`
        is True, raise a ValueError if the type is invalid for the given prefix.
        """
        resolved_prefix = current_prefix
        if "type" in data:
            stripped_prefix = (
                current_prefix[:-1]
                if current_prefix and current_prefix[-1] == "_"
                else current_prefix
            )
            # remove a trailing "_type"
            if stripped_prefix.endswith("_type"):
                stripped_prefix = stripped_prefix[: -len("_type")]
            type_key = str(data["type"]).lower()
            if data["type"] is None or type_key in {"none", "null", ""}:
                return current_prefix
            typed_child_name = cls.base_poly_children_types.get(
                stripped_prefix, {}
            ).get(type_key)
            if typed_child_name:
                resolved_prefix = f"{typed_child_name}_"
            elif strict:
                valid = list(
                    cls.base_poly_children_types.get(stripped_prefix, {}).keys()
                )
                raise ValueError(
                    f"Invalid type '{data['type']}' for '{stripped_prefix}_type'. Valid types: {valid}"
                )
        return resolved_prefix

    def _categorize_dict_items(d: Dict[str, Any], current_prefix: str = "") -> tuple:
        """Categorize dictionary items into lists, dicts, and primitives."""
        list_keys = []
        list_values = []
        non_list_items = {}
        dict_items = {}

        for key, value in d.items():
            prefixed_key = f"{current_prefix}{key}" if current_prefix else key
            expected_type = getattr(cls, "__annotations__", {}).get(prefixed_key, None)
            is_literal_list = expected_type and is_list(expected_type)

            if isinstance(value, list) and len(value) > 0:
                # (only) if the dataclass declares the field as a List[...]
                # we treat the whole list as a single literal value and don't
                # explode it
                if is_literal_list:
                    non_list_items[key] = value
                else:
                    # will explode
                    if isinstance(value[0], dict):
                        # list of config dictionaries
                        list_keys.append(key)
                        list_values.append(value)
                    else:
                        # list of primitive values
                        list_keys.append(key)
                        list_values.append(value)
            elif isinstance(value, dict):
                dict_items[key] = value
            else:
                non_list_items[key] = value

        return list_keys, list_values, non_list_items, dict_items

    def _explode_dict_list(
        dict_list: List[Dict[str, Any]], level: int, current_prefix: str = ""
    ) -> List[Dict[str, Any]]:
        """Explode a list of dictionaries recursively."""
        exploded_configs = []
        for config in dict_list:
            # child will resolve its own prefix types if needed
            exploded = _explode_dict_recursive(config, level + 1, current_prefix)
            exploded_configs.extend(exploded)
        return exploded_configs

    def _generate_dict_combinations(
        dict_items: Dict[str, Dict[str, Any]], level: int, current_prefix: str = ""
    ) -> List[Dict[str, Any]]:
        """Generate all combinations from nested dictionaries."""
        dict_combinations = [{}]

        for key, nested_dict in dict_items.items():
            # Build prefix for nested dictionary - add current key to prefix chain
            nested_prefix = (
                f"{current_prefix}{key}_" if current_prefix or key else f"{key}_"
            )
            exploded_nested = _explode_dict_recursive(
                nested_dict, level + 1, nested_prefix
            )
            new_combinations = []

            for base_combo in dict_combinations:
                for nested_combo in exploded_nested:
                    new_combo = base_combo.copy()
                    new_combo[key] = nested_combo
                    new_combinations.append(new_combo)

            dict_combinations = new_combinations

        return dict_combinations

    def _combine_non_list_items(
        non_list_items: Dict[str, Any], dict_combinations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine non-list items with dictionary combinations."""
        result = []
        for dict_combo in dict_combinations:
            combined = non_list_items.copy()
            combined.update(dict_combo)
            result.append(combined)
        _increment_counter(len(result))
        return result

    def _generate_all_combinations(
        list_keys: List[str],
        list_values: List[List[Any]],
        non_list_items: Dict[str, Any],
        dict_combinations: List[Dict[str, Any]],
        level: int,
        current_prefix: str = "",
    ) -> List[Dict[str, Any]]:
        """Generate all combinations including list values."""
        # handle list of config dictionaries vs primitives
        processed_list_values = []
        for values in list_values:
            if values and isinstance(values[0], dict):
                # explode each config dict in the list
                processed_list_values.append(
                    _explode_dict_list(values, level, current_prefix)
                )
            else:
                # keep primitive values as-is
                processed_list_values.append(values)

        result = []
        for combination in product(*processed_list_values):
            for dict_combo in dict_combinations:
                new_config = non_list_items.copy()
                new_config.update(dict_combo)

                for key, value in zip(list_keys, combination):
                    new_config[key] = value

                result.append(new_config)

        _increment_counter(len(result))
        return result

    def _explode_dict_recursive(
        d: Dict[str, Any], level: int = 0, current_prefix: str = ""
    ) -> List[Dict[str, Any]]:
        """Recursively explode a dictionary into all combinations."""

        # resolve effective prefix (might be typed)
        effective_prefix = _resolve_prefix_for_data(
            cls, current_prefix=current_prefix, data=d, strict=True
        )

        list_keys, list_values, non_list_items, dict_items = _categorize_dict_items(
            d, effective_prefix
        )

        # generate combinations from nested dictionaries
        dict_combinations = _generate_dict_combinations(
            dict_items, level, effective_prefix
        )

        # if no lists found, just combine non-list items with dict combinations
        if not list_keys:
            return _combine_non_list_items(non_list_items, dict_combinations)

        # generate all combinations including lists
        return _generate_all_combinations(
            list_keys,
            list_values,
            non_list_items,
            dict_combinations,
            level,
            effective_prefix,
        )

    def _add_prefix_to_dict(cls, d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """Add prefix to all keys in a dictionary."""

        def _add_prefix_recursive(
            cls, data: Dict[str, Any], current_prefix: str = ""
        ) -> Dict[str, Any]:
            result = {}

            # resolve effective typed prefix once per node
            effective_prefix = _resolve_prefix_for_data(
                cls, current_prefix=current_prefix, data=data, strict=True
            )

            for key, value in data.items():
                # For the 'type' meta-key itself, keep the current (un-typed) prefix
                if "type" in data and key == "type":
                    prefixed_key = f"{current_prefix}{key}"
                else:
                    prefixed_key = f"{effective_prefix}{key}"

                if isinstance(value, dict):
                    # for nested dicts, recursively process with composed prefix
                    flattened = _add_prefix_recursive(cls, value, f"{prefixed_key}_")
                    result.update(flattened)
                else:
                    # leaf value - add it with the full prefix
                    result[prefixed_key] = value

            return result

        return _add_prefix_recursive(cls, d, prefix)

    def _handle_list_config(
        config: Dict[str, Any], prefix: str
    ) -> List[Dict[str, Any]]:
        """Handle special case where config has a '_list' key."""
        list_data = config["_list"]
        if not isinstance(list_data, list):
            return []

        all_exploded = []
        for item in list_data:
            if isinstance(item, dict):
                exploded = _explode_dict_recursive(item, current_prefix=prefix)
                all_exploded.extend(exploded)
            else:
                # non-dict items are wrapped
                all_exploded.append({"_value": item})

        prefixed = [_add_prefix_to_dict(cls, cfg, prefix) for cfg in all_exploded]
        _increment_counter(len(prefixed))
        return prefixed

    # handle special case where config has a '_list' key (from load_yaml_config)
    if isinstance(config, dict) and len(config) == 1 and "_list" in config:
        return _handle_list_config(config, prefix)

    # standard case: explode the config and add prefixes
    exploded_configs = _explode_dict_recursive(config, current_prefix=prefix)
    _increment_counter(len(exploded_configs))
    return [_add_prefix_to_dict(cls, cfg, prefix) for cfg in exploded_configs]


def topological_sort(dataclass_dependencies: dict) -> list:
    """Topological sort of dataclass dependencies.

    Returns:
        List of dataclass names in topological order.
    """
    in_degree = defaultdict(int)
    for cls, dependencies in dataclass_dependencies.items():
        for dep in dependencies:
            in_degree[dep] += 1

    zero_in_degree_classes = deque(
        [cls for cls in dataclass_dependencies if in_degree[cls] == 0]
    )
    sorted_classes = []

    while zero_in_degree_classes:
        cls = zero_in_degree_classes.popleft()
        sorted_classes.append(cls)
        for dep in dataclass_dependencies[cls]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                zero_in_degree_classes.append(dep)

    return sorted_classes


def overwrite_args_with_config(
    args: dict,
    config: dict,
    keys_to_file_field_names: Dict[str, str],
    default_values: dict,
    cli_provided_args: set,
):
    """Overwrite args with values from config.

    Args:
        args: The dictionary of arguments to overwrite. Must be flat.
        config: The dictionary of values to overwrite args with. Can be nested.
        keys_to_file_field_names: A dictionary mapping keys to the file field names that provided them.
        default_values: A dictionary of default values for args.
        cli_provided_args: A set with the argument names that were provided via CLI.
    """

    for key, value in config.items():
        if isinstance(value, dict):
            # a nested config object: we compose the prefix
            overwrite_args_with_config(
                args, value, keys_to_file_field_names, default_values, cli_provided_args
            )
            continue

        # Ignore keys that are not recognised by the FlatClass.
        if not hasattr(args, key):
            logger.warning(
                f"Arg {key} provided by {keys_to_file_field_names.get(key, 'unknown')} not found in supported args."
            )
            continue

        if key not in cli_provided_args:
            # TODO: verbosity level for this (f"Overwriting {key} with {value}")
            setattr(args, key, value)
        else:
            logger.warning(
                f"Arg {key} provided by {keys_to_file_field_names[key]} set via CLI. Skipped overwrite."
            )


def reconstruct_original_dataclass(self) -> Any:
    """
    This function is dynamically mapped to FlatClass as an instance method.
    Reconstructs the original dataclass from the flattened representation.
    """
    # skip all classes with default None and that have not been provided by the user
    classes_to_skip = set()
    for cls, dependencies in self.dataclass_dependencies.items():
        # did the user provide anything that belongs to this dataclass?
        sub_arg_provided = any(k.startswith(f"{cls}_") for k in self.provided_args)
        # fallback for base poly configs: to specify a poly class, one provides the type
        cls_type_arg = cls + "_type"

        # skip if the field defaults to None and the user did not provide it
        #   – For polymorphic configs: no <cls>_type
        #   – For regular dataclasses: no sub-field with the <cls>_ prefix
        if (
            cls in self.args_with_default_none
            and cls_type_arg not in self.provided_args
            and not sub_arg_provided
        ):
            classes_to_skip.add(cls)
            for dependency in dependencies:
                classes_to_skip.add(dependency)

    filtered_dependencies = {}
    for cls, dependencies in self.dataclass_dependencies.items():
        if cls not in classes_to_skip:
            filtered_dependencies[cls] = [
                dep for dep in dependencies if dep not in classes_to_skip
            ]

    # list of classes, from the most dependent to the least dependent
    sorted_classes = topological_sort(filtered_dependencies)

    instances = {}

    # iter over classes from least dependent to most
    for _cls in reversed(sorted_classes):
        args = {}
        # instantiate current class fields
        for prefixed_field_name, original_field_name, field_type in self.dataclass_args[
            _cls
        ]:
            if is_subclass(field_type, BasePolyConfig):
                # pick the instantiated child that matches the selected type
                config_type = getattr(self, f"{prefixed_field_name}_type")
                if config_type == "None":
                    args[original_field_name] = None
                else:
                    type_key = config_type.lower()
                    try:
                        child_node_name = self.base_poly_children_types[
                            prefixed_field_name
                        ][type_key]
                    except KeyError:
                        valid = list(
                            self.base_poly_children_types.get(
                                prefixed_field_name, {}
                            ).keys()
                        )
                        raise ValueError(
                            f"Invalid type '{config_type}' (key: {type_key}) for '{prefixed_field_name}_type'. Valid types: {valid}"
                        ) from None
                    args[original_field_name] = instances[child_node_name]
            # child dataclass has already been instantiated, so just assign it
            elif hasattr(field_type, "__dataclass_fields__"):
                # find the dependency name corresponding to this field's type.
                dependency_name = None
                for dep_name in self.dataclass_dependencies[_cls]:
                    dep_cls = self.dataclass_names_to_classes.get(dep_name)
                    if dep_cls is field_type:
                        dependency_name = dep_name
                        break

                if dependency_name and dependency_name in instances:
                    args[original_field_name] = instances[dependency_name]
                else:
                    if (
                        prefixed_field_name not in self.args_with_default_none
                        and dependency_name is None
                    ):
                        raise ValueError(
                            f"Class {_cls} has no dependency name and is not in args_with_default_none"
                        )
                    else:
                        # not been provided by the user and is None by default
                        args[original_field_name] = None
            # primitive type
            else:
                value = getattr(self, prefixed_field_name)
                if value is not MISSING and callable(value):
                    # to handle default factory values
                    value = value()
                args[original_field_name] = value

        instances[_cls] = self.dataclass_names_to_classes[_cls](**args)

    return instances[sorted_classes[0]]


def instantiate_from_args(cls, args, provided_args):
    """Instantiate a dataclass from a list of args and provided args."""
    return_clss = []
    for i, arg_instance in enumerate(args):
        _return_cls = cls(**vars(arg_instance))
        _return_cls.provided_args = provided_args[i]
        return_clss.append(_return_cls)
    return return_clss


def init_iterable_args(loaded_configs, cli_provided_args, list_fields):
    """Initialize iterable args."""

    def _init_iterable_args(arg_map, list_fields):
        return_map = {}
        for arg_name, arg_value in arg_map.items():
            # at this point, an iterable field contains raw values. Can be maps, lists, primitives...
            # we check what value they need to be converted to, i.e. poly configs, dataclasses, primitives, etc
            if arg_name in list_fields:
                return_iterable = []
                target_type = list_fields[arg_name]
                # we assume each raw value is a dict with config values
                if isinstance(target_type, type) and issubclass(
                    target_type, BasePolyConfig
                ):
                    # get all subclasses of the target type
                    subclasses = get_all_subclasses(target_type)
                    for raw_value in arg_value:
                        assert (
                            "type" in raw_value
                        ), f"Each raw value in an iterable of BasePolyConfigs must contain a 'type' key. Obtained '{raw_value}'"
                        is_match = False
                        # linear matching... do we assume there is a registry?
                        for subclass in subclasses:
                            if (
                                subclass.get_type().name.upper()
                                == raw_value["type"].upper()
                            ):
                                subclass_kwargs = {
                                    k: v for k, v in raw_value.items() if k != "type"
                                }
                                return_iterable.append(subclass(**subclass_kwargs))
                                is_match = True
                                break
                        assert (
                            is_match
                        ), f"No class found for type '{raw_value['type']}' in children of {target_type}"
                elif hasattr(target_type, "__dataclass_fields__"):
                    for raw_value in arg_value:
                        return_iterable.append(target_type(**raw_value))
                elif isinstance(target_type, type):
                    for raw_value in arg_value:
                        return_iterable.append(target_type(raw_value))
                else:
                    raise ValueError(f"Unsupported target type: {target_type}")
                return_map[arg_name] = return_iterable
            else:
                return_map[arg_name] = arg_value
        return return_map

    # loaded_configs is a dict of file_field_name -> list of configs
    final_loaded_configs = {}
    for file_field_name, configs in loaded_configs.items():
        tmp_configs = []
        for config in configs:
            tmp_config = _init_iterable_args(config, list_fields)
            tmp_configs.append(tmp_config)
        final_loaded_configs[file_field_name] = tmp_configs

    # cli_provided_args is a dict of arg_name -> value
    final_cli_args = _init_iterable_args(cli_provided_args, list_fields)

    return final_loaded_configs, final_cli_args


@classmethod
def create_from_cli_args(cls) -> Any:
    """
    Create dataclass instances from CLI arguments and config files.

    Returns:
        List[cls]: A list of instances of the dataclass, one for each combination of configs created.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    all_default_values = {}
    argnames_to_field_names = {}

    # build argument parser from dataclass fields
    for field in fields(cls):
        _add_field_to_parser(
            cls, field, parser, all_default_values, argnames_to_field_names
        )

    args = parser.parse_args()
    cli_provided_args = _get_cli_provided_args(argnames_to_field_names, args)

    # load and process config files
    loaded_configs = _load_config_files(cls, args)
    final_loaded_configs, final_cli_args = init_iterable_args(
        loaded_configs, cli_provided_args, cls.list_fields
    )

    # update args with processed CLI values
    for key, value in final_cli_args.items():
        setattr(args, key, value)

    # create all combinations of configs
    all_config_combinations, all_keys_to_file_field_names = _create_config_combinations(
        final_loaded_configs
    )

    # merge cli args with config combinations
    final_args, all_provided_args = _merge_args_with_configs(
        args,
        all_config_combinations,
        all_keys_to_file_field_names,
        all_default_values,
        final_cli_args,
    )

    return_clss = instantiate_from_args(cls, final_args, all_provided_args)

    return return_clss


def _add_field_to_parser(
    cls, field, parser, all_default_values, argnames_to_field_names
):
    """Add a single dataclass field as an argument to the parser."""

    if not field.init:
        return

    nargs = None
    action = None
    field_type = field.type

    # extract metadata
    help_text = cls.metadata_mapping[field.name].get("help", None)
    argname = cls.metadata_mapping[field.name].get("argname", None)

    # validate argname uniqueness
    if argname in argnames_to_field_names:
        raise ValueError(
            f"Cannot have multiple fields with the same argname: {argname} already exists for field {argnames_to_field_names[argname]}"
        )
    elif argname is not None:
        argnames_to_field_names[argname] = field.name

    # handle optional types
    is_field_optional = is_optional(field.type)
    if is_field_optional:
        field_type = get_inner_type(field.type)

    # configure type-specific parameters
    if is_list(field_type):
        inner_type = get_args(field_type)[0]
        if is_primitive_type(inner_type):
            # --ids 1 2 3
            nargs = "+"
            # inner primitive for conversion
            field_type = inner_type
        elif is_subclass(inner_type, BasePolyConfig) or hasattr(
            inner_type, "__dataclass_fields__"
        ):
            # --layers '[{"type": "conv", "kernel": 3}, {"type": "relu"}]'
            field_type = json.loads
        else:
            raise TypeError(
                f"Unsupported list element type '{inner_type}'. "
                "Only primitives, dataclasses and BasePolyConfig subclasses are allowed."
            )
    elif is_dict(field_type):
        assert is_composed_of_primitives(field_type)
        field_type = json.loads
    elif is_bool(field_type):
        action = BooleanOptionalAction

    # build argument parameters
    arg_params = {
        "type": field_type,
        "action": action,
        "help": help_text,
    }

    # handle default values
    if field.default is not MISSING:
        value = field.default
        if callable(value):
            value = value()
        arg_params["default"] = value
        all_default_values[field.name] = value
    elif field.default_factory is not MISSING:
        arg_params["default"] = field.default_factory()
        all_default_values[field.name] = field.default_factory()
    else:
        all_default_values[field.name] = object()  # sentinel value
        if is_field_optional:
            arg_params["default"] = None
        else:
            arg_params["required"] = True

    if nargs:
        arg_params["nargs"] = nargs

    # add argument to parser
    cli_arg_name = field.name.replace("_", "-") if argname is None else argname
    parser.add_argument(f"--{cli_arg_name}", dest=field.name, **arg_params)


def _get_cli_provided_args(argnames_to_field_names, parsed_args) -> Dict[str, Any]:
    """Determine which arguments were explicitly provided via CLI and capture their values.

    Supports the following forms:
        1. --arg=value         -> value is "value"
        2. --arg value         -> value is "value"
        3. --flag              -> value is True (boolean flag)
    """

    cli_provided_args: Dict[str, Any] = {}

    argv = sys.argv
    idx = 1  # skip program name
    while idx < len(argv):
        token = argv[idx]

        # we only care about long-form options that start with "--"
        if not token.startswith("--"):
            idx += 1
            continue

        # strip leading dashes
        option = token[2:]

        # Case 1: --arg=value
        if "=" in option:
            arg_name, arg_value = option.split("=", 1)
        else:
            arg_name = option
            # Case 2 or 3: value may be the next token unless it's another flag
            if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
                arg_value = argv[idx + 1]
                idx += 1  # skip the value token on next iteration
            else:
                # Case 3: boolean flag with no explicit value
                arg_value = True

        # Map to dataclass field name if argname alias is present
        if arg_name in argnames_to_field_names:
            field_key = argnames_to_field_names[arg_name]
        else:
            # standard conversion: kebab-case -> snake_case
            field_key = arg_name.replace("-", "_")

        if hasattr(parsed_args, field_key):
            cli_provided_args[field_key] = getattr(parsed_args, field_key)

        idx += 1

    return cli_provided_args


def _load_config_files(cls, args):
    """Load and process all config files specified in arguments."""
    loaded_configs: Dict[str, List[Dict[str, Any]]] = {}

    for file_field_name in cls.dataclass_file_fields.values():
        file_path = getattr(args, file_field_name, None)
        if not file_path:
            continue

        file_config = load_yaml_config(file_path)

        # determine prefix for this config file
        # cli args are provided without the root class prefix except for the root _from_file arg
        name_of_class_for_file = file_field_name.replace("_from_file", "").replace(
            "-", ""
        )
        if name_of_class_for_file == cls.root_dataclass_name:
            prefix = ""
        else:
            prefix = f"{name_of_class_for_file}_"

        loaded_configs[file_field_name] = explode_dict(cls, file_config, prefix)

    # log config loading summary
    total_configs = 0
    for file_field_name, configs in loaded_configs.items():
        n_configs = len(configs)
        logger.info(
            f"File field name '{file_field_name}' expanded to {n_configs} config{'' if n_configs == 1 else 's'}."
        )
        total_configs += n_configs

    return loaded_configs


def _create_config_combinations(loaded_configs):
    """Create cartesian product of all loaded configs."""
    all_config_combinations = []
    all_keys_to_file_field_names: List[Dict[str, str]] = []

    if not loaded_configs:
        return all_config_combinations, all_keys_to_file_field_names

    # get config lists for cartesian product
    config_lists = list(loaded_configs.values())
    file_field_names = list(loaded_configs.keys())

    # generate all combinations
    # loaded config dicts are already flattened, so we just need to combine them
    # i.e. {a: [1, 2], b: [3, 4]} -> [{1, 3}, {1, 4}, {2, 3}, {2, 4}] (numbers represent flat dicts)
    for combination in product(*config_lists):
        combined_config = {}
        params_to_files = {}

        for config, current_file_field_name in zip(combination, file_field_names):
            # check for conflicts between configs
            for key, value in config.items():
                if key in combined_config:
                    raise ValueError(
                        f"Arg {key} provided by {current_file_field_name} is also set by {params_to_files[key]}."
                    )
                combined_config[key] = value
                params_to_files[key] = current_file_field_name

        all_config_combinations.append(combined_config)
        all_keys_to_file_field_names.append(params_to_files)

    logger.info(f"Created {len(all_config_combinations)} total config combinations.")
    logger.info("---")

    return all_config_combinations, all_keys_to_file_field_names


def _merge_args_with_configs(
    args,
    all_config_combinations,
    all_keys_to_file_field_names,
    all_default_values,
    cli_provided_args,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Merge cli arguments with all config combinations.

    Returns:
        list of all flatclass args and list of user-provided args for each config combination

    Args:
        args: The dictionary of arguments to overwrite. Must be flat.
        all_config_combinations: A list of all combinations of configs.
        all_keys_to_file_field_names: A list of dictionaries mapping keys to the file field names that provided them.
    """
    all_provided_args: List[Dict[str, Any]] = []

    if not all_config_combinations:
        return [args], [cli_provided_args]

    final_args = []
    for config, keys_to_file_field_names in zip(
        all_config_combinations, all_keys_to_file_field_names
    ):
        _provided_args_in_config = {
            **config,
            **cli_provided_args,
        }  # collision prevention is done prior to this
        all_provided_args.append(_provided_args_in_config)

        args_copy = copy.deepcopy(args)
        overwrite_args_with_config(
            args_copy,
            config,
            keys_to_file_field_names,
            all_default_values,
            cli_provided_args,
        )
        final_args.append(args_copy)

    return final_args, all_provided_args


def get_config_class_by_type_name(config_class: Any, type_name: str) -> Any:
    for subclass in get_all_subclasses(config_class):
        if subclass.get_type().name.upper() == type_name.upper():
            return subclass

    raise ValueError(f"Config class with name {type_name} not found.")


def _initialize_dataclass_state():
    """Initialize collections for tracking dataclass metadata during flattening."""
    return {
        "fields_with_defaults": [],
        "fields_without_defaults": [],
        "dataclass_args": defaultdict(list),
        "dataclass_dependencies": defaultdict(
            list
        ),  # maps unique nested dataclass names to their uniquely named dependencies
        "metadata_mapping": {},
        "file_fields": {},  # maps dataclass to its file field name
        "names_to_classes": {},  # maps unique nested dataclass names to their corresponding dataclass class
        "base_poly_children": {},  # maps unique (by name) base poly configs to {children names: children classes} (Dict[str, Dict[str, Any]])
        "base_poly_children_types": {},  # maps unique (by name) base poly configs to {children types: children names} (Dict[str, Dict[str, str]])
        "args_with_default_none": set(),  # the set of args that have a default value of None
        "list_fields": {},  # maps list field (prefixed) names to their inner types
    }


def _add_file_argument(state, target_cls, file_field_name: str):
    """Add a '_from_file' cli argument for the given dataclass."""
    state["fields_with_defaults"].append((file_field_name, Optional[str], None))
    state["metadata_mapping"][file_field_name] = {
        "help": f"Path to YAML/JSON configuration file for {target_cls.__name__}."
    }
    state["file_fields"][target_cls] = file_field_name


def _get_field_type_info(field):
    """Extract type information from a dataclass field."""
    if is_optional(field.type):
        return get_inner_type(field.type), True
    return field.type, False


def _get_default_value_for_poly_field(field, field_type):
    """Get the default value for a polymorphic config field."""
    if field.default_factory is not MISSING:
        return str(field.default_factory().get_type())
    elif field.default is not MISSING:
        if field.default is None:
            return "None"
        return str(field.default.get_type())
    else:
        raise ValueError(
            f"Field {field.name} of type {field_type} must have a default or default_factory"
        )


def _handle_polymorphic_config_field(
    state, field, field_type, prefixed_name, prefixed_input_dataclass, prefix
):
    """Process a field that is a BasePolyConfig subclass."""
    state["dataclass_args"][prefixed_input_dataclass].append(
        (prefixed_name, field.name, field_type)
    )
    state["base_poly_children"][prefixed_name] = {}
    state["base_poly_children_types"][prefixed_name] = {}

    type_field_name = f"{prefixed_name}_type"
    default_value = _get_default_value_for_poly_field(field, field_type)

    state["fields_with_defaults"].append(
        (type_field_name, type(default_value), default_value)
    )
    state["metadata_mapping"][type_field_name] = field.metadata

    # add _from_file to base poly config if attribute is set
    if has_allow_from_file_attribute(field_type):
        file_field_name = f"{prefixed_name}_from_file"
        _add_file_argument(state, field_type, file_field_name)

    # process all subclasses of the polymorphic config
    assert hasattr(field_type, "__dataclass_fields__")
    for subclass in get_all_subclasses(field_type):
        type_key = subclass.get_type().name.lower()
        child_node_name = f"{prefix}{to_snake_case(type_key)}_{field.name}"
        # map the child node name to the subclass
        state["base_poly_children"][prefixed_name][child_node_name] = subclass
        # map type -> child node name
        state["base_poly_children_types"][prefixed_name][type_key] = child_node_name
        # ensure parent depends on this child node name
        state["dataclass_dependencies"][prefixed_input_dataclass].append(
            child_node_name
        )

        _process_single_dataclass(state, subclass, f"{child_node_name}_")


def _handle_nested_dataclass_field(
    state, field, field_type, prefixed_name, prefixed_input_dataclass
):
    """Process a field that is a nested dataclass."""
    dependency_name = prefixed_name
    state["dataclass_dependencies"][prefixed_input_dataclass].append(dependency_name)
    state["dataclass_args"][prefixed_input_dataclass].append(
        (prefixed_name, field.name, field_type)
    )
    _process_single_dataclass(state, field_type, f"{prefixed_name}_")


def _handle_primitive_field(
    state, field, field_type, prefixed_name, prefixed_input_dataclass
):
    """Process a field that is a primitive type."""
    field_default = field.default if field.default is not MISSING else MISSING
    field_default_factory = (
        field.default_factory if field.default_factory is not MISSING else MISSING
    )

    if field_default is not MISSING:
        state["fields_with_defaults"].append((prefixed_name, field_type, field_default))
    elif field_default_factory is not MISSING:
        state["fields_with_defaults"].append(
            (prefixed_name, field_type, field_default_factory)
        )
    else:
        state["fields_without_defaults"].append((prefixed_name, field_type))

    state["dataclass_args"][prefixed_input_dataclass].append(
        (prefixed_name, field.name, field_type)
    )
    state["metadata_mapping"][prefixed_name] = field.metadata

    # a list can contain poly configs or nested dataclasses
    if is_list(field_type):
        inner_type = get_args(field_type)[0]
        state["list_fields"][prefixed_name] = inner_type


def _process_single_dataclass(state, input_dataclass, prefix=""):
    """Process a single dataclass, flattening its fields and handling special cases."""
    prefixed_class_name = (
        f"{prefix[:-1]}" if prefix else f"{to_snake_case(input_dataclass.__name__)}"
    )

    # initialize dependency tracking for this dataclass
    _ = state["dataclass_dependencies"][prefixed_class_name]
    state["names_to_classes"][prefixed_class_name] = input_dataclass

    # add _from_file argument if attribute is set
    if has_allow_from_file_attribute(input_dataclass):
        file_field_name = (
            f"{prefix}from_file"
            if prefix
            else f"{to_snake_case(input_dataclass.__name__)}_from_file"
        )
        _add_file_argument(state, input_dataclass, file_field_name)

    # Resolve forward references in annotations (e.g., when using future annotations)
    try:
        resolved_hints = get_type_hints(
            input_dataclass, include_extras=True
        )  # py>=3.11
    except TypeError:
        # Fallback for older Python versions without include_extras
        resolved_hints = get_type_hints(input_dataclass)

    # process each field in the dataclass
    for field in fields(input_dataclass):
        prefixed_name = f"{prefix}{field.name}"
        # Prefer resolved type hints to avoid string annotations like 'str'
        annotated_type = resolved_hints.get(field.name, field.type)
        # Unwrap Optional[T] if present
        field_type = (
            get_inner_type(annotated_type)
            if is_optional(annotated_type)
            else annotated_type
        )

        if field.default is None:
            state["args_with_default_none"].add(prefixed_name)

        if is_subclass(field_type, BasePolyConfig):
            _handle_polymorphic_config_field(
                state, field, field_type, prefixed_name, prefixed_class_name, prefix
            )
        elif hasattr(field_type, "__dataclass_fields__"):
            _handle_nested_dataclass_field(
                state, field, field_type, prefixed_name, prefixed_class_name
            )
        else:
            _handle_primitive_field(
                state, field, field_type, prefixed_name, prefixed_class_name
            )


def _create_flat_class_type(state):
    """Create the final flattened dataclass type with all metadata attached."""
    all_fields = state["fields_without_defaults"] + state["fields_with_defaults"]
    flat_class = make_dataclass("FlatClass", all_fields)

    # attach metadata to the class
    flat_class.dataclass_args = state["dataclass_args"]
    flat_class.dataclass_dependencies = state["dataclass_dependencies"]
    flat_class.dataclass_names_to_classes = state["names_to_classes"]
    flat_class.metadata_mapping = state["metadata_mapping"]
    flat_class.dataclass_file_fields = state["file_fields"]
    flat_class.base_poly_children = state["base_poly_children"]
    flat_class.base_poly_children_types = state["base_poly_children_types"]
    flat_class.args_with_default_none = state["args_with_default_none"]
    flat_class.list_fields = state["list_fields"]
    return flat_class


def create_flat_dataclass(input_dataclass: Any) -> Any:
    """
    Creates a new FlatClass type by recursively flattening the input dataclass.
    This allows for easy parsing of command line arguments along with storing/loading the configuration to/from a file.
    """
    state = _initialize_dataclass_state()
    _process_single_dataclass(state, input_dataclass)

    flat_class = _create_flat_class_type(state)
    flat_class.root_dataclass_name = to_snake_case(input_dataclass.__name__)

    # attach helper methods
    flat_class.reconstruct_original_dataclass = reconstruct_original_dataclass
    flat_class.create_from_cli_args = create_from_cli_args

    return flat_class
