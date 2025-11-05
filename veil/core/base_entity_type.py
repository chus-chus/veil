from veil.core.base_int_enum import BaseIntEnum


class EntityTypeBase(BaseIntEnum):
    """
    Marker base-class for all detector-specific entity-type enums.
    """

    @classmethod
    def aliases(cls) -> dict[str, str]:
        """Return a mapping of alternative names to canonical enum member names.

        Subclasses can override to provide synonyms, in any language or style.
        Keys and values are interpreted case-insensitively; the canonical value
        must be the enum member name defined in the subclass.
        """
        return {}

    def __init_subclass__(cls, **kwargs):  # type: ignore[override]
        # Ensure subclass initialization proceeds
        super().__init_subclass__(**kwargs)
        # Validate that all alias targets refer to valid enum member names
        try:
            alias_obj = cls.aliases() or {}
        except Exception as e:  # pragma: no cover
            # If subclass' aliases() misbehaves, surface clearly
            raise ValueError(
                f"{cls.__name__}.aliases() raised an exception: {e}"
            ) from e

        # Normalise into iterable of (src, target) pairs to preserve potential duplicates
        if isinstance(alias_obj, dict):
            alias_pairs = list(alias_obj.items())
        elif isinstance(alias_obj, (list, tuple)):
            alias_pairs = list(alias_obj)
        else:
            raise ValueError(
                f"{cls.__name__}.aliases() must return a dict or a list/tuple of pairs. Got: {type(alias_obj)}"
            )

        if not alias_pairs:
            return

        # Collect valid canonical names (case-insensitive compare)
        try:
            valid_names_upper = {m.name.upper() for m in cls}  # type: ignore[operator]
        except Exception as e:  # pragma: no cover
            raise ValueError(
                f"Failed to enumerate members of {cls.__name__} for alias validation: {e}"
            ) from e

        for pair in alias_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(
                    f"{cls.__name__}.aliases() must return 2-tuples; got invalid entry: {pair}"
                )
            src, target = pair
            target_upper = str(target).upper()
            if target_upper not in valid_names_upper:
                raise ValueError(
                    f"{cls.__name__}.aliases() maps '{src}' -> '{target}', "
                    f"but '{target}' is not a defined member of {cls.__name__}"
                )

        # Within-subclass duplicate/conflict checks early at definition time
        # Build canonical name set
        canonical_names_upper: set[str] = set()
        try:
            for member in cls:  # type: ignore[operator]
                name = getattr(member, "name", None)
                if isinstance(name, str):
                    canonical_names_upper.add(name.upper())
        except Exception:
            pass

        seen_alias_targets: dict[str, str] = {}
        for src, target in alias_pairs:
            src_upper = str(src).upper()
            tgt_upper = str(target).upper()
            # Prevent aliasing a canonical token to a different canonical
            if src_upper in canonical_names_upper and src_upper != tgt_upper:
                raise ValueError(
                    f"{cls.__name__}.aliases() attempts to remap canonical name '{src_upper}' to '{tgt_upper}', which is not allowed."
                )
            # Prevent multiple different targets for the same alias within this subclass
            if (
                src_upper in seen_alias_targets
                and seen_alias_targets[src_upper] != tgt_upper
            ):
                raise ValueError(
                    f"Conflicting aliases within {cls.__name__}: '{src_upper}' -> '{tgt_upper}' conflicts with existing mapping '{src_upper}' -> '{seen_alias_targets[src_upper]}'"
                )
            seen_alias_targets[src_upper] = tgt_upper

        # Cross-subclass conflict detection at definition time
        # Maintain a global registry of alias -> (canonical, subclass)
        try:
            registry = getattr(EntityTypeBase, "_global_alias_registry")
        except AttributeError:
            registry = {}
            setattr(EntityTypeBase, "_global_alias_registry", registry)

        for pair in alias_pairs:
            src, target = pair
            src_upper = str(src).upper()
            tgt_upper = str(target).upper()
            if src_upper in registry:
                prev_tgt, prev_cls = registry[src_upper]
                if prev_tgt != tgt_upper and prev_cls != cls.__name__:
                    raise ValueError(
                        f"Alias collision for '{src_upper}': {prev_cls} maps to '{prev_tgt}', "
                        f"but {cls.__name__} maps to '{tgt_upper}'. Aliases across subclasses must be consistent."
                    )
            else:
                registry[src_upper] = (tgt_upper, cls.__name__)

    @classmethod
    def _build_alias_map_for_subclass(cls) -> dict[str, str]:
        """Build upper-cased alias map for this subclass, including canonical names."""
        mapping: dict[str, str] = {}
        canonical_names_upper: set[str] = set()

        # Inject canonical names as identity mappings
        try:
            for member in cls:  # type: ignore[operator]
                name = getattr(member, "name", None)
                if isinstance(name, str):
                    up = name.upper()
                    mapping[up] = up
                    canonical_names_upper.add(up)
        except Exception:
            # If iteration fails (shouldn't for Enum subclasses), skip canonical injection
            pass

        # Now add aliases with conflict checks within the same subclass
        # Iterate alias pairs while preserving duplicates when provided as list/tuple
        alias_obj = cls.aliases() or {}
        pairs: list[tuple[str, str]]
        if isinstance(alias_obj, dict):
            pairs = [(str(k), str(v)) for k, v in alias_obj.items()]
        elif isinstance(alias_obj, (list, tuple)):
            pairs = [(str(a), str(b)) for a, b in alias_obj]  # type: ignore[misc]
        else:
            raise ValueError(
                f"{cls.__name__}.aliases() must return a dict or a list/tuple of pairs. Got: {type(alias_obj)}"
            )

        try:
            for raw_src, raw_target in pairs:
                src = raw_src.upper()
                tgt = raw_target.upper()

                # Prevent aliasing a canonical token to a different canonical name
                if src in canonical_names_upper and src != tgt:
                    raise ValueError(
                        f"{cls.__name__}.aliases() attempts to remap canonical name '{src}' "
                        f"to '{tgt}', which is not allowed."
                    )

                # Prevent multiple different targets for the same alias within the subclass
                if src in mapping and mapping[src] != tgt:
                    raise ValueError(
                        f"Conflicting aliases within {cls.__name__}: '{src}' -> '{tgt}' "
                        f"conflicts with existing mapping '{src}' -> '{mapping[src]}'"
                    )

                mapping[src] = tgt
        except Exception:
            raise
        return mapping

    @classmethod
    def global_alias_map(cls) -> dict[str, str]:
        """Aggregate alias mappings across all known `EntityTypeBase` subclasses.

        Note: Only subclasses that have been imported will be included.
        """
        mapping: dict[str, str] = {}
        provenance: dict[str, str] = {}

        for sub in cls.__subclasses__():
            try:
                sub_map = sub._build_alias_map_for_subclass()  # type: ignore[attr-defined]
            except Exception as e:
                # Surface errors from a specific subclass directly
                raise

            for alias_key, canonical_value in (sub_map or {}).items():
                if alias_key not in mapping:
                    mapping[alias_key] = canonical_value
                    provenance[alias_key] = sub.__name__
                    continue

                if mapping[alias_key] != canonical_value:
                    src_sub = provenance.get(alias_key, "<unknown>")
                    raise ValueError(
                        f"Alias collision for '{alias_key}': {src_sub} maps to '{mapping[alias_key]}', "
                        f"but {sub.__name__} maps to '{canonical_value}'. Aliases across subclasses must be consistent."
                    )

        return mapping
