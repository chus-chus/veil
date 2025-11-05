from veil.core.base_entity_type import EntityTypeBase


class SpacyEntityType(EntityTypeBase):
    """
    Available entity types to be detected.
    """

    NAME = 1
    COMPANY = 2
    ADDRESS = 3

    @classmethod
    def aliases(cls) -> list[tuple[str, str]]:
        # Map alternative names to canonical enum member names
        return [
            ("PER", "NAME"),
            ("ORG", "COMPANY"),
            ("LOC", "ADDRESS"),
        ]
