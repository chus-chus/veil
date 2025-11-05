from veil.core.base_entity_type import EntityTypeBase


class RegexEntityType(EntityTypeBase):
    """
    Available entity types to be detected. These are sample entities based on the provided example patterns.
    """

    DNI = 1
    CIF = 2
    NIE = 3
    NSS = 4
    EMAIL = 5
    PHONE = 6
    IBAN = 7
    IPV4 = 8
    IPV6 = 9

    @classmethod
    def aliases(cls) -> list[tuple[str, str]]:
        # Map alternative names to canonical enum member names
        return []
