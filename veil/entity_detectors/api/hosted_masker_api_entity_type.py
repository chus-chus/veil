from veil.core.base_entity_type import EntityTypeBase


class HostedMaskerApiEntityType(EntityTypeBase):
    """
    Available entity types to be detected by the Masker API.
    """

    NAME = 1
    ADDRESS = 2
    COMPANY = 3

    @classmethod
    def aliases(cls) -> list[tuple[str, str]]:
        return [
            # just samples
            ("STREET", "ADDRESS"),
            ("CITY", "ADDRESS"),
            ("COMPANY_NAME", "COMPANY"),
            ("PERSON", "NAME"),
        ]
