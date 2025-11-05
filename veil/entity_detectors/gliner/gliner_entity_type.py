from veil.core.base_entity_type import EntityTypeBase


class GlinerEntityType(EntityTypeBase):
    """
    Minimal set of Gliner entity types we support in Veil. These models allow, in principle,
    any entity type, but we restrict these to our currently allowed types.
    """

    NAME = 1
    COMPANY = 2
    ADDRESS = 3

    @classmethod
    def aliases(cls) -> list[tuple[str, str]]:
        """Specify how model output labels will be mapped to our canonical types."""

        # gliner config accepts a list of labels that we will pass to the model as entity types to detect
        # we add here sample ones that we want to map to our canonical types
        return [
            ("PERSON", "NAME"),
            ("ORGANISATION", "COMPANY"),
            ("ORGANIZATION", "COMPANY"),
            ("STREET", "ADDRESS"),
            ("CITY", "ADDRESS"),
        ]
