from enum import IntEnum


class BaseIntEnum(IntEnum):
    def __str__(self):
        return self.name.lower()

    @classmethod
    def from_str(cls, string):
        try:
            return cls[string.upper()]
        except KeyError:
            valid_values = [member.name.lower() for member in cls]
            raise ValueError(
                f"Invalid {cls.__name__} value: '{string}'. "
                f"Valid values: {', '.join(valid_values)}"
            )
