"""
Advanced validators for Spanish identifiers.

This module implements validators that check not only the format but also
the control digits (checksums) for DNI, CIF, and NIE.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

from veil.entity_detectors.regex.regex_entity_type import RegexEntityType


class BaseRegexPatternValidator(ABC):
    """Base class for identifier validators."""

    @abstractmethod
    def validate(self, identifier: str) -> bool:
        """
        Validate an identifier.

        Args:
            identifier: String to validate

        Returns:
            bool: True if the identifier is valid
        """

    @abstractmethod
    def extract_clean_identifier(self, identifier: str) -> Optional[str]:
        """
        Extract and clean an identifier from text.

        Args:
            identifier: String that may contain the identifier

        Returns:
            Optional[str]: Clean identifier or None if invalid
        """


class DNIValidator(BaseRegexPatternValidator):
    """Validator for Spanish DNI with control-digit verification."""

    # Letter table for control-digit calculation
    CONTROL_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"

    def __init__(self):
        """Initialize the validator with the regex pattern."""
        self.pattern = re.compile(r"^(\d{8})([A-HJ-NP-TV-Z])$", re.IGNORECASE)

    def validate(self, identifier: str) -> bool:
        """
        Validate a DNI by checking format and control digit.

        Args:
            identifier: DNI to validate

        Returns:
            bool: True if the DNI is valid
        """
        # Trim spaces and convert to uppercase
        clean_dni = identifier.strip().upper().replace("-", "").replace(" ", "")

        # Verify format
        match = self.pattern.match(clean_dni)
        if not match:
            return False

        number_part = match.group(1)
        letter_part = match.group(2)

        # Compute control letter
        expected_letter = self._calculate_control_letter(number_part)

        return letter_part == expected_letter

    def _calculate_control_letter(self, number: str) -> str:
        """
        Compute the control letter for a DNI number.

        Args:
            number: Numeric part of the DNI

        Returns:
            str: Corresponding control letter
        """
        return self.CONTROL_LETTERS[int(number) % 23]

    def extract_clean_identifier(self, identifier: str) -> Optional[str]:
        """
        Extract and clean a DNI from text.

        Args:
            identifier: Text that may contain a DNI

        Returns:
            Optional[str]: Clean DNI or None if invalid
        """
        clean_dni = identifier.strip().upper().replace("-", "").replace(" ", "")

        if self.validate(clean_dni):
            return clean_dni

        return None


class CIFValidator(BaseRegexPatternValidator):
    """Validator for Spanish CIF with control-digit verification."""

    def __init__(self):
        """Initialize the validator with the regex pattern."""
        self.pattern = re.compile(r"^([A-HJNP-SW])(\d{7})([0-9A-J])$", re.IGNORECASE)

    def validate(self, identifier: str) -> bool:
        """
        Validate a CIF by checking format and control digit.

        Args:
            identifier: CIF to validate

        Returns:
            bool: True if the CIF is valid
        """
        # Trim spaces and convert to uppercase
        clean_cif = identifier.strip().upper().replace("-", "").replace(" ", "")

        # Verify format
        match = self.pattern.match(clean_cif)
        if not match:
            return False

        org_type = match.group(1)
        number_part = match.group(2)
        control_char = match.group(3)

        # Compute control digit
        expected_control = self._calculate_control_digit(org_type, number_part)

        return control_char == expected_control

    def _calculate_control_digit(self, org_type: str, number: str) -> str:
        """
        Compute the control digit for a CIF.

        Args:
            org_type: Letter for the organization type
            number: Numeric part of the CIF

        Returns:
            str: Corresponding control digit
        """
        # CIF control-digit calculation algorithm
        sum_a = 0
        sum_b = 0

        for i, digit in enumerate(number):
            if i % 2 == 0:  # Posiciones impares (1, 3, 5, 7)
                doubled = int(digit) * 2
                sum_a += doubled // 10 + doubled % 10
            else:  # Posiciones pares (2, 4, 6)
                sum_b += int(digit)

        total = sum_a + sum_b
        control_digit = (10 - (total % 10)) % 10

        # Some letters require a control letter instead of a number
        if org_type in "NPQRSW":
            control_letters = "JABCDEFGHI"
            return control_letters[control_digit]
        else:
            return str(control_digit)

    def extract_clean_identifier(self, identifier: str) -> Optional[str]:
        """
        Extract and clean a CIF from text.

        Args:
            identifier: Text that may contain a CIF

        Returns:
            Optional[str]: Clean CIF or None if invalid
        """
        clean_cif = identifier.strip().upper().replace("-", "").replace(" ", "")

        if self.validate(clean_cif):
            return clean_cif

        return None


class NIEValidator(BaseRegexPatternValidator):
    """Validator for Spanish NIE with control-digit verification."""

    # Letter table for control-digit calculation (same as DNI)
    CONTROL_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"

    def __init__(self):
        """Initialize the validator with the regex pattern."""
        self.pattern = re.compile(r"^([XYZ])(\d{7})([A-Z])$", re.IGNORECASE)

    def validate(self, identifier: str) -> bool:
        """
        Validate a NIE by checking format and control digit.

        Args:
            identifier: NIE to validate

        Returns:
            bool: True if the NIE is valid
        """
        # Trim spaces and convert to uppercase
        clean_nie = identifier.strip().upper().replace("-", "").replace(" ", "")

        # Verify format
        match = self.pattern.match(clean_nie)
        if not match:
            return False

        first_letter = match.group(1)
        number_part = match.group(2)
        control_letter = match.group(3)

        # Compute control letter
        expected_letter = self._calculate_control_letter(first_letter, number_part)

        return control_letter == expected_letter

    def _calculate_control_letter(self, first_letter: str, number: str) -> str:
        """
        Compute the control letter for a NIE.

        Args:
            first_letter: First letter of the NIE (X, Y, Z)
            number: Numeric part of the NIE

        Returns:
            str: Corresponding control letter
        """
        # Convert first letter to a number
        letter_to_number = {"X": "0", "Y": "1", "Z": "2"}
        full_number = letter_to_number[first_letter] + number

        # Calculate as if it were a DNI
        return self.CONTROL_LETTERS[int(full_number) % 23]

    def extract_clean_identifier(self, identifier: str) -> Optional[str]:
        """
        Extract and clean a NIE from text.

        Args:
            identifier: Text that may contain a NIE

        Returns:
            Optional[str]: Clean NIE or None if invalid
        """
        clean_nie = identifier.strip().upper().replace("-", "").replace(" ", "")

        if self.validate(clean_nie):
            return clean_nie

        return None


# Simple map of available validators (module level)
VALIDATORS: Dict[RegexEntityType, BaseRegexPatternValidator] = {
    RegexEntityType.DNI: DNIValidator(),
    RegexEntityType.CIF: CIFValidator(),
    RegexEntityType.NIE: NIEValidator(),
}


def get_validator(
    identifier_type: Union[str, RegexEntityType],
) -> Optional[BaseRegexPatternValidator]:
    if isinstance(identifier_type, RegexEntityType):
        return VALIDATORS.get(identifier_type)
    try:
        key = RegexEntityType.from_str(identifier_type.lower())
    except Exception:
        return None
    return VALIDATORS.get(key)


def validate_identifier(
    identifier_type: Union[str, RegexEntityType], identifier: str
) -> bool:
    validator = get_validator(identifier_type)
    return validator.validate(identifier) if validator else False


def get_supported_validator_types() -> list:
    return list(VALIDATORS.keys())
