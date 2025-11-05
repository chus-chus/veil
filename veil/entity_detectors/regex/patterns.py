"""
Regex patterns for Spanish sensitive identifiers.
"""

import re
from typing import Dict, List, Tuple

from veil.entity_detectors.regex.regex_entity_type import RegexEntityType

REGEX_PATTERNS: Dict[RegexEntityType, Dict[str, object]] = {
    RegexEntityType.DNI: {
        "pattern": r"\b\d{1,2}\.?\d{3}\.?\d{3}[-\s]?[A-HJ-NP-TV-Za-hj-np-tv-z]\b",
        "confidence": 0.95,
        "description": "Spanish National Identity Document (flexible: dots, hyphen, spaces, lowercase)",
    },
    RegexEntityType.CIF: {
        "pattern": r"\b[A-HJNP-SWa-hjnp-sw][-\s]?\d{2}\.?\d{3}\.?\d{3}[0-9A-Ja-j]?\b",
        "confidence": 0.90,
        "description": "Spanish Tax Identification Code (letter + 8 digits, flexible: dots, hyphen, spaces, lowercase)",
    },
    RegexEntityType.NIE: {
        "pattern": r"\b[XYZxyz][-\s]?\d{1,2}\.?\d{3}\.?\d{3}[-\s]?[A-Za-z]\b",
        "confidence": 0.92,
        "description": "Spanish Foreigner Identification Number (flexible: dots, hyphen, spaces, lowercase)",
    },
    RegexEntityType.NSS: {
        "pattern": r"\b\d{2}[-\s/.]{0,1}\d{10}|\b\d{2}[-\s/.]{0,1}\d{8}[-\s/.]{0,1}\d{2}\b",
        "confidence": 0.85,
        "description": "Spanish Social Security Number (12 digits, flexible: hyphen, spaces, dots, slashes)",
    },
    RegexEntityType.EMAIL: {
        "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "confidence": 0.80,
        "description": "Email address",
    },
    RegexEntityType.PHONE: {
        "pattern": r"(?:\+34|0034)[-\s]?\d{3}[-\s]\d{3}[-\s]\d{3}|\+34[-\s]?\d{9}|\b[6789]\d{2}(?:[-\s.]?\d{2}){3}(?:[-\s.]?\d{2})?\b|\b[6789]\d{2}[-\s.]\d{3}[-\s.]\d{3}\b",
        "confidence": 0.88,
        "description": "Spanish mobile phone (9 digits, ultra-flexible: +34 612 345 678, +34612345678, 612345678, 612 345 678, 612-345-678, 612.345.678)",
    },
    RegexEntityType.IBAN: {
        "pattern": r"\b[A-Za-z]{2}\d{2}(?:[-\s./]?[A-Za-z0-9]{4}){3,6}(?:[-\s./]?[A-Za-z0-9]{1,4})?\b",
        "confidence": 0.90,
        "description": "International Bank Account Number (country code + control digits + account, flexible: spaces, hyphens, dots, slashes)",
    },
    RegexEntityType.IPV4: {
        "pattern": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        "confidence": 0.88,
        "description": "IPv4 address",
    },
    RegexEntityType.IPV6: {
        "pattern": r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|(?:[0-9a-fA-F]{1,4}:)*::(?:[0-9a-fA-F]{1,4}:)*[0-9a-fA-F]{1,4}\b",
        "confidence": 0.85,
        "description": "IPv6 address",
    },
}

# Module-level compilation (adds 'compiled' key to each meta)
for _, _meta in REGEX_PATTERNS.items():
    _meta["compiled"] = re.compile(_meta["pattern"], re.IGNORECASE | re.MULTILINE)


def get_pattern_meta(entity_type: RegexEntityType) -> Dict[str, object]:
    """Get the metadata dictionary for an entity type."""
    return REGEX_PATTERNS[entity_type]


def get_all_pattern_meta() -> Dict[RegexEntityType, Dict[str, object]]:
    """Copy of the pattern mapping."""
    return {k: v.copy() for k, v in REGEX_PATTERNS.items()}


def validate_pattern(entity_type: RegexEntityType, test_string: str) -> bool:
    """Validate whether a string exactly matches the pattern for the given type."""
    meta = get_pattern_meta(entity_type)
    return meta["compiled"].fullmatch(test_string) is not None


def find_matches(text: str, entity_type: RegexEntityType) -> List[Tuple[int, int, str]]:
    """Find matches for a specific entity type."""
    meta = get_pattern_meta(entity_type)
    return [(m.start(), m.end(), m.group()) for m in meta["compiled"].finditer(text)]
