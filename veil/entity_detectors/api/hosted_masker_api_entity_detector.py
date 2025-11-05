import logging

from veil.config.entity_detectors import HostedMaskerApiEntityDetectorConfig
from veil.entity_detectors.api.hosted_masker_api_entity_type import (
    HostedMaskerApiEntityType,
)
from veil.entity_detectors.api.masker_api_entity_detector import MaskerApiEntityDetector

logger = logging.getLogger(__name__)


class HostedMaskerApiEntityDetector(MaskerApiEntityDetector):
    """
    Anonymizer backed by a hosted Masker API. Wrapper around MaskerApiEntityDetector with
    specific entity types.
    """

    ENTITY_TYPES = {e for e in HostedMaskerApiEntityType}

    def __init__(self, cfg: HostedMaskerApiEntityDetectorConfig):
        """Initialize the detector with a configuration."""
        super().__init__(cfg)
