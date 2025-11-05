from dataclasses import field
from typing import Dict, List

from veil.config.core.base_poly_config import BasePolyConfig
from veil.config.core.frozen_dataclass import frozen_dataclass
from veil.core.enums.entity_detector_type import EntityDetectorType


@frozen_dataclass
class BaseEntityDetectorConfig(BasePolyConfig):
    """Marker base for entity detector configuration objects."""

    # Priority per canonical entity type name. Lower number = higher priority (0 is highest).
    # When multiple detectors produce overlapping spans, the span with the
    # lowest priority value for its entity type will be preferred.
    priority: Dict[str, int] = field(
        default_factory=dict,
        metadata={
            "help": "Map canonical entity type name -> integer priority. Lower wins; 0 is highest.",
        },
    )

    # Detector hierarchy position: 0 is highest precedence across components.
    hierarchy_position: int = field(
        default=0,
        metadata={
            "help": "Detector-level hierarchy position. 0 is highest precedence; larger = lower.",
        },
    )

    @classmethod
    def get_type(cls):
        raise NotImplementedError


@frozen_dataclass
class RegexEntityDetectorConfig(BaseEntityDetectorConfig):
    """Configuration wrapper for Veil’s RegexEntityDetector engine."""

    enable_validation: bool = field(
        default=True, metadata={"help": "Enable checksum validation where available"}
    )
    min_confidence: float = field(
        default=0.0, metadata={"help": "Minimum confidence threshold to keep entity"}
    )
    preserve_format: bool = field(
        default=True,
        metadata={"help": "Preserve original spacing/punctuation where possible"},
    )
    case_sensitive: bool = field(
        default=False, metadata={"help": "Case-sensitive regex search"}
    )

    @classmethod
    def get_type(cls):
        return EntityDetectorType.REGEX


@frozen_dataclass
class MaskerApiEntityDetectorConfig(BaseEntityDetectorConfig):
    """Marker base for masking-API entity detector configuration objects."""

    api_url: str = field(
        default="",
        metadata={"help": "URL of the hosted Masker API."},
    )
    headers: Dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Headers to send to the hosted Masker API."},
    )
    model: str = field(
        default="",
        metadata={"help": "Model to use for the hosted Masker API."},
    )
    system_prompt: str = field(
        default="You are a helpful assistant that masks sensitive information.",
        metadata={"help": "System prompt to send to the hosted Masker API."},
    )
    max_tokens: int = field(
        default=4000,
        metadata={"help": "Maximum number of tokens to generate."},
    )
    top_p: float = field(
        default=1,
        metadata={"help": "Top-p value for the hosted Masker API."},
    )
    top_k: int = field(
        default=40,
        metadata={"help": "Top-k value for the hosted Masker API."},
    )
    presence_penalty: float = field(
        default=0,
        metadata={"help": "Presence penalty for the hosted Masker API."},
    )
    frequency_penalty: float = field(
        default=0,
        metadata={"help": "Frequency penalty for the hosted Masker API."},
    )
    temperature: float = field(
        default=0.6,
        metadata={"help": "Temperature for the hosted Masker API."},
    )
    timeout: float = field(
        default=30,
        metadata={"help": "Timeout for the hosted Masker API."},
    )
    retries: int = field(
        default=2,
        metadata={
            "help": "Number of retries on network errors or suspected truncation."
        },
    )
    retry_backoff_base: float = field(
        default=0.5,
        metadata={"help": "Base seconds for exponential backoff between retries."},
    )
    retry_on_truncation: bool = field(
        default=True,
        metadata={"help": "Retry when truncation is detected in the response."},
    )
    chunk_on_truncation: bool = field(
        default=True,
        metadata={"help": "Fallback to chunking the input if truncation persists."},
    )
    chunk_char_limit: int = field(
        default=4000,
        metadata={
            "help": "Approximate maximum characters per chunk when chunking input."
        },
    )
    truncation_min_fraction: float = field(
        default=0.6,
        metadata={
            "help": "Minimum fraction of original length expected; smaller suggests truncation."
        },
    )

    @classmethod
    def get_type(cls):
        return EntityDetectorType.MASKER_API

    def __post_init__(self):
        if not self.api_url:
            raise ValueError("Masking API URL not provided in config.")
        if not self.model:
            raise ValueError("Model not provided in config. Expected 'model'.")


@frozen_dataclass
class HostedMaskerApiEntityDetectorConfig(MaskerApiEntityDetectorConfig):
    """Configuration for the hosted Masker API entity detector integration."""

    @classmethod
    def get_type(cls):
        return EntityDetectorType.HOSTED_MASKER_API


@frozen_dataclass
class GlinerEntityDetectorConfig(BaseEntityDetectorConfig):
    """Configuration for the Gliner entity detector integration."""

    labels: List[str] = field(
        default_factory=lambda: ["name", "company", "address"],
        metadata={"help": "Entity types to detect."},
    )
    model: str = field(
        default="urchade/gliner_multi-v2.1",
        metadata={"help": "Name of the HF Gliner model to be used."},
    )
    cuda_device: int = field(
        default=0, metadata={"help": "CUDA device number to use (-1 for CPU)"}
    )
    threshold: float = field(
        default=0.6,
        metadata={
            "help": "Score threshold. Increase for higher precision (fewer spans)."
        },
    )
    batch_size: int = field(default=8, metadata={"help": "Batch size for inference."})
    max_length: int = field(
        default=384,
        metadata={
            "help": "Maximum sequence length in tokens. Documents longer than this will be chunked."
        },
    )
    chunk_overlap: int = field(
        default=50,
        metadata={
            "help": "Number of tokens to overlap between chunks when processing long documents."
        },
    )

    # Post-processing controls
    nms_iou_threshold: float = field(
        default=0.8,
        metadata={
            "help": "IoU threshold for merging overlapping spans across chunks (higher = more aggressive merge)."
        },
    )
    min_span_chars: int = field(
        default=3,
        metadata={"help": "Drop spans shorter than this many characters."},
    )
    max_span_chars: int = field(
        default=80,
        metadata={"help": "Drop spans longer than this many characters."},
    )
    top_k_per_chunk: int = field(
        default=100,
        metadata={
            "help": "Keep only top-K highest-scoring spans per chunk after thresholding (0 disables)."
        },
    )

    @classmethod
    def get_type(cls):
        return EntityDetectorType.GLINER


@frozen_dataclass
class SpacyEntityDetectorConfig(BaseEntityDetectorConfig):
    """Configuration for the Spacy entity detector integration."""

    model: str = field(
        default="es_core_news_sm",
        metadata={"help": "Name of the Spacy model to be used."},
    )
    cuda_device: int = field(
        default=0, metadata={"help": "CUDA device number to use (-1 for CPU)"}
    )

    @classmethod
    def get_type(cls):
        return EntityDetectorType.SPACY
