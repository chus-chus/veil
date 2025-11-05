from dataclasses import field
from typing import Optional

from veil.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass
class DataHandlerConfig:
    """Configuration for input/output data handling."""

    input_path: str = field(
        default=None,
        metadata={"help": "Path to input file (jsonl)."},
    )
    output_path: str = field(
        default=None,
        metadata={"help": "Path to output file (jsonl)."},
    )
    text_field: str = field(
        default="text",
        metadata={"help": "Key in input docs that holds the text to mask."},
    )
    ground_truth_field: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the key in input docs that holds the masked text. If omitted, it will be generated."
        },
    )
    doc_id_field: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the key in input docs that holds the doc id. If omitted, it will be generated."
        },
    )
    text_encoding: str = field(
        default="utf-8",
        metadata={"help": "File encoding when reading text files"},
    )
