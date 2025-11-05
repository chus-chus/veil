"""JSON Lines data handler (input + output).

Reads `Document`s from an input JSONL and writes `MaskResult`s to an output JSONL.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from veil.config.datahandler import DataHandlerConfig
from veil.core.base_entity_type import EntityTypeBase
from veil.core.document import Document
from veil.core.mask_result import MaskResult
from veil.core.span import Span
from veil.logger import init_logger

logger = init_logger(__name__)


class DataHandler:
    """Unified handler for offline data I/O.

    - Iterates over input documents (JSONL)
    - Persists per-document `MaskResult` records to output JSONL
    """

    def __init__(self, config: DataHandlerConfig) -> None:
        self.config = config
        self.input_path = self.config.input_path
        self.output_path = self.config.output_path
        self.text_field = self.config.text_field
        self.doc_id_field = self.config.doc_id_field
        self.ground_truth_field = self.config.ground_truth_field
        self.text_encoding = self.config.text_encoding

        self._index = 0

        if self.input_path is None:
            raise ValueError("Input path is currently required by DataHandler")
        else:
            self.in_path = Path(self.input_path)
            if self.in_path.suffix == ".jsonl":
                self._in_file = self.in_path.open("r", encoding=self.text_encoding)
            else:
                raise ValueError("Input must be a .jsonl file")

        if self.output_path is None:
            # default output name: <input_stem>.masking_results-<timestamp>.jsonl
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            stem = self.in_path.stem
            default_name = f"{stem}.masking_results-{timestamp}.jsonl"
            self.out_path = self.in_path.parent / default_name
        else:
            self.out_path = Path(self.output_path)
            # if file already exists, add timestamp to avoid overwriting
            if self.out_path.exists():
                timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
                stem = self.out_path.stem
                suffix = self.out_path.suffix
                timestamped_name = f"{stem}-{timestamp}{suffix}"
                self.out_path = self.out_path.parent / timestamped_name
        # ensure parent directory exists
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.out_path.suffix != ".jsonl":
            raise ValueError("Output must be a .jsonl file")
        # open output in write mode to start a fresh run
        self._out_file = self.out_path.open("w", encoding=self.text_encoding)
        # ensure writes are serialized across threads
        self._write_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Helpers for JSONL docs
    # ------------------------------------------------------------------

    def _extract_text(self, obj: Dict[str, Any]) -> str:
        if self.text_field in obj and obj[self.text_field]:
            return obj[self.text_field]
        raise ValueError(
            f"JSONL object must contain field '{self.text_field}' with a non-empty value."
        )

    def _extract_doc_id(self, obj: Dict[str, Any]) -> str:
        if self.doc_id_field and self.doc_id_field in obj:
            return obj[self.doc_id_field]
        return f"doc-{self._index}"

    def _extract_metadata(self, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "metadata" in obj and obj["metadata"]:
            return obj["metadata"]
        return None

    # ------------------------------------------------------------------
    # Ground–truth parsing helpers (list-of-spans or Inline Entity Blocks)
    # ------------------------------------------------------------------

    def _parse_ground_truth(
        self,
        raw_gt: Any,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Span]]:
        if raw_gt is None:
            return None

        if isinstance(raw_gt, list):
            logger.info(f"Parsing ground truth as a list of spans.")
            spans: List[Span] = []
            for span_obj in raw_gt:
                if isinstance(span_obj, Span):
                    # Rebuild to ensure canonicalized entity type
                    spans.append(
                        Span(
                            start=int(span_obj.start),
                            end=int(span_obj.end),
                            entity_type=self._canonicalize_type_value(getattr(span_obj, "entity_type", None)),  # type: ignore[arg-type]
                            id=getattr(span_obj, "id", None),
                            replacement=getattr(span_obj, "replacement", None),
                            confidence=getattr(span_obj, "confidence", None),
                        )
                    )
                elif isinstance(span_obj, dict):
                    span_dict = dict(span_obj)
                    span_dict["entity_type"] = self._canonicalize_type_value(
                        span_dict.get("entity_type")
                    )
                    spans.append(Span(**span_dict))
                else:
                    raise ValueError(
                        "Each span must be a Span instance or a serialisable dict."
                    )
            return spans or None

        if isinstance(raw_gt, str):
            logger.info(f"Parsing ground truth as Inline Entity Blocks.")
            # Always reconstruct original text from inline blocks for alignment
            reconstructed = self._reconstruct_text_from_inline_blocks(raw_gt)
            return self._parse_inline_entity_blocks(raw_gt, reconstructed)

        raise ValueError(
            "Ground-truth field must be either a list of spans, a masked string, or null."
        )

    @staticmethod
    def _parse_inline_entity_blocks(marked_text: str, original_text: str) -> List[Span]:
        """Parse Inline Entity Blocks of the form
        ##TYPE;ID;RAW_TEXT@@ inside a string and return spans against original_text.

        We reconstruct the original text by replacing each block with RAW_TEXT
        while tracking indices. The reconstructed text MUST match original_text.
        If not, an error is raised.
        """
        token_re = re.compile(
            r"##(?P<etype>[^;#@]+);(?P<id>[^;#@]*);(?P<raw>.*?)@@", re.DOTALL
        )

        # First pass: build reconstructed text and spans with indices relative to it
        recon_parts: List[str] = []
        recon_len = 0  # number of characters accumulated in recon_parts
        spans_temp: List[tuple[str, Optional[str], int, int]] = (
            []
        )  # (etype, id, start_char, end_char)

        pos = 0
        n = len(marked_text)
        while pos < n:
            if marked_text.startswith("##", pos):
                m = token_re.match(marked_text, pos)
                if m:
                    etype_raw = m.group("etype").strip().upper()
                    span_id = m.group("id").strip() or None
                    raw = m.group("raw")
                    start = recon_len
                    recon_parts.append(raw)
                    recon_len += len(raw)
                    end = recon_len
                    spans_temp.append((etype_raw, span_id, start, end))
                    pos = m.end()
                    continue
            # normal character
            recon_parts.append(marked_text[pos])
            recon_len += 1
            pos += 1

        recon_text = "".join(recon_parts)

        spans: List[Span] = []
        if recon_text != original_text:
            logger.info(f"recon_text: {recon_text}")
            logger.info("--------------------------------")
            logger.info(f"original_text: {original_text}")
            raise ValueError(
                "Inline Entity Blocks expansion does not match original text. "
                f"reconstructed_len={len(recon_text)} original_len={len(original_text)}"
            )

        # Indices are correct; build spans
        for etype_raw, span_id, start, end in spans_temp:
            spans.append(
                Span(
                    start=start,
                    end=end,
                    entity_type=DataHandler._canonicalize_type_str(etype_raw),  # type: ignore[arg-type]
                    id=span_id,
                    replacement=original_text[start:end],
                    confidence=1.0,
                )
            )

        spans.sort(key=lambda s: (s.start, s.end))
        return spans

    @staticmethod
    def _reconstruct_text_from_inline_blocks(marked_text: str) -> str:
        """Replace all Inline Entity Blocks (##TYPE;ID;RAW_TEXT@@) with RAW_TEXT only."""
        token_re = re.compile(
            r"##(?P<etype>[^;#@]+);(?P<id>[^;#@]*);(?P<raw>.*?)@@", re.DOTALL
        )
        parts: List[str] = []
        pos = 0
        n = len(marked_text)
        while pos < n:
            if marked_text.startswith("##", pos):
                m = token_re.match(marked_text, pos)
                if m:
                    parts.append(m.group("raw"))
                    pos = m.end()
                    continue
            parts.append(marked_text[pos])
            pos += 1
        return "".join(parts)

    # ------------------------------------------------------------------
    # Canonicalization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _canonicalize_type_str(raw: Optional[str]) -> Optional[str]:
        if raw is None:
            return None
        name = str(raw).upper()
        alias_map = EntityTypeBase.global_alias_map()
        return alias_map.get(name, name)

    def _canonicalize_type_value(self, value: Any) -> Optional[str]:
        # Accept enum-like with .name, plain strings, or None
        if value is None:
            return None
        name = getattr(value, "name", value)
        return self._canonicalize_type_str(name)

    # Iteration APIs
    def __iter__(self):
        return self

    def __next__(self):
        doc = self.get_next()
        if doc is None:
            raise StopIteration
        return doc

    def get_next(self):
        line = self._in_file.readline()
        if line == "":
            return None

        obj = json.loads(line)
        metadata = self._extract_metadata(obj)
        text: str = self._extract_text(obj)
        doc_id: str = self._extract_doc_id(obj)

        ground_truth: Optional[List[Span]] = None
        if self.ground_truth_field:
            raw_gt = obj.get(self.ground_truth_field)
            # If ground truth is inline entity blocks, reconstruct original text from it
            if isinstance(raw_gt, str):
                try:
                    reconstructed = self._reconstruct_text_from_inline_blocks(raw_gt)
                    text = reconstructed
                except Exception:
                    logger.exception(
                        "Failed to reconstruct original text from Inline Entity Blocks for doc %s",
                        doc_id,
                    )
            ground_truth = self._parse_ground_truth(raw_gt, text, metadata)
            if ground_truth:
                logger.info(
                    f"Parsed or found {len(ground_truth)} ground truth spans in document {doc_id}"
                )

        document = Document(
            text=text,
            doc_id=doc_id,
            ground_truth=ground_truth,
            metadata=metadata,
        )

        self._index += 1
        return document

    # Output API
    def write_result(self, doc: Document, result: MaskResult) -> None:
        """Writes the result of masking doc (result) to the output file"""
        payload = {
            "doc_id": result.doc_id or doc.doc_id,
            "masked_text": result.masked_text,
            "original_text": doc.text,
            "evaluation": result.evaluation,
            "ground_truth_entities": [
                {
                    "start": int(s.start),
                    "end": int(s.end),
                    "entity_type": getattr(s, "entity_type", None),
                    "id": getattr(s, "id", None),
                    "confidence": getattr(s, "confidence", None),
                }
                for s in (doc.ground_truth or [])
            ],
            "predicted_entities": [
                {
                    "start": int(s.start),
                    "end": int(s.end),
                    "entity_type": getattr(
                        getattr(s, "entity_type", None), "name", None
                    ),
                    "id": getattr(s, "id", None),
                    "confidence": getattr(s, "confidence", None),
                }
                for s in (result.entities or [])
            ],
        }
        with self._write_lock:
            self._out_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._out_file.flush()

    # Maintenance
    def reset(self) -> None:
        if self.in_path and self.in_path.suffix == ".jsonl" and self._in_file:
            self._in_file.seek(0)
            self._index = 0

    def close(self) -> None:
        try:
            if getattr(self, "_in_file", None):
                self._in_file.close()
        finally:
            if getattr(self, "_out_file", None):
                self._out_file.close()

    def __len__(self):
        with self.in_path.open("r", encoding=self.text_encoding) as f:
            self._len = sum(1 for _ in f)
        return self._len
