from __future__ import annotations

import json
import logging
import re
import time
from typing import Dict, List, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from veil.config.entity_detectors import MaskerApiEntityDetectorConfig
from veil.core.base_entity_detector import BaseEntityDetector
from veil.core.base_entity_type import EntityTypeBase
from veil.core.document import Document
from veil.core.span import Span
from veil.datahandler import DataHandler

logger = logging.getLogger(__name__)


class MaskerApiEntityDetector(BaseEntityDetector[EntityTypeBase]):

    ENTITY_TYPES: Set[EntityTypeBase]

    def __init__(self, cfg: MaskerApiEntityDetectorConfig):
        """Initialize the masking-API entity detector with a configuration.

        Expected config attributes (subclasses should provide):
        - api_url | url | endpoint: str -> endpoint to POST masking requests to
        - headers: dict[str, str] (optional)
        - timeout: float | int (optional, default 30)
        """
        super().__init__(cfg)
        # Allow flexible config attribute names to avoid tight coupling here
        self.api_url = cfg.api_url
        if not self.api_url:
            raise ValueError(
                "Masking API URL not provided in config. Expected 'api_url', 'url' or 'endpoint'."
            )
        self.timeout: float = float(getattr(cfg, "timeout", 30))
        headers: Dict[str, str] | None = getattr(cfg, "headers", None)
        self.headers: Dict[str, str] = dict(headers) if headers else {}
        # Optional chat API parameters (e.g., Fireworks)
        self.model: str | None = getattr(cfg, "model", None)
        self.system_prompt: str | None = getattr(cfg, "system_prompt", None)
        self.max_tokens: int = int(getattr(cfg, "max_tokens", 4000))
        self.top_p: float = float(getattr(cfg, "top_p", 1))
        self.top_k: int = int(getattr(cfg, "top_k", 40))
        self.presence_penalty: float = float(getattr(cfg, "presence_penalty", 0))
        self.frequency_penalty: float = float(getattr(cfg, "frequency_penalty", 0))
        self.temperature: float = float(getattr(cfg, "temperature", 0.6))
        # Resilience controls
        self.retries: int = int(getattr(cfg, "retries", 2))
        self.retry_backoff_base: float = float(getattr(cfg, "retry_backoff_base", 0.5))
        self.retry_on_truncation: bool = bool(getattr(cfg, "retry_on_truncation", True))
        self.chunk_on_truncation: bool = bool(getattr(cfg, "chunk_on_truncation", True))
        self.chunk_char_limit: int = int(getattr(cfg, "chunk_char_limit", 4000))
        self.truncation_min_fraction: float = float(
            getattr(cfg, "truncation_min_fraction", 0.6)
        )

    def send_request(self, doc: Document) -> str:
        """
        Send request to the masking API and return the masked text.
        """
        masked_text, _ = self._send_request_internal(doc)
        return masked_text

    def _send_request_internal(self, doc: Document) -> Tuple[str, bool]:
        """
        Send one request and return (masked_text, truncated_flag).
        Does not implement retries; the caller decides.
        """
        # If a model is configured, assume a chat-completions style API (e.g., Fireworks)
        if self.model:
            supported_types = ", ".join(
                getattr(et, "name", str(et))
                for et in getattr(self, "ENTITY_TYPES", set())
            )
            system_prompt = self.system_prompt or (
                "You are an anonymization engine. Mask the specified entity types by"
                " replacing each sensitive span with a token in the form [ENTITY] where"
                " ENTITY is the entity type name in UPPERCASE. Only output the fully"
                " masked text, preserving original spacing and punctuation as much as possible."
                f" Entity types to mask: {supported_types}."
            )
            payload: Dict[str, object] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": doc.text},
                ],
            }
        else:
            payload = {
                "text": doc.text,
                # Provide entity type names if defined for the subclass; remote API may ignore
                "entity_types": [
                    getattr(et, "name", str(et))
                    for et in getattr(self, "ENTITY_TYPES", set())
                ],
            }

        data = json.dumps(payload).encode("utf-8")
        # avoid Cloudflare bot blocks with a reasonable user-agent
        default_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.headers.get(
                "User-Agent",
                "veil-masker/0.1 (+https://github.com/chus-chus/veil) Python-urllib",
            ),
        }
        headers = {**default_headers, **self.headers}
        req = Request(self.api_url, data=data, headers=headers, method="POST")

        logger.info(
            f"Sent request for document {doc.doc_id} to '{self.api_url}', model '{self.model}'"
        )

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                resp_bytes = resp.read()
                content_type = resp.headers.get("Content-Type", "")
                content_length_header = resp.headers.get("Content-Length")
        except HTTPError as e:
            body = (
                e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            )
            logger.error(f"HTTP error from masking API: {e.code} {e.reason} - {body}")
            raise
        except URLError as e:
            logger.error(f"Failed to reach masking API: {e.reason}")
            raise

        text = resp_bytes.decode("utf-8", errors="replace")

        # Try JSON response first
        masked_text: str | None = None
        truncated_flag: bool = False
        finish_reason: str | None = None
        try:
            if "json" in content_type or (
                text and text.lstrip().startswith(("{", "["))
            ):
                obj = json.loads(text)
                if isinstance(obj, dict):
                    masked_text = (
                        obj.get("masked_text")
                        or obj.get("maskedText")
                        or obj.get("text")
                    )
                    # Fireworks-like chat completions: choices[0].message.content
                    if masked_text is None:
                        choices = obj.get("choices")
                        if isinstance(choices, list) and choices:
                            first = choices[0]
                            if isinstance(first, dict):
                                msg = first.get("message")
                                if isinstance(msg, dict):
                                    content = msg.get("content")
                                    if isinstance(content, str):
                                        masked_text = content
                                # OpenAI/Fireworks style finish_reason
                                fr = first.get("finish_reason") or first.get(
                                    "finishReason"
                                )
                                if isinstance(fr, str):
                                    finish_reason = fr
                    # Some providers include a top-level stop/finish reason
                    if finish_reason is None:
                        fr_top = (
                            obj.get("finish_reason")
                            or obj.get("finishReason")
                            or obj.get("stop_reason")
                        )
                        if isinstance(fr_top, str):
                            finish_reason = fr_top
                elif isinstance(obj, str):
                    masked_text = obj
        except json.JSONDecodeError:
            # Fall back to plain text below
            pass

        if masked_text is None:
            masked_text = text

        if not isinstance(masked_text, str):
            raise ValueError(
                "Masking API response did not contain a masked text string"
            )

        # Detect truncation heuristics
        try:
            if finish_reason:
                # 'length' is common indicator of hitting max tokens
                if finish_reason.lower() in {"length", "max_tokens", "maxTokens"}:
                    truncated_flag = True
            # Network-level incomplete transfer
            if not truncated_flag and content_length_header is not None:
                try:
                    expected_len = int(content_length_header)
                    if expected_len > len(resp_bytes):
                        truncated_flag = True
                except ValueError:
                    pass
            # Heuristic on relative lengths
            if not truncated_flag:
                if len(masked_text) < self.truncation_min_fraction * len(doc.text):
                    # If much shorter than original and doesn’t end on a clean boundary
                    tail = masked_text[-64:]
                    if tail and not tail.strip().endswith(
                        (".", "!", "?", ")", "]", "\n")
                    ):
                        truncated_flag = True
            # Unmatched token bracket at the end
            if (
                not truncated_flag
                and "[" in masked_text
                and not masked_text.rstrip().endswith("]")
            ):
                # Unclosed token near the tail may indicate cut-off
                if masked_text.rfind("[") > masked_text.rfind("]"):
                    truncated_flag = True
        except Exception:
            # Heuristics must not fail the call
            pass

        return masked_text, truncated_flag

    def send_request_with_retries(self, doc: Document) -> Tuple[str, bool]:
        """
        Send request with retries and backoff. Returns (masked_text, truncated_flag).
        """
        last_err: Exception | None = None
        masked_text: str = ""
        truncated: bool = False
        for attempt in range(self.retries + 1):
            try:
                masked_text, truncated = self._send_request_internal(doc)
                if truncated and self.retry_on_truncation and attempt < self.retries:
                    backoff = self.retry_backoff_base * (2**attempt)
                    time.sleep(backoff)
                    continue
                return masked_text, truncated
            except (HTTPError, URLError) as e:
                last_err = e
                if attempt >= self.retries:
                    raise
                backoff = self.retry_backoff_base * (2**attempt)
                time.sleep(backoff)
        if last_err:
            raise last_err
        return masked_text, truncated

    def detect_entities(self, doc: Document) -> List[Span]:
        """
        Detect entities by delegating to a remote masking API and parsing its masked output.
        """
        masked_text, truncated = self.send_request_with_retries(doc)
        # logger.info("-" * 100)
        # logger.info(f"Original text: {doc.text}")
        # logger.info(f"Masked text: {masked_text}")
        # logger.info(f"Truncated: {truncated}")
        # logger.info("*" * 100)
        if truncated and self.chunk_on_truncation:
            spans = self._detect_with_chunking(doc)
            logger.info(
                f"Detected {len(spans)} entity spans in document {doc.doc_id} using chunking fallback"
            )
            return self._map_and_filter_supported_spans(spans)
        spans = self._diff_to_spans(doc.text, masked_text)
        logger.info(f"Detected {len(spans)} entity spans in document {doc.doc_id}")
        # logger.info(f"Detected entities: {spans}")
        return self._map_and_filter_supported_spans(spans)

    def _detect_with_chunking(self, doc: Document) -> List[Span]:
        """
        Fallback: split the document text into manageable chunks and merge spans.
        """
        text = doc.text
        if not text:
            return []
        limit = max(256, int(self.chunk_char_limit))
        overlap = min(128, max(32, limit // 16))
        chunks: List[tuple[int, str]] = []  # (start_index, chunk_text)

        start = 0
        n = len(text)
        while start < n:
            end = min(start + limit, n)
            if end < n:
                # try to break at whitespace near the boundary
                ws = text.rfind(" ", start, end)
                if ws != -1 and ws - start >= limit * 0.7:
                    end = ws
            chunk_text = text[start:end]
            chunks.append((start, chunk_text))
            if end >= n:
                break
            # advance with overlap to avoid cutting entities
            start = max(0, end - overlap)

        merged_spans: List[Span] = []
        for idx, (offset, chunk_text) in enumerate(chunks):
            sub_doc = Document(text=chunk_text, doc_id=f"{doc.doc_id}-chunk-{idx}")
            chunk_masked, _ = self.send_request_with_retries(sub_doc)
            chunk_spans = self._diff_to_spans(chunk_text, chunk_masked)
            for sp in chunk_spans:
                adj = Span(
                    start=sp.start + offset,
                    end=sp.end + offset,
                    entity_type=sp.entity_type,  # type: ignore[arg-type]
                    id=sp.id,
                    replacement=sp.replacement,
                    confidence=sp.confidence,
                )
                merged_spans.append(adj)

        # Sort and coalesce overlapping spans conservatively (prefer earlier span)
        merged_spans.sort(key=lambda s: (s.start, s.end))
        coalesced: List[Span] = []
        for sp in merged_spans:
            if not coalesced or sp.start >= coalesced[-1].end:
                coalesced.append(sp)
            else:
                # overlap: extend the last one if this extends further
                last = coalesced[-1]
                if sp.end > last.end:
                    coalesced[-1] = Span(
                        start=last.start,
                        end=sp.end,
                        entity_type=last.entity_type,  # type: ignore[arg-type]
                        id=last.id,
                        replacement=doc.text[last.start : sp.end],
                    )
        return coalesced

    def _map_and_filter_supported_spans(self, spans: List[Span]) -> List[Span]:
        """
        Convert string entity types in spans to enum members and drop unsupported types.
        """
        if not spans:
            return spans

        # Build mapping from enum name to enum member for supported types
        supported_map: Dict[str, EntityTypeBase] = {
            getattr(e, "name", str(e)).upper(): e
            for e in getattr(self, "ENTITY_TYPES", set())
        }

        filtered: List[Span] = []
        for sp in spans:
            et = getattr(sp, "entity_type", None)
            if isinstance(et, EntityTypeBase):
                # Already an enum member; ensure it's supported
                if getattr(et, "name", str(et)).upper() in supported_map:
                    filtered.append(sp)
                else:
                    logger.warning(
                        f"Dropping span with unsupported entity type: {getattr(et, 'name', et)}"
                    )
                continue

            # Attempt to map from string to enum member
            if isinstance(et, str):
                key = et.upper()
                if key in supported_map:
                    mapped = supported_map[key]
                    filtered.append(
                        Span(
                            start=sp.start,
                            end=sp.end,
                            entity_type=mapped,
                            id=sp.id,
                            replacement=sp.replacement,
                            confidence=sp.confidence,
                        )
                    )
                else:
                    logger.warning(
                        f"Dropping span with unknown/unsupported entity type from API output: {et}"
                    )
            else:
                # Unknown type information; drop
                logger.warning(
                    "Dropping span with missing entity type information from API output"
                )

        return filtered

    @staticmethod
    def _diff_to_spans(original: str, masked: str) -> List[Span]:
        def normalise_with_map(s: str) -> tuple[str, List[int]]:
            norm_chars: List[str] = []
            index_map: List[int] = []
            i = 0
            n = len(s)
            while i < n:
                ch = s[i]
                if ch.isspace():
                    start_ws = i
                    while i < n and s[i].isspace():
                        i += 1
                    norm_chars.append(" ")
                    index_map.append(start_ws)
                    continue
                norm_chars.append(ch)
                index_map.append(i)
                i += 1
            return ("".join(norm_chars), index_map)

        def map_norm_to_orig(
            norm_index: int, index_map: List[int], orig_len: int
        ) -> int:
            if norm_index <= 0:
                return 0
            if norm_index >= len(index_map):
                return orig_len
            return index_map[norm_index]

        spans: List[Span] = []
        token_re = re.compile(r"\[(?P<etype>[A-Z_]+)(?P<num>\d*)\]", re.IGNORECASE)

        norm_orig, orig_map = normalise_with_map(original)
        tokens = list(token_re.finditer(masked))
        if not tokens:
            return []

        masked_segments: List[tuple[str, Optional[re.Match[str]]]] = []
        prev = 0
        for m in tokens:
            if m.start() > prev:
                masked_segments.append((masked[prev : m.start()], None))
            masked_segments.append((masked[m.start() : m.end()], m))
            prev = m.end()
        if prev < len(masked):
            masked_segments.append((masked[prev:], None))

        def normalise_segment(seg: str) -> str:
            return normalise_with_map(seg)[0]

        def find_in_norm(haystack: str, needle: str, start: int) -> int:
            if not needle:
                return start
            return haystack.find(needle, start)

        def find_fallback_end(haystack: str, start: int) -> int:
            # Try to cap the wildcard span at a reasonable boundary if the next literal is not found.
            # Prefer sentence/phrase delimiters; otherwise cap to a max length.
            delimiters = {",", ";", ":", ".", ")"}
            max_len = 512
            end = len(haystack)
            for i in range(start, len(haystack)):
                if haystack[i] in delimiters and (
                    i + 1 >= len(haystack) or haystack[i + 1] == " "
                ):
                    end = i
                    break
                # hard cap
                if i - start >= max_len:
                    end = i
                    break
            return end

        pos_norm_orig = 0
        i = 0
        while i < len(masked_segments):
            seg_text, seg_tok = masked_segments[i]
            if seg_tok is None:
                seg_norm = normalise_segment(seg_text)
                if seg_norm:
                    found = find_in_norm(norm_orig, seg_norm, pos_norm_orig)
                    if found != -1:
                        pos_norm_orig = found + len(seg_norm)
                i += 1
                continue

            start_norm = pos_norm_orig
            j = i + 1
            next_text_norm = ""
            while j < len(masked_segments):
                t_text, t_tok = masked_segments[j]
                if t_tok is None:
                    next_text_norm = normalise_segment(t_text)
                    break
                j += 1

            if next_text_norm:
                end_found = find_in_norm(norm_orig, next_text_norm, start_norm)
                if end_found == -1:
                    end_norm = find_fallback_end(norm_orig, start_norm)
                else:
                    end_norm = end_found
            else:
                end_norm = find_fallback_end(norm_orig, start_norm)

            start_idx = map_norm_to_orig(start_norm, orig_map, len(original))
            end_idx = map_norm_to_orig(end_norm, orig_map, len(original))

            if 0 <= start_idx <= end_idx <= len(original) and start_idx != end_idx:
                entity_type_raw = seg_tok.group("etype").upper()
                entity_type = DataHandler._canonicalize_type_str(entity_type_raw)
                num_suffix = seg_tok.group("num")
                if num_suffix:
                    normalized_id = num_suffix.lstrip("0")
                    span_id = normalized_id if normalized_id != "" else "0"
                else:
                    span_id = None
                sensitive_text = original[start_idx:end_idx]
                spans.append(
                    Span(
                        start=start_idx,
                        end=end_idx,
                        entity_type=entity_type,  # type: ignore[arg-type]
                        id=span_id,
                        replacement=sensitive_text,
                        confidence=1.0,
                    )
                )

            pos_norm_orig = end_norm
            i += 1

        spans.sort(key=lambda s: (s.start, s.end))
        merged: List[Span] = []
        for sp in spans:
            if not merged or sp.start >= merged[-1].end:
                merged.append(sp)
            else:
                last = merged[-1]
                if sp.end > last.end:
                    merged[-1] = Span(
                        start=last.start,
                        end=sp.end,
                        entity_type=last.entity_type,  # type: ignore[arg-type]
                        id=last.id,
                        replacement=original[last.start : sp.end],
                    )
        return merged
