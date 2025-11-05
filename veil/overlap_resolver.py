from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from veil.config.overlap_resolver import OverlapResolverConfig
from veil.core.base_entity_type import EntityTypeBase
from veil.core.span import Span
from veil.logger import init_logger

logger = init_logger(__name__)


class OverlapResolver:
    """Resolve overlaps between spans coming from multiple detectors using priorities only.

    Selection order:
    1) Lower numeric priority value first (0 is highest), per canonical entity type and detector
    2) Earlier hierarchy position (smaller number wins), when provided
    3) Only spans of the SAME canonical entity type are considered conflicting. Cross-type overlaps are allowed.
    4) If a conflict remains after (1) and (2) for the same type, break ties deterministically by:
       - higher confidence first; then longer span; then earlier start; otherwise keep the existing selection

    """

    def __init__(self, config: OverlapResolverConfig) -> None:
        self.iou_threshold = config.iou_threshold
        self.overlap_policy = config.overlap_policy.lower()

    @staticmethod
    def _span_brief(span: Span) -> dict:
        try:
            type_name = OverlapResolver._canonical_type_name(span)
        except Exception:
            type_name = None
        return {
            "start": int(getattr(span, "start", 0)),
            "end": int(getattr(span, "end", 0)),
            "type": type_name,
            "id": getattr(span, "id", None),
            "confidence": getattr(span, "confidence", None),
        }

    @staticmethod
    def _canonical_type_name(span: Span) -> Optional[str]:
        et = getattr(span, "entity_type", None)
        if et is None:
            return None
        name: Optional[str] = getattr(et, "name", None)
        if not isinstance(name, str):
            if isinstance(et, str):
                name = et
            else:
                try:
                    name = str(et)
                except Exception:
                    name = None
        if name is None:
            return None
        alias_map = EntityTypeBase.global_alias_map()
        return alias_map.get(name.upper(), name.upper())

    @staticmethod
    def _iou(a: Span, b: Span) -> float:
        inter = max(0, min(int(a.end), int(b.end)) - max(int(a.start), int(b.start)))
        if inter <= 0:
            return 0.0
        len_a = int(a.end) - int(a.start)
        len_b = int(b.end) - int(b.start)
        union = len_a + len_b - inter
        if union <= 0:
            return 0.0
        return float(inter) / float(union)

    def resolve(
        self,
        *,
        component_keys_in_order: List[str],
        component_to_spans: Dict[str, List[Span]],
        component_to_priority: Dict[str, Dict[str, int]],
        component_to_hierarchy: Dict[str, int] | None = None,
    ) -> List[Span]:
        # Build sortable candidates
        # (priority, detector_order, start, end, component_key, span)
        Candidate = Tuple[int, int, int, int, str, Span]
        candidates: List[Candidate] = []

        for order_idx, key in enumerate(component_keys_in_order):
            spans = component_to_spans.get(key, [])
            # Deduplicate identical spans (same start, end, canonical type) within the same component
            seen: set[tuple[int, int, Optional[str]]] = set()
            pr_map = component_to_priority.get(key, {})
            for s in spans:
                ctype = self._canonical_type_name(s)
                start_i = int(getattr(s, "start", 0))
                end_i = int(getattr(s, "end", 0))
                dedupe_key = (start_i, end_i, ctype)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                base_priority = pr_map.get(ctype or "", 1_000_000)
                priority_val = int(base_priority)
                candidates.append((priority_val, order_idx, start_i, end_i, key, s))

        try:
            logger.debug(
                "OverlapResolver: built %d candidates from %d components (policy=%s, iou_threshold=%.3f)",
                len(candidates),
                len(component_keys_in_order),
                self.overlap_policy,
                float(self.iou_threshold),
            )
        except Exception:
            pass

        # Sort (without using detector order or span-derived tiebreakers):
        # 1) lower numeric priority first (0 is highest)
        # 2) earlier hierarchy position (lower numeric wins) when provided
        if component_to_hierarchy:

            def _sort_key(c: Candidate):
                comp = c[4]
                hier = int(component_to_hierarchy.get(comp, 1_000_000))
                return (c[0], hier)

            candidates.sort(key=_sort_key)
        else:
            candidates.sort(key=lambda c: (c[0],))

        selected: List[Tuple[Span, int, str]] = []  # (span, priority, component_key)
        for prio, _ord, _start, _end, comp_key, span in candidates:
            try:
                logger.debug(
                    "Candidate: %s prio=%d comp=%s",
                    self._span_brief(span),
                    prio,
                    comp_key,
                )
            except Exception:
                pass
            conflict = False
            for kept_span, kept_prio, kept_comp in selected:
                try:
                    cand_type = self._canonical_type_name(span)
                    kept_type = self._canonical_type_name(kept_span)
                except Exception:
                    cand_type = None
                    kept_type = None
                iou_val = self._iou(span, kept_span)

                same_type_conflict = (
                    cand_type is not None
                    and cand_type == kept_type
                    and iou_val > self.iou_threshold
                )
                cross_type_conflict = (
                    self.overlap_policy == "cross_type" and iou_val > self.iou_threshold
                )

                try:
                    logger.debug(
                        "Check vs kept: kept=%s prio=%d comp=%s | types=(%s vs %s) iou=%.3f -> same_type_conflict=%s cross_type_conflict=%s",
                        self._span_brief(kept_span),
                        kept_prio,
                        kept_comp,
                        cand_type,
                        kept_type,
                        float(iou_val),
                        same_type_conflict,
                        cross_type_conflict,
                    )
                except Exception:
                    pass

                if same_type_conflict or cross_type_conflict:
                    conflict = True
                    if prio == kept_prio:
                        # Exact duplicate from same component -> skip silently
                        if (
                            comp_key == kept_comp
                            and int(getattr(span, "start", 0))
                            == int(getattr(kept_span, "start", 0))
                            and int(getattr(span, "end", 0))
                            == int(getattr(kept_span, "end", 0))
                        ):
                            try:
                                logger.debug(
                                    "Decision: duplicate from same component -> skip candidate"
                                )
                            except Exception:
                                pass
                            break
                        # Prefer lower hierarchy position if provided
                        cand_h = (
                            int(component_to_hierarchy.get(comp_key, 1_000_000))
                            if component_to_hierarchy is not None
                            else 1_000_000
                        )
                        kept_h = (
                            int(component_to_hierarchy.get(kept_comp, 1_000_000))
                            if component_to_hierarchy is not None
                            else 1_000_000
                        )
                        if cand_h < kept_h:
                            try:
                                logger.debug(
                                    "Decision: cand wins by hierarchy (%d < %d)",
                                    cand_h,
                                    kept_h,
                                )
                            except Exception:
                                pass
                            selected.remove((kept_span, kept_prio, kept_comp))
                            selected.append((span, prio, comp_key))
                            break
                        if cand_h == kept_h:
                            # Deterministic tiebreaker for equal priority and hierarchy within same type
                            cand_conf = float(getattr(span, "confidence", 0) or 0)
                            kept_conf = float(getattr(kept_span, "confidence", 0) or 0)
                            if cand_conf > kept_conf:
                                try:
                                    logger.debug(
                                        "Decision: cand wins by confidence (%.4f > %.4f)",
                                        cand_conf,
                                        kept_conf,
                                    )
                                except Exception:
                                    pass
                                selected.remove((kept_span, kept_prio, kept_comp))
                                selected.append((span, prio, comp_key))
                                break
                            if cand_conf == kept_conf:
                                cand_len = int(getattr(span, "end", 0)) - int(
                                    getattr(span, "start", 0)
                                )
                                kept_len = int(getattr(kept_span, "end", 0)) - int(
                                    getattr(kept_span, "start", 0)
                                )
                                if cand_len > kept_len:
                                    try:
                                        logger.debug(
                                            "Decision: cand wins by length (%d > %d)",
                                            cand_len,
                                            kept_len,
                                        )
                                    except Exception:
                                        pass
                                    selected.remove((kept_span, kept_prio, kept_comp))
                                    selected.append((span, prio, comp_key))
                                    break
                                if cand_len == kept_len:
                                    cand_start = int(getattr(span, "start", 0))
                                    kept_start = int(getattr(kept_span, "start", 0))
                                    if cand_start < kept_start:
                                        try:
                                            logger.debug(
                                                "Decision: cand wins by earlier start (%d < %d)",
                                                cand_start,
                                                kept_start,
                                            )
                                        except Exception:
                                            pass
                                        selected.remove(
                                            (kept_span, kept_prio, kept_comp)
                                        )
                                        selected.append((span, prio, comp_key))
                                        break
                            # Otherwise, keep the previously selected span
                        # Otherwise, kept wins; do nothing
                        try:
                            logger.debug(
                                "Decision: kept wins (higher hierarchy or tiebreakers)"
                            )
                        except Exception:
                            pass
                        break
            if not conflict:
                try:
                    logger.debug("No conflict -> append candidate to selected")
                except Exception:
                    pass
                selected.append((span, prio, comp_key))

        result = [s for (s, _p, _k) in selected]
        try:
            logger.debug("OverlapResolver: selected %d spans", len(result))
        except Exception:
            pass
        return result
