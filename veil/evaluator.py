from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from veil.config.evaluator import EvaluatorConfig
from veil.core.base_entity_type import EntityTypeBase
from veil.core.document import Document
from veil.core.mask_result import MaskResult
from veil.core.span import Span
from veil.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from veil.metric_store import MetricStore


@dataclass
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {"tp": int(self.tp), "fp": int(self.fp), "fn": int(self.fn)}


class Evaluator:
    def __init__(self, config: EvaluatorConfig):
        self.config = config

    # -----------------------------
    # Span matching helpers
    # -----------------------------
    def _span_brief(self, span: Span) -> Dict[str, Any]:
        """Return a concise dict for logging: start, end, canonical type, id."""
        try:
            type_name = self._canonical_type(self._get_type_name(span))
        except Exception:
            type_name = None
        return {
            "start": getattr(span, "start", None),
            "end": getattr(span, "end", None),
            "type": type_name,
            "id": getattr(span, "id", None),
        }

    @staticmethod
    def _span_iou(a: Span, b: Span) -> float:
        inter = max(0, min(a.end, b.end) - max(a.start, b.start))
        if inter <= 0:
            return 0.0
        union = (a.end - a.start) + (b.end - b.start) - inter
        if union <= 0:
            return 0.0
        return float(inter) / float(union)

    def _canonical_type(self, raw: Optional[str]) -> str:
        # Normalize to upper and apply alias maps defined by enums only.
        name = (raw or "UNKNOWN").upper()
        alias_map = EntityTypeBase.global_alias_map()
        return alias_map.get(name, name)

    def _is_type_match(self, a: Optional[str], b: Optional[str]) -> bool:
        if not self.config.strict_type:
            return True
        return self._canonical_type(a) == self._canonical_type(b)

    def _get_type_name(self, span: Span) -> Optional[str]:
        et = getattr(span, "entity_type", None)
        if et is None:
            return None
        # Support either enums with .name or plain strings in ground truth
        name = getattr(et, "name", None)
        if isinstance(name, str):
            return name
        if isinstance(et, str):
            return et
        # Fallback to str(et) last-resort
        try:
            return str(et)
        except Exception:
            return None

    def _match_pred_to_gold_mode(
        self,
        preds: List[Span],
        golds: List[Span],
        *,
        mode: str,
        iou_threshold: float | None = None,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy matching of predicted spans to ground-truth spans for a given mode.

        mode: "exact", "exact_by_id", "iou", or "iou_by_id".
        When mode starts with "iou", iou_threshold must be provided.
        Returns (matches, unmatched_pred_indices, unmatched_gold_indices).
        """
        matches: List[Tuple[int, int]] = []
        used_gold: set[int] = set()

        # When evaluating by_id, build a mapping from predicted ID labels to gold ID labels
        # that maximizes compatible overlaps per entity type, making the metric invariant
        # to arbitrary ID label permutations. Optionally use Hungarian-like optimal assignment.
        pred_id_to_gold_id: Dict[Tuple[str, str], Optional[str]] = {}
        if "by_id" in mode:
            logger.debug(
                "By-id matching start: mode=%s, preds=%d, golds=%d, iou_thr=%s",
                mode,
                len(preds),
                len(golds),
                str(iou_threshold),
            )

            # Group indices by (type, id)
            preds_by_cluster: Dict[Tuple[str, str], List[int]] = {}
            golds_by_cluster: Dict[Tuple[str, str], List[int]] = {}

            def _norm_type(span: Span) -> str:
                return self._canonical_type(self._get_type_name(span))

            for pi, p in enumerate(preds):
                pid_raw = getattr(p, "id", None)
                if pid_raw is None:
                    continue
                key = (_norm_type(p), str(pid_raw))
                preds_by_cluster.setdefault(key, []).append(pi)
            for gi, g in enumerate(golds):
                gid_raw = getattr(g, "id", None)
                if gid_raw is None:
                    continue
                key = (_norm_type(g), str(gid_raw))
                golds_by_cluster.setdefault(key, []).append(gi)

            # Build overlap scores between predicted and gold clusters of the same type
            # Score: number of span pairs meeting the positional condition (exact/iou)
            # We also keep earliest position to break ties deterministically.
            clusters_by_type: Dict[str, List[Tuple[str, List[int]]]] = {}
            for (t, pid), idxs in preds_by_cluster.items():
                clusters_by_type.setdefault(t, []).append((pid, idxs))

            for etype, pred_clusters in clusters_by_type.items():
                # Collect gold clusters for this type
                gold_clusters: Dict[str, List[int]] = {
                    gid: idxs
                    for (t2, gid), idxs in golds_by_cluster.items()
                    if t2 == etype
                }
                # Build candidate list: (score, earliest_gold_pos, pid, gid)
                candidates: List[Tuple[int, int, str, str]] = []
                for pid, p_indices in pred_clusters:
                    for gid, g_indices in gold_clusters.items():
                        score = 0
                        for pi in p_indices:
                            p = preds[pi]
                            for gi in g_indices:
                                g = golds[gi]
                                if mode == "exact_by_id":
                                    ok = p.start == g.start and p.end == g.end
                                else:
                                    # iou_by_id
                                    thr = (
                                        float(iou_threshold)
                                        if iou_threshold is not None
                                        else 0.0
                                    )
                                    ok = self._span_iou(p, g) >= thr
                                if ok:
                                    score += 1
                        if score > 0:
                            # earliest gold position
                            earliest_g = min(
                                (golds[gi].start for gi in g_indices), default=10**9
                            )
                            candidates.append((score, earliest_g, pid, gid))
                if getattr(self.config, "use_hungarian_id_mapping", False):
                    # Hungarian-like optimal assignment via cost matrix built from -score
                    # Build index mapping
                    pid_list = [pid for pid, _ in pred_clusters]
                    gid_list = list(gold_clusters.keys())
                    p_index = {pid: i for i, pid in enumerate(pid_list)}
                    g_index = {gid: j for j, gid in enumerate(gid_list)}
                    # Initialize cost matrix with large costs
                    P, G = len(pid_list), len(gid_list)
                    INF = 10**9
                    cost = [[INF for _ in range(G)] for _ in range(P)]
                    # Fill with -score to maximize score
                    for score, _pos, pid, gid in candidates:
                        i = p_index[pid]
                        j = g_index[gid]
                        cost[i][j] = -score
                    # Simple O(P!*G!) fallback to greedy when trivial sizes
                    # For simplicity and to avoid heavy deps, we implement a minimal Kuhn-Munkres variant for small matrices
                    # If matrix is empty, skip
                    if P > 0 and G > 0:
                        assignment = _hungarian_min_cost(cost)
                        for i, j in assignment:
                            if i is None or j is None:
                                continue
                            if cost[i][j] == INF:
                                continue
                            pred_id_to_gold_id[(etype, pid_list[i])] = gid_list[j]
                    # Unassigned predicted clusters map to None
                    for pid in pid_list:
                        pred_id_to_gold_id.setdefault((etype, pid), None)
                else:
                    # Greedy assignment by score desc, then earliest gold position
                    candidates.sort(key=lambda x: (-x[0], x[1]))
                    used_gids: set[str] = set()
                    assigned_pids: set[str] = set()
                    for score, _pos, pid, gid in candidates:
                        if pid in assigned_pids or gid in used_gids:
                            continue
                        pred_id_to_gold_id[(etype, pid)] = gid
                        assigned_pids.add(pid)
                        used_gids.add(gid)
                    # Unassigned predicted clusters map to None (won't match any gold id)
                    for pid, _ in pred_clusters:
                        pred_id_to_gold_id.setdefault((etype, pid), None)

        for pi, p in enumerate(preds):
            p_type = self._get_type_name(p)
            best_gi = -1
            best_score = -1.0
            for gi, g in enumerate(golds):
                if gi in used_gold:
                    continue
                g_type = self._get_type_name(g)
                if not self._is_type_match(p_type, g_type):
                    continue
                if mode == "exact":
                    score = 1.0 if (p.start == g.start and p.end == g.end) else 0.0
                elif mode == "exact_by_id":
                    # Must have identical boundaries and IDs equal up to optimal cluster relabeling
                    pid_raw = getattr(p, "id", None)
                    gid = getattr(g, "id", None)
                    # Map predicted id to gold id namespace based on overlap mapping
                    mapped_pid = None
                    if pid_raw is not None:
                        mapped_pid = pred_id_to_gold_id.get(
                            (self._canonical_type(p_type), str(pid_raw))
                        )
                    score = (
                        1.0
                        if (
                            p.start == g.start
                            and p.end == g.end
                            and mapped_pid is not None
                            and gid is not None
                            and mapped_pid == str(gid)
                        )
                        else 0.0
                    )
                elif mode == "iou_by_id":
                    pid_raw = getattr(p, "id", None)
                    gid = getattr(g, "id", None)
                    mapped_pid = None
                    if pid_raw is not None:
                        mapped_pid = pred_id_to_gold_id.get(
                            (self._canonical_type(p_type), str(pid_raw))
                        )
                    if (
                        mapped_pid is not None
                        and gid is not None
                        and mapped_pid == str(gid)
                    ):
                        score = self._span_iou(p, g)
                    else:
                        score = 0.0
                else:
                    score = self._span_iou(p, g)
                if score > best_score:
                    best_score = score
                    best_gi = gi
            if best_gi != -1:
                if mode in ("exact", "exact_by_id") and best_score >= 1.0:
                    matches.append((pi, best_gi))
                    used_gold.add(best_gi)
                elif (
                    mode in ("iou", "iou_by_id")
                    and iou_threshold is not None
                    and best_score >= float(iou_threshold)
                ):
                    matches.append((pi, best_gi))
                    used_gold.add(best_gi)

            if "by_id" in mode:
                logger.debug(
                    "By-id selection: pred_idx=%d %s -> best_gold_idx=%s score=%.4f",
                    pi,
                    self._span_brief(p),
                    str(best_gi),
                    float(best_score),
                )

        unmatched_preds = [
            i for i in range(len(preds)) if i not in {m[0] for m in matches}
        ]
        unmatched_golds = [i for i in range(len(golds)) if i not in used_gold]
        return matches, unmatched_preds, unmatched_golds

    # -----------------------------
    # Public API
    # -----------------------------
    def evaluate_document(
        self,
        *,
        document: Document,
        mask_result: MaskResult,
        component_spans: Dict[str, List[Span]] | None,
        metric_store: Optional["MetricStore"],
        component_supported_types: Optional[Dict[str, List[str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Compute confusion matrices for a single document and send to metric store.

        Returns only variant-based evaluations under "variants":
          - "exact" (if enabled)
          - "iou@THRESH" for each threshold in config.report_iou_thresholds
        """
        if not self.config.enabled:
            return None
        if document.ground_truth is None:
            return None

        gold_spans: List[Span] = document.ground_truth or []
        pred_spans_all: List[Span] = mask_result.entities or []

        # High-level visibility into inputs when by-id evaluation is enabled
        if getattr(self.config, "report_exact_by_id", False):
            try:
                gold_log = [self._span_brief(s) for s in (gold_spans or [])]
                pred_log = [self._span_brief(s) for s in (pred_spans_all or [])]
                # logger.info(
                #     "Eval by-id inputs (doc_id=%s) gold_spans=%s",
                #     getattr(document, "doc_id", None),
                #     gold_log,
                # )
                # logger.info(
                #     "Eval by-id inputs (doc_id=%s) pred_spans_all=%s",
                #     getattr(document, "doc_id", None),
                #     pred_log,
                # )
                if component_spans:
                    for comp_key, comp_preds in (component_spans or {}).items():
                        # Also log filtered views based on supported types if provided
                        filtered_preds, filtered_golds = (
                            (comp_preds or []),
                            (gold_spans or []),
                        )
                        if component_supported_types is not None:
                            try:
                                filtered_preds, filtered_golds = _filter_supported(
                                    comp_key, comp_preds or []
                                )
                            except Exception:
                                pass
                        # logger.info(
                        #     "Eval by-id component '%s' (doc_id=%s) raw_preds=%s filtered_preds=%s filtered_golds=%s",
                        #     comp_key,
                        #     getattr(document, "doc_id", None),
                        #     [self._span_brief(s) for s in (comp_preds or [])],
                        #     [self._span_brief(s) for s in (filtered_preds or [])],
                        #     [self._span_brief(s) for s in (filtered_golds or [])],
                        # )
            except Exception as e:
                logger.warning("Failed logging by-id inputs: %s", e)

        def _filter_supported(
            comp_key: str, preds: List[Span]
        ) -> Tuple[List[Span], List[Span]]:
            supported_raw = (
                component_supported_types.get(comp_key)
                if component_supported_types is not None
                else None
            )
            if supported_raw is not None:
                supported_set = {self._canonical_type(s) for s in supported_raw}

                def _is_supported(span: Span) -> bool:
                    return (
                        self._canonical_type(self._get_type_name(span)) in supported_set
                    )

                filtered_preds = [p for p in (preds or []) if _is_supported(p)]
                filtered_golds = [g for g in (gold_spans or []) if _is_supported(g)]
            else:
                filtered_preds = preds
                filtered_golds = gold_spans
            return filtered_preds, filtered_golds

        def _compute_counts_for_mode(
            comp_spans: Dict[str, List[Span]] | None,
            *,
            mode: str,
            iou_thr: float | None,
        ) -> Tuple[
            Dict[str, Dict[str, ConfusionCounts]],
            Dict[str, ConfusionCounts],
            Dict[str, ConfusionCounts],
            ConfusionCounts,
        ]:
            per_comp_by_type: Dict[str, Dict[str, ConfusionCounts]] = {}
            per_comp_all_types: Dict[str, ConfusionCounts] = {}
            # Per-component
            if comp_spans:
                for comp_key, comp_preds in comp_spans.items():
                    filtered_preds, filtered_golds = _filter_supported(
                        comp_key, comp_preds
                    )
                    by_type, all_types = self._evaluate_pairwise(
                        filtered_preds,
                        filtered_golds,
                        mode=mode,
                        iou_threshold=iou_thr,
                    )
                    per_comp_by_type[comp_key] = by_type
                    per_comp_all_types[comp_key] = all_types
            # Global
            global_by_type, global_all_types = self._evaluate_pairwise(
                pred_spans_all,
                gold_spans,
                mode=mode,
                iou_threshold=iou_thr,
            )
            return (
                per_comp_by_type,
                per_comp_all_types,
                global_by_type,
                global_all_types,
            )

        # Build variants only
        variants: Dict[str, Dict[str, Any]] = {}
        if self.config.report_exact:
            pcbt, pcat, gbt, gat = _compute_counts_for_mode(
                component_spans, mode="exact", iou_thr=None
            )
            variants["exact"] = {
                "per_component_by_type": {
                    comp: {etype: cc.to_dict() for etype, cc in counts.items()}
                    for comp, counts in pcbt.items()
                },
                "per_component_all_types": {
                    comp: ct.to_dict() for comp, ct in pcat.items()
                },
                "global_by_type": {et: cc.to_dict() for et, cc in gbt.items()},
                "global_all_types": gat.to_dict(),
            }
        if getattr(self.config, "report_exact_by_id", False):
            pcbt, pcat, gbt, gat = _compute_counts_for_mode(
                component_spans, mode="exact_by_id", iou_thr=None
            )
            variants["exact_by_id"] = {
                "per_component_by_type": {
                    comp: {etype: cc.to_dict() for etype, cc in counts.items()}
                    for comp, counts in pcbt.items()
                },
                "per_component_all_types": {
                    comp: ct.to_dict() for comp, ct in pcat.items()
                },
                "global_by_type": {et: cc.to_dict() for et, cc in gbt.items()},
                "global_all_types": gat.to_dict(),
            }
        for thr in self.config.report_iou_thresholds or []:
            try:
                t = float(thr)
            except Exception:
                continue
            pcbt, pcat, gbt, gat = _compute_counts_for_mode(
                component_spans, mode="iou", iou_thr=t
            )
            variants[f"iou@{t:.2f}"] = {
                "per_component_by_type": {
                    comp: {etype: cc.to_dict() for etype, cc in counts.items()}
                    for comp, counts in pcbt.items()
                },
                "per_component_all_types": {
                    comp: ct.to_dict() for comp, ct in pcat.items()
                },
                "global_by_type": {et: cc.to_dict() for et, cc in gbt.items()},
                "global_all_types": gat.to_dict(),
            }
            # ID-aware IoU variant (entity resolution) when by-id reporting is enabled
            if getattr(self.config, "report_exact_by_id", False):
                pcbt_id, pcat_id, gbt_id, gat_id = _compute_counts_for_mode(
                    component_spans, mode="iou_by_id", iou_thr=t
                )
                variants[f"iou@{t:.2f}_by_id"] = {
                    "per_component_by_type": {
                        comp: {etype: cc.to_dict() for etype, cc in counts.items()}
                        for comp, counts in pcbt_id.items()
                    },
                    "per_component_all_types": {
                        comp: ct.to_dict() for comp, ct in pcat_id.items()
                    },
                    "global_by_type": {et: cc.to_dict() for et, cc in gbt_id.items()},
                    "global_all_types": gat_id.to_dict(),
                }

        evaluation: Dict[str, Any] = {"variants": variants}

        # Send variants to metric store
        if metric_store is not None and variants:
            record_variant = getattr(metric_store, "record_evaluation_variant", None)
            if callable(record_variant):
                for v_key, payload in variants.items():
                    record_variant(
                        variant=v_key,
                        per_component_by_type=payload["per_component_by_type"],
                        per_component_all_types=payload["per_component_all_types"],
                        global_by_type=payload["global_by_type"],
                        global_all_types=payload["global_all_types"],
                    )

        return evaluation

    def _evaluate_pairwise(
        self,
        preds: List[Span],
        golds: List[Span],
        *,
        mode: str,
        iou_threshold: float | None,
    ) -> Tuple[Dict[str, ConfusionCounts], ConfusionCounts]:
        # Group spans by type name
        def type_name(s: Span) -> str:
            raw = self._get_type_name(s)
            return self._canonical_type(raw)

        preds_by_type: Dict[str, List[Span]] = {}
        golds_by_type: Dict[str, List[Span]] = {}
        for p in preds:
            preds_by_type.setdefault(type_name(p), []).append(p)
        for g in golds:
            golds_by_type.setdefault(type_name(g), []).append(g)

        all_type_names = set(preds_by_type.keys()) | set(golds_by_type.keys())

        by_type_counts: Dict[str, ConfusionCounts] = {}
        total = ConfusionCounts()

        for etype in sorted(all_type_names):
            p_list = preds_by_type.get(etype, [])
            g_list = golds_by_type.get(etype, [])
            matches, unmatched_p, unmatched_g = self._match_pred_to_gold_mode(
                p_list, g_list, mode=mode, iou_threshold=iou_threshold
            )

            tp = len(matches)
            fp = len(unmatched_p)
            fn = len(unmatched_g)

            by_type_counts[etype] = ConfusionCounts(tp=tp, fp=fp, fn=fn)
            total.tp += tp
            total.fp += fp
            total.fn += fn

        return by_type_counts, total


def _hungarian_min_cost(
    cost: List[List[int]],
) -> List[Tuple[Optional[int], Optional[int]]]:
    """Minimal Hungarian algorithm for rectangular matrices with non-negative costs.

    Returns list of (row_index, col_index) for assigned pairs. Rows or columns
    may remain unassigned if sizes differ. For simplicity, this is a compact
    implementation suitable for small P,G (typical for per-document clusters).
    """
    # Implementation adapted to small sizes: reduce rows/cols, then augment.
    if not cost or not cost[0]:
        return []
    import math

    n_rows = len(cost)
    n_cols = len(cost[0])
    # Pad to square by adding dummy rows/cols with 0 cost
    n = max(n_rows, n_cols)
    a = [row + [0] * (n - n_cols) for row in cost] + [
        [0] * n for _ in range(n - n_rows)
    ]

    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [(-1, -1)] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = (p[j] - 1, j - 1)
    # Filter to original matrix bounds
    result: List[Tuple[Optional[int], Optional[int]]] = []
    for i, j in assignment:
        if i == -1 or j == -1:
            result.append((None, None))
        elif i < n_rows and j < n_cols:
            result.append((i, j))
        else:
            result.append((None, None))
    return result
