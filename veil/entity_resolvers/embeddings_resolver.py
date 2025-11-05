from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from veil.config.entity_resolvers import EmbeddingsEntityResolverConfig
from veil.core.document import Document
from veil.core.span import Span
from veil.logger import init_logger


class EmbeddingsEntityResolver:
    """Within-document entity resolver using simple char-ngrams + cosine.

    v1 intentionally avoids external dependencies. It computes normalized
    3-gram character vectors for mention strings and links pairs with cosine
    similarity >= threshold. Clusters are connected components. Only spans of
    the same entity_type are compared. IDs are assigned deterministically by
    first-appearance in the document.
    """

    def __init__(self, config: EmbeddingsEntityResolverConfig):
        self.config = config
        self._logger = init_logger(__name__)

    # ----------------------------- public API -----------------------------
    def resolve(
        self,
        doc: Document,
        spans: List[Span],
        entity_cache: Optional[Dict[str, Dict[int, Set[str]]]] = None,
    ) -> List[Span]:
        if not spans:
            return spans

        # Group spans by entity_type string key to avoid mixing types
        by_type: Dict[str, List[Tuple[int, Span]]] = defaultdict(list)
        for idx, sp in enumerate(spans):
            key = getattr(sp.entity_type, "name", str(sp.entity_type))
            by_type[key].append((idx, sp))

        # Work per type
        id_assignments: Dict[int, int] = {}

        for _etype, typed in by_type.items():
            if not typed:
                continue
            indices, typed_spans = zip(*typed)

            # Build mention strings
            mentions: List[str] = [self._get_mention_text(doc, s) for s in typed_spans]
            vectors = [self._char_ngram_vector(m) for m in mentions]
            norms = [self._l2_norm(v) for v in vectors]

            # Prepare cache candidates for this entity type (only within same type)
            cache_ids_for_type: Dict[int, List[Dict[str, float]]] = {}
            cache_alias_to_id: Dict[str, int] = {}
            max_cached_id_for_type: int = 0
            if entity_cache and _etype in entity_cache:
                for raw_id, alias_set in entity_cache[_etype].items():
                    try:
                        cache_id_int = int(raw_id)
                    except Exception:
                        continue
                    if cache_id_int > max_cached_id_for_type:
                        max_cached_id_for_type = cache_id_int
                    alias_list = sorted(list(alias_set or []))
                    vecs: List[Dict[str, float]] = [
                        self._char_ngram_vector(a) for a in alias_list
                    ]
                    cache_ids_for_type[cache_id_int] = vecs
                    # Preserve exact string aliases for hard equality match
                    for alias in alias_list:
                        try:
                            cache_alias_to_id[str(alias)] = cache_id_int
                        except Exception:
                            continue

            # Reset ID counter per entity type. New local ids start after max cached id for this type
            next_cluster_id = (
                (max_cached_id_for_type + 1) if max_cached_id_for_type > 0 else 1
            )
            try:
                self._logger.debug(
                    "Resolver prep: type=%s spans=%d starting_local_id=%d cache_ids=%d max_cached_id=%d",
                    _etype,
                    len(typed_spans),
                    next_cluster_id,
                    len(cache_ids_for_type),
                    max_cached_id_for_type,
                )
            except Exception:
                pass

            # Build similarity graph via thresholded cosine
            n = len(typed_spans)
            adj: List[List[int]] = [[] for _ in range(n)]
            thr = max(0.0, min(1.0, float(self.config.threshold)))
            for i in range(n):
                for j in range(i + 1, n):
                    sim = self._cosine(vectors[i], vectors[j], norms[i], norms[j])
                    if sim >= thr:
                        adj[i].append(j)
                        adj[j].append(i)

            # Connected components
            comp_id: List[int] = [-1] * n
            comp_counter = 0
            for i in range(n):
                if comp_id[i] != -1:
                    continue
                # BFS
                comp_counter += 1
                q: deque[int] = deque([i])
                comp_id[i] = comp_counter
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if comp_id[v] == -1:
                            comp_id[v] = comp_counter
                            q.append(v)

            # Order components by first appearance to get stable IDs
            comp_to_first_pos: Dict[int, Tuple[int, int]] = {}
            for local_idx, s in enumerate(typed_spans):
                comp = comp_id[local_idx]
                pos_tuple = comp_to_first_pos.get(comp)
                cur_pos = (int(getattr(s, "start", 0)), local_idx)
                if pos_tuple is None or cur_pos < pos_tuple:
                    comp_to_first_pos[comp] = cur_pos

            ordered_components = sorted(comp_to_first_pos.items(), key=lambda x: x[1])
            comp_to_global_id: Dict[int, int] = {}

            # Assign IDs sequentially per component in order of first appearance
            for comp, _pos in ordered_components:
                # Collect members of this component (for logging only)
                member_indices: List[int] = [i for i in range(n) if comp_id[i] == comp]

                # Try to match to cache within this type
                assigned_global_id: Optional[int] = None
                best_cache_sim: float = -1.0
                if cache_alias_to_id:
                    # First, hard equality: if any mention equals a cached alias, reuse that id
                    for mi in member_indices:
                        mention_text = mentions[mi]
                        cid_eq = cache_alias_to_id.get(mention_text)
                        if cid_eq is not None:
                            assigned_global_id = cid_eq
                            best_cache_sim = 1.0
                            try:
                                self._logger.debug(
                                    "Resolver assignment (exact): type=%s comp_size=%d using_cache_id=%d",
                                    _etype,
                                    len(member_indices),
                                    assigned_global_id,
                                )
                            except Exception:
                                pass
                            break

                if assigned_global_id is None and cache_ids_for_type:
                    for cid, alias_vecs in cache_ids_for_type.items():
                        # Compute maximum similarity between any mention in component and any alias for this cache id
                        max_sim_for_cid = 0.0
                        for mi in member_indices:
                            for alias_vec in alias_vecs:
                                sim = self._cosine(
                                    vectors[mi],
                                    alias_vec,
                                    norms[mi],
                                    self._l2_norm(alias_vec),
                                )
                                if sim > max_sim_for_cid:
                                    max_sim_for_cid = sim
                        if max_sim_for_cid > best_cache_sim:
                            best_cache_sim = max_sim_for_cid
                            assigned_global_id = cid

                thr = max(0.0, min(1.0, float(self.config.threshold)))
                if assigned_global_id is not None and best_cache_sim >= thr:
                    # Use cached id
                    try:
                        self._logger.debug(
                            "Resolver assignment: type=%s comp_size=%d using_cache_id=%d sim=%.3f thr=%.3f",
                            _etype,
                            len(member_indices),
                            assigned_global_id,
                            best_cache_sim,
                            thr,
                        )
                    except Exception:
                        pass
                else:
                    # Allocate next local id within type, starting at 1
                    assigned_global_id = next_cluster_id
                    next_cluster_id += 1
                    try:
                        self._logger.debug(
                            "Resolver assignment: type=%s comp_size=%d assigned_local_id=%d",
                            _etype,
                            len(member_indices),
                            assigned_global_id,
                        )
                    except Exception:
                        pass

                comp_to_global_id[comp] = int(assigned_global_id)

            # Write back ids for original span indices
            for local_idx, original_idx in enumerate(indices):
                id_assignments[original_idx] = comp_to_global_id[comp_id[local_idx]]

        # Produce new Spans with id filled (keep dataclass immutability if frozen)
        resolved: List[Span] = []
        for i, sp in enumerate(spans):
            assigned = id_assignments.get(i)
            if assigned is None:
                resolved.append(sp)
            else:
                # ids must be ints per spec
                resolved.append(
                    Span(
                        start=sp.start,
                        end=sp.end,
                        entity_type=sp.entity_type,
                        id=int(assigned),
                        replacement=sp.replacement,
                        confidence=sp.confidence,
                    )
                )

        return resolved

    # --------------------------- helper methods ---------------------------
    def _get_mention_text(self, doc: Document, span: Span) -> str:
        if span.replacement is not None:
            return str(span.replacement)
        start = max(0, min(len(doc.text), int(span.start)))
        end = max(0, min(len(doc.text), int(span.end)))
        return doc.text[start:end]

    def _char_ngram_vector(self, text: str, n: int = 3) -> Dict[str, float]:
        # Basic normalization: lowercase and collapse spaces
        t = " ".join((text or "").lower().split())
        if not t:
            return {}
        feats: Dict[str, float] = defaultdict(float)
        # add boundary markers to encourage prefix/suffix similarity
        padded = f"^{t}$"
        for i in range(len(padded) - n + 1):
            ng = padded[i : i + n]
            feats[ng] += 1.0
        # L1 normalize for stability before cosine
        total = sum(feats.values())
        if total > 0:
            for k in list(feats.keys()):
                feats[k] /= total
        return dict(feats)

    def _l2_norm(self, vec: Dict[str, float]) -> float:
        return math.sqrt(sum(v * v for v in vec.values())) if vec else 0.0

    def _cosine(
        self,
        a: Dict[str, float],
        b: Dict[str, float],
        norm_a: float,
        norm_b: float,
    ) -> float:
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        # Iterate over smaller dict for speed
        if len(a) > len(b):
            a, b = b, a
        dot = 0.0
        for k, v in a.items():
            bv = b.get(k)
            if bv is not None:
                dot += v * bv
        return dot / (norm_a * norm_b)
