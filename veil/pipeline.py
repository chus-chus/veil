import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Set

from veil.config.pipeline import PipelineConfig
from veil.core.base_entity_type import EntityTypeBase
from veil.core.document import Document
from veil.core.mask_result import MaskResult
from veil.core.span import Span
from veil.entity_resolvers.registry import EntityResolverRegistry
from veil.evaluator import Evaluator
from veil.logger import init_logger
from veil.masker import Masker
from veil.metric_store import MetricStore
from veil.overlap_resolver import OverlapResolver

logger = init_logger(__name__)


class Pipeline:
    """Build and execute a masking pipeline based on a config object."""

    def __init__(
        self,
        config: PipelineConfig,
    ) -> None:
        from veil.entity_detectors.registry import EntityDetectorRegistry

        self.config = config

        self.entity_detectors = [
            EntityDetectorRegistry.get(m_cfg.get_type(), m_cfg)
            for m_cfg in config.entity_detectors
        ]

        self.metric_store: MetricStore | None = None
        if config.metric_store.enabled:
            self.metric_store = MetricStore(config.metric_store)
            if config.metric_store.save_config_json:
                self.metric_store.save_config(
                    getattr(self.config, "__flat_config__", self.config)
                )

            # Preserve original YAML/JSON config files used to build this run
            try:
                flat_cfg = getattr(self.config, "__flat_config__", None)
                if flat_cfg is not None:
                    cfg_dir = self.metric_store.path_in_run_dir("configs")
                    cfg_dir.mkdir(parents=True, exist_ok=True)
                    seen_names: set[str] = set()
                    for attr_name, attr_value in vars(flat_cfg).items():
                        if not (
                            isinstance(attr_name, str)
                            and attr_name.endswith("_from_file")
                        ):
                            continue
                        if not (isinstance(attr_value, str) and attr_value):
                            continue
                        try:
                            src_path = Path(attr_value).expanduser().resolve()
                        except Exception:
                            continue
                        if not (src_path.exists() and src_path.is_file()):
                            continue
                        # Avoid collisions; if same basename repeats, suffix with -N
                        dest_name = src_path.name
                        counter = 1
                        while dest_name in seen_names or (cfg_dir / dest_name).exists():
                            dest_name = f"{src_path.stem}-{counter}{src_path.suffix}"
                            counter += 1
                        try:
                            shutil.copy2(src_path, cfg_dir / dest_name)
                            seen_names.add(dest_name)
                        except Exception:
                            logger.exception(
                                "Failed to copy config file %s", str(src_path)
                            )
            except Exception:
                logger.exception(
                    "Failed to preserve source config files into run directory"
                )

        if config.evaluator:
            self.evaluator = Evaluator(config.evaluator)
        else:
            self.evaluator = None
            logger.info("No evaluator configured, no evaluation will be performed.")

        # Masker
        self.masker = Masker(config.masker)

        # Entity resolvers
        self.entity_resolvers = []
        resolvers_cfg = getattr(config, "entity_resolvers", None)

        if resolvers_cfg:
            self.entity_resolvers = [
                EntityResolverRegistry.get(r_cfg.get_type(), r_cfg)
                for r_cfg in resolvers_cfg
            ]

    def process(
        self,
        doc: Document,
        entity_cache: Dict[str, Dict[int, Set[str]]] | None = None,
    ) -> MaskResult:
        """
        Ejecuta el pipeline completo sobre un texto.
        """
        logger.debug(
            "Processing document %s with %d characters",
            doc.doc_id,
            len(doc.text),
        )

        if self.metric_store:
            self.metric_store.start_document(len(doc.text))

        # entity detection
        all_entities: Dict[str, List[Span]] = {}
        for index, entity_detector in enumerate(self.entity_detectors, start=1):
            t0 = time.perf_counter()
            detected = entity_detector.detect_entities(doc)
            logger.debug(
                f"Detected {len(detected)} entity spans in document {doc.doc_id} with entity detector {entity_detector.config.get_type().name}"
            )
            duration = time.perf_counter() - t0

            detector_type = entity_detector.config.get_type().name
            key = f"{index:02d}-{detector_type}"
            all_entities[key] = detected

            if self.metric_store:
                self.metric_store.record_component_step(
                    component=key,
                    component_type="entity_detector",
                    duration_seconds=duration,
                    detected_spans=detected,
                    supported_entity_types=getattr(
                        entity_detector, "get_supported_entities", lambda: []
                    )(),
                )

        # entity resolution
        if self.entity_resolvers:
            for index, resolver in enumerate(self.entity_resolvers, start=1):
                t0 = time.perf_counter()
                # Flatten current entities across detectors to pass into resolver
                combined: List[Span] = []
                for spans in all_entities.values():
                    combined.extend(spans)
                try:
                    resolved = resolver.resolve(doc, combined, entity_cache=entity_cache)  # type: ignore[call-arg]
                except TypeError:
                    resolved = resolver.resolve(doc, combined)
                duration = time.perf_counter() - t0

                # Write resolved ids back into original detector lists by matching positions
                pos_to_id = {
                    (s.start, s.end, getattr(s.entity_type, "name", "")): s.id
                    for s in resolved
                }
                for key, spans in all_entities.items():
                    updated: List[Span] = []
                    for s in spans:
                        assigned = pos_to_id.get(
                            (s.start, s.end, getattr(s.entity_type, "name", ""))
                        )
                        if assigned is None:
                            updated.append(s)
                        else:
                            updated.append(
                                Span(
                                    start=s.start,
                                    end=s.end,
                                    entity_type=s.entity_type,
                                    id=assigned,
                                    replacement=s.replacement,
                                    confidence=s.confidence,
                                )
                            )
                    all_entities[key] = updated

                if self.metric_store:
                    self.metric_store.record_component_step(
                        component=f"{index:02d}-resolver",
                        component_type="entity_resolver",
                        duration_seconds=duration,
                        detected_spans=resolved,
                    )

        # validation

        # Conflict resolution across detectors using OverlapResolver and priorities
        # Build mapping from component key to its priority map (canonical type -> int)
        comp_keys: List[str] = []
        for index, entity_detector in enumerate(self.entity_detectors, start=1):
            detector_type = entity_detector.config.get_type().name
            comp_keys.append(f"{index:02d}-{detector_type}")

        key_to_priority: Dict[str, Dict[str, int]] = {}
        for key, detector in zip(comp_keys, self.entity_detectors):
            pr: Dict[str, int] = getattr(detector.config, "priority", {}) or {}
            # normalize keys to canonical uppercase
            canon_map = EntityTypeBase.global_alias_map()
            norm: Dict[str, int] = {}
            for raw_name, score in pr.items():
                try:
                    name_up = str(raw_name or "").upper()
                    norm_name = canon_map.get(name_up, name_up)
                    norm[norm_name] = int(score)
                except Exception:
                    continue
            key_to_priority[key] = norm

        # Build component hierarchy positions (0 = highest precedence)
        key_to_hierarchy: Dict[str, int] = {}
        for key, detector in zip(comp_keys, self.entity_detectors):
            try:
                key_to_hierarchy[key] = int(
                    getattr(detector.config, "hierarchy_position", 0) or 0
                )
            except Exception:
                key_to_hierarchy[key] = 0

        resolver = OverlapResolver(self.config.overlap_resolver)
        selected: List[Span] = resolver.resolve(
            component_keys_in_order=comp_keys,
            component_to_spans=all_entities,
            component_to_priority=key_to_priority,
            component_to_hierarchy=key_to_hierarchy,
        )

        # masking
        combined_spans: List[Span] = selected

        t0 = time.perf_counter()
        result = self.masker.mask(doc, combined_spans)
        duration = time.perf_counter() - t0

        try:
            result_entities_debug = [
                {
                    "start": int(s.start),
                    "end": int(s.end),
                    "entity_type": getattr(
                        getattr(s, "entity_type", None), "name", None
                    ),
                    "id": getattr(s, "id", None),
                    "confidence": getattr(s, "confidence", None),
                    "replacement": getattr(s, "replacement", None),
                }
                for s in result.entities
            ]
            logger.debug(
                "Masker result entities (doc_id=%s): %s",
                doc.doc_id,
                result_entities_debug,
            )
        except Exception:
            pass

        if self.metric_store:
            self.metric_store.record_component_step(
                component="masker",
                component_type="masker",
                duration_seconds=duration,
                masked_spans=result.entities,
            )

        # evaluation
        if self.evaluator:
            try:
                # Build supported type map for components as passed to metrics
                # Keys match those used in all_entities
                supported_map: Dict[str, List[str]] = {}
                for index, entity_detector in enumerate(self.entity_detectors, start=1):
                    detector_type = entity_detector.config.get_type().name
                    key = f"{index:02d}-{detector_type}"
                    supported_map[key] = getattr(
                        entity_detector, "get_supported_entities", lambda: []
                    )()
                # logger.info("Evaluating spans")
                # logger.info(f"Golden spans: ----------- {doc.ground_truth}\n\n")
                # logger.info(f"Predicted spans: ----------- {all_entities}")
                evaluation = self.evaluator.evaluate_document(
                    document=doc,
                    mask_result=result,
                    component_spans=all_entities,
                    metric_store=self.metric_store,
                    component_supported_types=supported_map,
                )
                if evaluation is not None:
                    result = replace(result, evaluation=evaluation)
            except Exception:
                logger.exception("Evaluator failed for document %s", doc.doc_id)

        if self.metric_store:
            self.metric_store.end_document()

        # return result
        return result
