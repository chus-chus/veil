---
title: Implementing new pipeline components
---

# Implementing pipeline components

This chapter describes how to extend Veil by implementing new pipeline components, with a special focus on entity detectors. You will see what input each stage receives, what output it produces, and how it fits into the overall flow.

See also:
- Architecture: [Basic Architecture](architecture.md)
- Declaring pipelines: [Declaring pipelines](declaring-pipelines.md)
- API Reference: [veil.pipeline](api/veil.pipeline.md), [veil.entity_detectors](api/veil.entity_detectors.md), [veil.entity_resolvers](api/veil.entity_resolvers.md), [veil.masker](api/veil.masker.md), [veil.config](api/veil.config.md)

## Shared Types and Data

- `Document` (input to almost all components): see `veil.core.document.Document` API in [veil.pipeline](api/veil.pipeline.md) (used within the pipeline). Key fields: `text`, `doc_id`, `ground_truth`.
- `Span` (entity unit): see `veil.core.span.Span` referenced from [veil.pipeline](api/veil.pipeline.md). Fields: `start`, `end`, `entity_type`, `id`, `replacement`, `confidence`.
- `MaskResult` (output of the masker and the pipeline): see `veil.core.mask_result.MaskResult` in [veil.masker](api/veil.masker.md). Fields: `masked_text`, `entities`, `evaluation`.

These classes are propagated through the pipeline stages and define the standard inputs/outputs.

## Flow and Interfaces by Component

At a high level, the pipeline (see `Pipeline.process` in [veil.pipeline](api/veil.pipeline.md)) executes the following steps, given an input `Document`:
1. Entity detectors → Lists of `Span` per detector.
2. Entity resolvers (optional) → Adjust/merge `Span`s and can assign consistent `id`s.
3. Overlap resolution → Final selection of `Span` according to priorities/hierarchy.
4. Masking → `MaskResult` with `masked_text` and `entities`.
5. Evaluation (optional) → Metrics with `ground_truth` if available.

### 1) Entity Detectors (main extension)

- Base class to inherit from: `veil.core.base_entity_detector.BaseEntityDetector` in [veil.entity_detectors](api/veil.entity_detectors.md).
- Associated config: subclasses of `veil.config.entity_detectors.BaseEntityDetectorConfig` in [veil.config](api/veil.config.md).
- Registry: `veil.entity_detectors.registry.EntityDetectorRegistry` in [veil.entity_detectors](api/veil.entity_detectors.md) maps `EntityDetectorType` → concrete class.

Implementation contract:
- Define `ENTITY_TYPES: Set[EntityTypeBase]` or a specific `EntityTypeBase` (enum) with the types supported by the detector. Examples of types: `RegexEntityType`, `GlinerEntityType` (see [veil.entity_detectors](api/veil.entity_detectors.md)).
- Implement `detect_entities(self, doc: Document) -> List[Span]` returning spans with `start`, `end`, `entity_type`, and, if applicable, `confidence`.
- Use the `config` passed in the constructor (a subclass of `BaseEntityDetectorConfig`) for its own parameters. Common available fields: `priority` (by canonical type) and `hierarchy_position` (global precedence). See [veil.config](api/veil.config.md).

Inputs and outputs:
- Input: `Document` (`doc.text`, `doc.doc_id`).
- Output: `List[Span]` detected by the detector.

Automatic integration into the pipeline:
- The pipeline instantiates detectors from the `entity_detectors` list in `PipelineConfig` using the registry. The order in the list defines the execution order. See `Pipeline.__init__` and `Pipeline.process` in [veil.pipeline](api/veil.pipeline.md).

Quick steps to add a new detector

1. Define the entity types (subclass of `EntityTypeBase`) and, if applicable, their `aliases()`.
2. Implement the detector class inheriting from `BaseEntityDetector`, define `ENTITY_TYPES` and `detect_entities()`.
3. Create the `Config` class inheriting from `BaseEntityDetectorConfig`, add your parameters and `get_type()`.
4. Register the class in `EntityDetectorRegistry` under an `EntityDetectorType`.
5. Add the block to your `PipelineConfig.entity_detectors` YAML with `type:` and your parameters.
6. Adjust `priority` by type and `hierarchy_position` for the desired overlap behavior.

Important note on polymorphic types (`type`):
- In addition to registering the class, import your new configuration class in `veil/config/__init__.py` to expose it to the configuration system, and if you create a new detector module/package, expose it also in `veil/entity_detectors/__init__.py`. This allows selecting it by `type` from YAML/CLI.

Best practices for detectors:
- Fill in `confidence` when a natural score exists; it helps to break ties in overlaps.
- Maintain `ENTITY_TYPES` and `EntityTypeBase.aliases()` if you need to normalize alternative names (the pipeline uses `EntityTypeBase.global_alias_map()` for canonical names).
- Use `priority` by type and `hierarchy_position` to guide selection in conflicts (see overlap section).

Complete example (skeleton)

```python
# 1) Types
from veil.core.base_entity_type import EntityTypeBase
class MyEntityType(EntityTypeBase):
    NAME = 1

# 2) Detector
from typing import List, Set
from veil.core.base_entity_detector import BaseEntityDetector
from veil.core.document import Document
from veil.core.span import Span
class MyDetector(BaseEntityDetector[MyEntityType]):
    ENTITY_TYPES: Set[MyEntityType] = {MyEntityType.NAME}
    def __init__(self, config):
        super().__init__(config)
    def detect_entities(self, doc: Document) -> List[Span]:
        return []

# 3) Config
from dataclasses import field
from veil.config.core.frozen_dataclass import frozen_dataclass
from veil.config.entity_detectors import BaseEntityDetectorConfig
from veil.core.enums.entity_detector_type import EntityDetectorType
@frozen_dataclass
class MyDetectorConfig(BaseEntityDetectorConfig):
    my_threshold: float = field(default=0.5)
    @classmethod
    def get_type(cls):
        return EntityDetectorType.GLINER  # or a new type if added

# 4) Registry
from veil.entity_detectors.registry import EntityDetectorRegistry
EntityDetectorRegistry.register(EntityDetectorType.GLINER, MyDetector)

# 5) (polymorphic) Expose Config and detector in __init__.py
# - Add `from .entity_detectors import MyDetectorConfig` in `veil/config/__init__.py`
# - Add `from .my_detector import MyDetector` in `veil/entity_detectors/__init__.py` (if applicable)
```

And in YAML:

```yaml
entity_detectors:
  - type: gliner  # or your new type
    my_threshold: 0.6
    priority:
      NAME: 0
    hierarchy_position: 0
```

### 2) Entity Resolvers (optional)

- Practical interface: any class with `resolve(self, doc: Document, spans: List[Span], entity_cache: Optional[Dict[str, Dict[int, Set[str]]]] = None) -> List[Span]` works (see `EmbeddingsEntityResolver` in [veil.entity_resolvers](api/veil.entity_resolvers.md)).
- Config: inherit from `BaseEntityResolverConfig` and return `get_type()`.
- Input: the document and the combined list of detected spans so far (accumulated from all detectors); optionally an `entity_cache` by type.
- Output: the same list of spans, potentially with assigned `id`s or deduplications/merges applied.

Simple example: assign sequential IDs by type

```python
from typing import Dict, List, Optional, Set
from veil.core.document import Document
from veil.core.span import Span

class SimpleIdResolver:
    def __init__(self, config):
        self.config = config

    def resolve(self, doc: Document, spans: List[Span], entity_cache: Optional[Dict[str, Dict[int, Set[str]]]] = None) -> List[Span]:
        next_id_per_type: Dict[str, int] = {}
        out: List[Span] = []
        for s in spans:
            et = getattr(s.entity_type, "name", "")
            if getattr(s, "id", None) is None:
                nid = next_id_per_type.get(et, 1)
                next_id_per_type[et] = nid + 1
                out.append(Span(start=s.start, end=s.end, entity_type=s.entity_type, id=nid, replacement=s.replacement, confidence=s.confidence))
            else:
                out.append(s)
        return out
```

### 3) Overlap Resolution (OverlapResolver)

- Class: `veil.overlap_resolver.OverlapResolver` in [veil.pipeline](api/veil.pipeline.md) and configuration in [veil.config](api/veil.config.md).
- Inputs: per detector, its list of spans and a `priority` map by canonical type; also a global `hierarchy_position` per detector. `iou_threshold` parameter in `OverlapResolverConfig`.
- Output: `List[Span]` selected, resolving conflicts between spans of the same type.

It is not usually necessary to implement a new one; adjust `priority` and `hierarchy_position` from the detectors' configuration.

### 4) Masker

- Class: `veil.masker.Masker` in [veil.masker](api/veil.masker.md). It receives a `Document` and spans and returns a `MaskResult`.
- Mask types: controlled by `MaskerConfig.method` (see [veil.config](api/veil.config.md)).

You will generally not need to extend it to add detectors; its interface is stable.

### 5) Evaluation (Evaluator)

- Orchestration class: `veil.evaluator.Evaluator` in [veil.pipeline](api/veil.pipeline.md); configuration in [veil.config](api/veil.config.md).
- Main input: `document`, `mask_result` (final spans), `component_spans` (spans per detector), a map of supported types per component, and `metric_store`.
- Output: a dictionary with metric variants (exact, iou@THRESH, etc.) which is attached to `MaskResult.evaluation`.

For most extensions (new detectors), you do not need to touch the evaluation.
