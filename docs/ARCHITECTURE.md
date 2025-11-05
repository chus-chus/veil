# Basic Architecture

## Quick-summary

Veil is a framework for building and running text masking pipelines. It allows defining a pipeline declaratively and running it in offline or online mode.

The core component, `Pipeline`, generalizes a chain of configurable components.

## Pipeline-Architecture

```
Dataloader / API server -> entity detectors -> entity resolvers (optional) -> overlap resolution -> masker -> evaluation (optional)
```

**High-level flow:**
1. **Input** → Data arrives from batch files (Dataloader) or real-time requests (API Server).
2. **Detection** → Sensitive entities are identified with configurable detectors.
3. **Entity Resolution (optional)** → Resolvers can unify identifiers/attributes, enrich or deduplicate entities. They support `entity_cache` when available.
4. **Overlap Resolution** → `OverlapResolver` selects the final spans by combining priorities by type and a component hierarchy. Priorities are declared per detector using `priority` (by entity type, normalized with `EntityTypeBase.global_alias_map()`), and global precedence by `hierarchy_position`.
5. **Masking** → `Masker` applies the final masking/redaction on the selected spans.
6. **Evaluation (optional)** → If an `Evaluator` is configured, metrics are calculated per document, logging results and supported types per component.

Each component is configurable and can be easily chained. For example, multiple detectors can be combined:

```
entity_detectors:
  - type: regex
    min_confidence: 0.3
  - type: gliner
  - type: spacy
```

### Instrumentation and Tracing

- **MetricStore**: when `metric_store.enabled` is active, the duration, detected/masked spans, and metadata such as supported types are recorded per component. If `save_config_json` is true, the flat pipeline configuration is saved in the execution directory. If the pipeline mode is offline and the input data has `ground_truth`, exhaustive evaluation metrics are calculated.
- **Configuration Preservation**: in addition, the source YAML/JSON files used to build the run are copied to the `configs/` subdirectory of the run, avoiding name collisions.
- **Logging**: at the `DEBUG` level, details of the resulting entities after masking are emitted to facilitate inspection and troubleshooting.

## Veil Configuration System

The configuration system:

- Allows a 1-1 mapping between configuration classes, real objects, and CLI parameters.
- Allows running Veil from configuration files, with the option of CLI overrides and argument collision validation.
- Adds CLI parameters automatically just by existing in the dependency tree of the root class (`PipelineConfig`).
- Allows running thousands of configurations from a single file. For example, this configuration:

```yaml
mode: offline
dataloader:
  - path: data/input/example_1.jsonl
  - path: data/input/example_2.jsonl
entity_detectors:
  - type: regex
    min_confidence: [0.3, 0.5, 0.7]
```

will run 6 instances of the offline pipeline, one for each combination of dataloader and detector. This is very useful for experimentation.

It achieves this by checking if a configuration file argument is a list (see `YAML syntax`).
- If it is and the corresponding configuration class is NOT typed as a list, it creates as many instances as there are values.
- If it is and the configuration class IS typed as a list, it creates a single instance with the list as the argument's value.

The configuration system includes advanced features such as:
- providing short names for arguments through the `arg_name` field in their metadata.
- running a Veil instance with multiple configuration files, where each can define a subset of the pipeline and allow combinations between them.

See `veil/config/core/flat_dataclass.py` for more details.