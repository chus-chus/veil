# Declaring Pipelines

This section explains how to declare pipelines in Veil using configuration files (YAML/JSON) and the CLI. After reading it, you will be able to build any pipeline supported by default: offline/online modes, combining entity detectors, resolvers (optional), overlap resolution, and masking.

- See the configuration classes API at: [API Reference › veil.config](api/veil.config.md)
- Pipeline implementation: [API Reference › veil.pipeline](api/veil.pipeline.md)

## Basic Concepts

A pipeline is described by a {py:class}`veil.config.PipelineConfig` object and its children. The main fields are:

- `mode` (offline|online): execution mode. Online mode exposes an API; offline mode processes in batches.
- `datahandler` (offline only): input/output configuration ({py:class}`veil.config.DataHandlerConfig`).
- `api_server` (online only): API server configuration ({py:class}`veil.config.ApiServerConfig`).
- `entity_detectors` (list): entity detectors executed in order (child configs of {py:class}`veil.config.BaseEntityDetectorConfig`).
- `entity_resolvers` (list, optional): resolvers that refine/join entities after detection (child configs of {py:class}`veil.config.BaseEntityResolverConfig`).
- `overlap_resolver`: policy for resolving overlaps between spans from different detectors ({py:class}`veil.config.OverlapResolverConfig`).
- `masker`: final masking strategy ({py:class}`veil.config.MaskerConfig`).
- `metric_store`: instrumentation and saving of metrics/results ({py:class}`veil.config.MetricStoreConfig`).
- `evaluator` (optional): evaluation against ground truth ({py:class}`veil.config.EvaluatorConfig`).
- `concurrency`: parallelism in offline mode.
- `log_level`: global logging level.

You can pass this configuration from a file with:

```bash
python3 -m veil --pipeline-config-from-file run_configs/my_pipeline.yml
```

Or directly from the CLI (the parameters of `PipelineConfig` and its children are exposed as flags and options available for configuration files).

### Quick Component References (API)

- Detectors: [Reference › veil.entity_detectors](api/veil.entity_detectors.md)
- Resolvers: [Reference › veil.entity_resolvers](api/veil.entity_resolvers.md)
- Masker: [Reference › veil.masker](api/veil.masker.md)
- Root Config: {py:class}`veil.config.PipelineConfig`

## Polymorphic Configuration with `type`

Several subcomponents are polymorphic and are selected with a `type` field. Examples:

- `entity_detectors[*]`: each element is a child configuration of `BaseEntityDetectorConfig` selected by `type`.
  - Out-of-the-box supported types and their classes:
    - `REGEX` → {py:class}`veil.config.RegexEntityDetectorConfig`
    - `GLINER` → {py:class}`veil.config.GlinerEntityDetectorConfig`
    - `SPACY` → {py:class}`veil.config.SpacyEntityDetectorConfig`
    - `MASKER_API` → {py:class}`veil.config.MaskerApiEntityDetectorConfig`
    - `HOSTED_MASKER_API` → {py:class}`veil.config.HostedMaskerApiEntityDetectorConfig`
- `entity_resolvers[*]`: children of `BaseEntityResolverConfig`, e.g. `EMBEDDINGS` → {py:class}`veil.config.EmbeddingsEntityResolverConfig`.
- `masker.method`: selects the masking type according to `MaskType` (e.g. `ENTITY_TAG`) with {py:class}`veil.config.MaskerConfig`.

How it works:
- If a block is polymorphic, include `type: ...` and the rest of the subtype-specific fields.
- The `type` keys are case-insensitive and must be provided as a string.
- If `type` is invalid, the valid values are reported in the error.

Minimal example of a polymorphic detector:

```yaml
entity_detectors:
  - type: regex
    min_confidence: 0.3
```

## Detectors: Order, Hierarchy, and Priority by Type

When you combine several detectors, Veil can return overlapping spans of the same type. The final selection is made by `OverlapResolver` by combining:

- Priority by type (per detector): `priority` is a map `CANONICAL_TYPE -> integer`. Lower numbers win (0 is maximum priority).
- Component hierarchy: `hierarchy_position` is an integer per detector; lower numbers have global precedence.
- IoU: two spans of the same type conflict if `IoU > iou_threshold` (default 0.75). IoU measures the intersection over the union of lengths.

Selection rules (same canonical type):
1) The span with the lower priority by type wins (0 > 1 > 2 ...).
2) If they tie, the one with the lower `hierarchy_position` from the detector wins.
3) If the tie persists, it is decided deterministically: higher `confidence`, then longer span, then earlier start.
4) Overlaps between different types are allowed; they are not considered a conflict.

The `priority` keys are normalized to canonical types with `EntityTypeBase.global_alias_map()`. For example, `Name`, `NAME`, and `name` are normalized to the same type.

## Key Parameters per Component

- Common detectors:
  - {py:class}`veil.config.RegexEntityDetectorConfig`: `min_confidence`, `enable_validation`, `case_sensitive`, `preserve_format`, plus `priority` and `hierarchy_position`.
  - {py:class}`veil.config.GlinerEntityDetectorConfig`: `model`, `labels`, `threshold`, `batch_size`, `cuda_device`, `max_length`, `chunk_overlap`, and post-processing (`nms_iou_threshold`, size limits, `top_k_per_chunk`).
  - {py:class}`veil.config.SpacyEntityDetectorConfig`: `model`, `cuda_device`.
  - {py:class}`veil.config.MaskerApiEntityDetectorConfig` / {py:class}`veil.config.HostedMaskerApiEntityDetectorConfig`: `api_url`, `model`, `system_prompt`, `timeout`, `retries`, etc.
- Resolvers:
  - {py:class}`veil.config.EmbeddingsEntityResolverConfig`: similarity `threshold` and `context_chars`.
- Overlap:
  - {py:class}`veil.config.OverlapResolverConfig`: `iou_threshold` (0.0–<1.0).
- Masker:
  - {py:class}`veil.config.MaskerConfig`: `method` (`ENTITY_TAG`, etc.).
- Metrics:
  - {py:class}`veil.config.MetricStoreConfig`: `enabled`, `output_dir`, `save_config_json`.
- Evaluation:
  - {py:class}`veil.config.EvaluatorConfig`: `enabled` and report options; requires `ground_truth` to calculate metrics.

## CLI Parameters and File Format (--pipeline-config-from-file)

Veil automatically exposes CLI flags for all fields in `PipelineConfig` and its children. In addition, it adds a special flag to load configuration from a file:

- `--pipeline-config-from-file <path>`: accepts a YAML or JSON file. The content is merged with the CLI flags; explicit CLI flags take precedence.

Supported file format:
- YAML or JSON. If the top level is a dictionary (mapping), it is interpreted as a single configuration.
- If the top level is a list, it is wrapped as `{"_list": [...]}` internally and combinations are expanded according to list rules.
- Any field can be a list to generate Cartesian combinations, unless the field in the dataclass is already of type list (in which case it is taken as a literal list and is NOT Cartesianized).
- Polymorphic fields: use `type` to select the subtype and add its specific keys.
- Unknown keys produce an error; they are validated against the dataclasses.

Example YAML file with combinations:

```yaml
mode: offline
entity_detectors:
  - type: regex
    min_confidence: [0.3, 0.5]
  - type: gliner
    threshold: [0.4, 0.6]
```

This file will execute 4 combinations (2×2) in offline mode.

Types of arguments in CLI (rules):
- Primitives (str, int, float, bool): `--log-level DEBUG`, `--concurrency 4`.
- Lists of primitives: `--labels name company address` or by passing JSON/YAML in a single string for lists of objects.
- Dictionaries of primitives: pass JSON: `--priority '{"COMPANY": 0, "NAME": 1}'`.
- Dataclasses or lists of dataclasses/polys: pass JSON as a string: `--entity-detectors '[{"type":"regex","min_confidence":0.3}]'`.

Loader errors and validations:
- Non-existent or unreadable file: clear error.
- Invalid YAML: tries to parse as JSON; if it fails, shows the original YAML error.
- Top level is not a mapping or a list: error.
- `iou_threshold` out of range [0.0, 1.0): validation error.

## Complete Examples

### Production API (online)

Based on `run_configs/prod_pipeline_v1.yml`:

```yaml
mode: online
log_level: info
concurrency: 1

entity_detectors:

  - type: hosted_masker_api
    api_url: https://api.fireworks.ai/inference/v1/chat/completions
    model: accounts/sample/model-name
    timeout: 120
    headers: { "Accept": "application/json", "Content-Type": "application/json", "Authorization": "Bearer <token>"}
    priority:
      NAME: 0
      ADDRESS: 0
    system_prompt: >
      Please anonymize the following text:

  - type: gliner
    hierarchy_position: 1
    model: urchade/gliner_multi-v2.1
    labels: ["empresa"]
    threshold: 0.4
    batch_size: 8
    cuda_device: 0
    max_length: 384
    chunk_overlap: 100
    priority:
      COMPANY: 0

  - type: regex
    min_confidence: 0.3
    hierarchy_position: 2
    priority:
      DNI: 0
      CIF: 0
      NIE: 0
      NSS: 0
      EMAIL: 0
      PHONE: 0
      IBAN: 0
      IPV4: 0
      IPV6: 0

overlap_resolver:
  iou_threshold: 0.75

entity_resolvers:
  - type: embeddings
    threshold: 0.75

masker:
  method: ENTITY_TAG
```

Notes:
- `HOSTED_MASKER_API` prioritizes `NAME` and `ADDRESS` with 0; `GLINER` reinforces only `COMPANY` with 0. By canonical type, 0 wins; if both are 0, `hierarchy_position` breaks the tie (HOSTED_MASKER_API=0 before GLINER=1).
- The embeddings resolver adds/joins entity ids before masking.

### Simple Offline Pipeline with Regex

```yaml
mode: offline
log_level: DEBUG

# datahandler required in offline (generic example)
datahandler:
  input_path: data/input/example.jsonl
  output_path: data/output/masked.jsonl

entity_detectors:
  - type: regex
    min_confidence: 0.5
    priority:
      EMAIL: 0
      PHONE: 0

overlap_resolver:
  iou_threshold: 0.0  # considers any overlap as a conflict

masker:
  method: ENTITY_TAG

metric_store:
  enabled: true
  output_dir: veil_runs/local
  save_config_json: true
```

## Best Practices

- Define `priority` only for the types you are interested in prioritizing; undefined ones take a high (worse) priority.
- Use `hierarchy_position` to express global confidence per detector, and `priority` for granularity by type.
- Adjust `iou_threshold` if you see almost identical spans that should be merged or coexist.
- In `GLINER`, increase `threshold` if there are too many false positives; decrease if recall is lacking.
- Activate `metric_store` and, if you have `ground_truth`, the `evaluator` to iterate with real data.

## Common Errors

- Unknown `type` in a polymorphic block: check the list of supported types and capitalization.
- Forgetting `api_server` in `mode: online`: one will be created by default, but it is advisable to adjust it for production.
- In `mode: offline`, `datahandler` is missing: it is mandatory.
