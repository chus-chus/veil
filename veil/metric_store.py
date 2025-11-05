import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from veil.config.metric_store import MetricStoreConfig
from veil.config.utils import dataclass_to_dict
from veil.core.span import Span
from veil.logger import init_logger

logger = init_logger(__name__)


class MetricStore:
    """File-system backed metric store initialization.

    Creates a unique run directory under the configured output directory to store
    all metric files for a single run, e.g.:

        <output_dir>/run-20250814T102234-ab12cd/
    """

    def __init__(self, config: MetricStoreConfig):
        self.config = config

        # ensure base output directory exists
        self.base_dir: Path = Path(self.config.output_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # unique per-run directory
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.run_id: str = f"run-{timestamp}-{uuid4().hex[:6]}"
        self.run_dir: Path = self.base_dir / self.run_id
        # uuid-based run_id should be unique; avoid overwriting by failing if exists
        self.run_dir.mkdir(parents=False, exist_ok=False)

        # --- In-memory metric state ---
        # Counters
        self.documents_processed_total: int = 0
        self.documents_failed_total: int = 0
        self.spans_detected_total: int = 0
        self.spans_masked_total: int = 0

        # Errors per component/error_type
        self.errors_total: Dict[str, int] = {}

        # Entity-type breakdowns
        self.detected_by_entity_type: Dict[str, int] = {}
        self.masked_by_entity_type: Dict[str, int] = {}

        # Component timings: key=f"{component_type}:{component}" -> list[seconds]
        self.component_durations_seconds: Dict[str, List[float]] = {}
        # Normalized by current document length (seconds per char)
        self.component_durations_per_char: Dict[str, List[float]] = {}

        # Per-component span counts (raw and per-char normalized totals)
        # key: component_key, value: {entity_type -> total_count}
        self.component_span_counts_raw: Dict[str, Dict[str, int]] = {}
        # key: component_key, value: {entity_type -> sum(count/doc_length)}
        self.component_span_counts_per_char: Dict[str, Dict[str, float]] = {}
        # Supported entity types per component (names)
        self.component_supported_entities: Dict[str, List[str]] = {}

        # Concurrency primitives
        self._lock: threading.Lock = threading.Lock()
        # Per-thread document context
        self._doc_ctx = threading.local()

        # Per-document durations to compute throughput stats
        self.document_durations_seconds: List[float] = []
        self.document_lengths_chars: List[int] = []

        # Run timing baseline
        self.run_start_ts: float = time.time()

        # --- Evaluation (confusion matrix) aggregates ---
        # Variant evaluations only (e.g., exact, iou@0.50, ...)
        # Structure: variant -> same four maps: per_component_by_type, per_component_all_types,
        # global_by_type, global_all_types
        self.eval_variants: Dict[
            str,
            Dict[
                str,
                Dict[str, Dict[str, int]] | Dict[str, int],
            ],
        ] = {}

        # Persist a minimal manifest
        self._write_json(
            self.path_in_run_dir("manifest.json"),
            {
                "run_id": self.run_id,
                "created_at_epoch": self.run_start_ts,
            },
        )

    def path_in_run_dir(self, *relative: str) -> Path:
        """Return a path within the current run directory.

        Example: metric_store.path_in_run_dir("component_times.jsonl")
        """
        return self.run_dir.joinpath(*relative)

    def save_config(self, config_obj: Any, filename: str = "config.json") -> Path:
        """Save a JSON-serialisable view of the pipeline config into the run dir."""
        config_dict = dataclass_to_dict(config_obj)

        target_path = self.path_in_run_dir(filename)
        with target_path.open("w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return target_path

    # Public recording API
    def start_document(self, doc_length_chars: int) -> None:
        # store per-thread context
        setattr(self._doc_ctx, "start_ts", time.time())
        setattr(self._doc_ctx, "doc_len", int(doc_length_chars))

    def end_document(self) -> None:
        start_ts = getattr(self._doc_ctx, "start_ts", None)
        doc_len = getattr(self._doc_ctx, "doc_len", None)
        if start_ts is None:
            return
        elapsed = time.time() - float(start_ts)
        # clear thread-local
        setattr(self._doc_ctx, "start_ts", None)
        setattr(self._doc_ctx, "doc_len", None)
        with self._lock:
            self.document_durations_seconds.append(elapsed)
            self.documents_processed_total += 1
            if doc_len is not None:
                self.document_lengths_chars.append(int(doc_len))
            self._write_metrics()

    def record_component_step(
        self,
        *,
        component: str,
        component_type: str,
        duration_seconds: float,
        detected_spans: Optional[Iterable[Span]] = None,
        masked_spans: Optional[Iterable[Span]] = None,
        supported_entity_types: Optional[Iterable[str]] = None,
    ) -> None:
        key = f"{component_type}:{component}"
        doc_len = getattr(self._doc_ctx, "doc_len", None)
        with self._lock:
            self.component_durations_seconds.setdefault(key, []).append(
                float(duration_seconds)
            )

            # Normalized duration per character if current document length is known and > 0
            if doc_len and int(doc_len) > 0:
                norm = float(duration_seconds) / float(int(doc_len))
                self.component_durations_per_char.setdefault(key, []).append(norm)

            if supported_entity_types is not None:
                sup_list = [str(s) for s in supported_entity_types]
                self.component_supported_entities[key] = sup_list

            if detected_spans is not None:
                total_count = 0
                per_type_counts: Dict[str, int] = {}
                for span in detected_spans:
                    etype = (
                        getattr(getattr(span, "entity_type", None), "name", None)
                        or "UNKNOWN"
                    )
                    # Only count supported types if we know them for this component
                    if (
                        key in self.component_supported_entities
                        and etype not in self.component_supported_entities[key]
                    ):
                        continue
                    per_type_counts[etype] = per_type_counts.get(etype, 0) + 1
                    self.detected_by_entity_type[etype] = (
                        self.detected_by_entity_type.get(etype, 0) + 1
                    )
                    total_count += 1
                # Update global total
                self.spans_detected_total += total_count
                # Update per-component totals
                raw_map = self.component_span_counts_raw.setdefault(key, {})
                norm_map = self.component_span_counts_per_char.setdefault(key, {})
                for etype, cnt in per_type_counts.items():
                    raw_map[etype] = raw_map.get(etype, 0) + cnt
                    if doc_len and int(doc_len) > 0:
                        norm_map[etype] = norm_map.get(etype, 0.0) + (
                            float(cnt) / float(int(doc_len))
                        )

            if masked_spans is not None:
                count = 0
                for span in masked_spans:
                    etype = (
                        getattr(getattr(span, "entity_type", None), "name", None)
                        or "UNKNOWN"
                    )
                    self.masked_by_entity_type[etype] = (
                        self.masked_by_entity_type.get(etype, 0) + 1
                    )
                    count += 1
                self.spans_masked_total += count

    def record_error(
        self, *, component: str, component_type: str, error_type: str
    ) -> None:
        key = f"{component_type}:{component}:{error_type}"
        with self._lock:
            self.errors_total[key] = self.errors_total.get(key, 0) + 1

    # Derived stats and snapshotting
    def overall_duration_seconds(self) -> float:
        return max(0.0, time.time() - self.run_start_ts)

    def mean_docs_per_second(self) -> float:
        total_time = sum(self.document_durations_seconds)
        if total_time <= 0.0:
            return 0.0
        return float(self.documents_processed_total) / total_time

    def _compute_quantiles(
        self, values: List[float], points: List[float]
    ) -> Dict[str, float]:
        if not values:
            return {f"p{int(p)}": 0.0 for p in points}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        out: Dict[str, float] = {}
        for p in points:
            # Nearest-rank method
            rank = max(1, math.ceil((p / 100.0) * n))
            out[f"p{int(p)}"] = float(sorted_vals[rank - 1])
        return out

    def _compute_min_max(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {"min": 0.0, "max": 0.0}
        return {"min": float(min(values)), "max": float(max(values))}

    def _compute_stats_pack(
        self, values: List[float], percentiles: List[float]
    ) -> Dict[str, float]:
        stats = {}
        stats.update(self._compute_min_max(values))
        stats.update(self._compute_quantiles(values, percentiles))
        return stats

    def _write_metrics(self) -> None:
        # Build component span metrics ensuring only supported entities are included when known
        component_spans: Dict[str, Dict[str, Dict[str, float | int]]] = {}
        for key, raw_counts in self.component_span_counts_raw.items():
            supported = self.component_supported_entities.get(key)
            # Filter maps based on supported entities if provided
            if supported is not None:
                raw_filtered = {
                    e: raw_counts.get(e, 0) for e in supported if e in raw_counts
                }
                norm_src = self.component_span_counts_per_char.get(key, {})
                norm_filtered = {
                    e: norm_src.get(e, 0.0) for e in supported if e in norm_src
                }
            else:
                raw_filtered = dict(raw_counts)
                norm_filtered = dict(self.component_span_counts_per_char.get(key, {}))
            component_spans[key] = {
                "raw_totals": raw_filtered,
                "per_char_totals": norm_filtered,
            }

        # Helper to compute precision/recall/F1 from counts
        def _derive_metrics(counts: Dict[str, int]) -> Dict[str, float | int]:
            tp = int(counts.get("tp", 0))
            fp = int(counts.get("fp", 0))
            fn = int(counts.get("fn", 0))
            precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
            recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall) > 0.0
                else 0.0
            )
            enriched = dict(counts)
            enriched.update(
                {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
            return enriched

        # Variants enriched
        eval_variants_enriched: Dict[
            str,
            Dict[
                str,
                Dict[str, Dict[str, float | int]] | Dict[str, float | int],
            ],
        ] = {}
        for variant, maps in self.eval_variants.items():
            per_comp_by_type_raw = maps.get("per_component_by_type", {})
            per_comp_all_types_raw = maps.get("per_component_all_types", {})
            global_by_type_raw = maps.get("global_by_type", {})
            global_all_types_raw = maps.get(
                "global_all_types", {"tp": 0, "fp": 0, "fn": 0}
            )

            per_comp_by_type_enr: Dict[str, Dict[str, Dict[str, float | int]]] = {}
            for comp, by_type in per_comp_by_type_raw.items():
                per_comp_by_type_enr[comp] = {
                    etype: _derive_metrics(counts) for etype, counts in by_type.items()
                }
            per_comp_all_types_enr: Dict[str, Dict[str, float | int]] = {
                comp: _derive_metrics(counts)
                for comp, counts in per_comp_all_types_raw.items()
            }
            global_by_type_enr: Dict[str, Dict[str, float | int]] = {
                etype: _derive_metrics(counts)
                for etype, counts in global_by_type_raw.items()
            }
            global_all_types_enr: Dict[str, float | int] = _derive_metrics(
                global_all_types_raw  # type: ignore[arg-type]
            )
            eval_variants_enriched[variant] = {
                "per_component_by_type": per_comp_by_type_enr,
                "per_component_all_types": per_comp_all_types_enr,
                "global_by_type": global_by_type_enr,
                "global_all_types": global_all_types_enr,
            }

        metrics = {
            # Counters
            "veil_pipeline_documents_processed_total": self.documents_processed_total,
            "veil_pipeline_documents_failed_total": self.documents_failed_total,
            "veil_pipeline_spans_detected_total": self.spans_detected_total,
            "veil_pipeline_spans_masked_total": self.spans_masked_total,
            # Durations
            "veil_pipeline_overall_duration_seconds": self.overall_duration_seconds(),
            # Entity-type breakdowns
            "veil_pipeline_detected_spans_by_type_total": self.detected_by_entity_type,
            "veil_pipeline_masked_spans_by_type_total": self.masked_by_entity_type,
            # Throughput
            "veil_pipeline_mean_docs_per_second": self.mean_docs_per_second(),
            # Component duration aggregates: raw seconds and per-char seconds
            "veil_pipeline_component_duration_seconds": {
                key: {
                    "raw_seconds": self._compute_stats_pack(values, [50, 95, 99]),
                    "per_char_seconds": self._compute_stats_pack(
                        self.component_durations_per_char.get(key, []), [50, 95, 99]
                    ),
                }
                for key, values in self.component_durations_seconds.items()
            },
            # Detected spans by component (entity detector): raw and per-char normalized totals
            "veil_pipeline_component_spans": component_spans,
            # Document length statistics (chars)
            "veil_pipeline_document_length_chars": self._compute_stats_pack(
                [float(x) for x in self.document_lengths_chars], [50, 90, 99]
            ),
            # Timestamping
            "run_id": self.run_id,
            "ts_epoch": time.time(),
            # Evaluation variants (only)
            "veil_eval_variants": eval_variants_enriched,
        }
        target = self.path_in_run_dir("metrics.json")
        with target.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
            f.write("\n")

    # Plotting / finalization
    def finalize(self) -> None:
        """Write final metrics and plots to the run directory."""
        logger.info("Finalizing metrics and plots...")
        # Always emit the latest metrics snapshot
        with self._lock:
            self._write_metrics()
        # Then generate plots from in-memory buffers
        self._generate_plots()
        # And finally, generate a compact evaluation summary document from metrics.json
        try:
            self._generate_evaluation_summary_docs()
        except Exception:
            logger.exception("Failed to write evaluation summary docs.")

    def _generate_plots(self) -> None:
        # Lazy-import plotting stack and use a non-interactive backend
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except Exception:
            # If plotting stack is unavailable, skip silently
            return

        plots_dir = self.path_in_run_dir("plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Subfolders for organization
        durations_dir = plots_dir / "durations"
        durations_components_dir = durations_dir / "components"
        entity_spans_dir = plots_dir / "entity_spans"
        entity_spans_components_dir = entity_spans_dir / "components"
        evaluation_dir = plots_dir / "evaluation"
        evaluation_aggregated_dir = evaluation_dir / "aggregated"
        evaluation_components_dir = evaluation_dir / "components"

        durations_dir.mkdir(parents=True, exist_ok=True)
        durations_components_dir.mkdir(parents=True, exist_ok=True)
        entity_spans_dir.mkdir(parents=True, exist_ok=True)
        entity_spans_components_dir.mkdir(parents=True, exist_ok=True)
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        evaluation_aggregated_dir.mkdir(parents=True, exist_ok=True)
        evaluation_components_dir.mkdir(parents=True, exist_ok=True)

        # Organized subfolders
        durations_dir = plots_dir / "durations"
        durations_components_dir = durations_dir / "components"
        entity_spans_dir = plots_dir / "entity_spans"
        entity_spans_components_dir = entity_spans_dir / "components"
        evaluation_dir = plots_dir / "evaluation"
        evaluation_aggregated_dir = evaluation_dir / "aggregated"
        evaluation_components_dir = evaluation_dir / "components"

        durations_components_dir.mkdir(parents=True, exist_ok=True)
        entity_spans_components_dir.mkdir(parents=True, exist_ok=True)
        evaluation_aggregated_dir.mkdir(parents=True, exist_ok=True)
        evaluation_components_dir.mkdir(parents=True, exist_ok=True)

        def _sanitize(name: str) -> str:
            return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)

        def _plot_cdf(
            values: List[float], title: str, xlabel: str, outfile: Path
        ) -> None:
            if not values:
                return
            xs = sorted(values)
            n = len(xs)
            ys = [i / n for i in range(1, n + 1)]
            plt.figure(figsize=(6, 4))
            plt.plot(xs, ys, drawstyle="steps-post")
            plt.xlabel(xlabel)
            plt.ylabel("CDF")
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(outfile)
            plt.close()

        def _plot_bars(
            mapping: Dict[str, float], title: str, xlabel: str, outfile: Path
        ) -> None:
            if not mapping:
                return
            labels = sorted(mapping.keys())
            values = [mapping[k] for k in labels]
            plt.figure(figsize=(max(6, len(labels) * 0.6), 4))
            plt.bar(labels, values)
            plt.xlabel(xlabel)
            plt.ylabel("value")
            plt.title(title)
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(outfile)
            plt.close()

        # 1) Document-level CDFs
        _plot_cdf(
            self.document_durations_seconds,
            title="Document processing time CDF",
            xlabel="seconds",
            outfile=durations_dir / "doc_duration_cdf.png",
        )
        _plot_cdf(
            [float(x) for x in self.document_lengths_chars],
            title="Document length CDF",
            xlabel="characters",
            outfile=durations_dir / "doc_length_cdf.png",
        )

        # 2) Component duration CDFs (per component and aggregated)
        # Aggregated raw durations (overlay per component)
        if self.component_durations_seconds:
            try:
                import matplotlib.pyplot as plt  # type: ignore[no-redef]

                plt.figure(figsize=(7, 5))
                for key, values in self.component_durations_seconds.items():
                    if not values:
                        continue
                    xs = sorted(values)
                    n = len(xs)
                    ys = [i / n for i in range(1, n + 1)]
                    plt.plot(xs, ys, drawstyle="steps-post", label=key)
                plt.xlabel("seconds")
                plt.ylabel("CDF")
                plt.title("Component duration CDFs (raw seconds)")
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize="small")
                plt.tight_layout()
                plt.savefig(durations_dir / "components_duration_cdf.png")
                plt.close()
            except Exception:
                pass

        # Aggregated per-char durations (overlay per component)
        if self.component_durations_per_char:
            try:
                import matplotlib.pyplot as plt  # type: ignore[no-redef]

                plt.figure(figsize=(7, 5))
                for key, values in self.component_durations_per_char.items():
                    if not values:
                        continue
                    xs = sorted(values)
                    n = len(xs)
                    ys = [i / n for i in range(1, n + 1)]
                    plt.plot(xs, ys, drawstyle="steps-post", label=key)
                plt.xlabel("seconds per char")
                plt.ylabel("CDF")
                plt.title("Component duration CDFs (normalized per char)")
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize="small")
                plt.tight_layout()
                plt.savefig(durations_dir / "components_per_char_duration_cdf.png")
                plt.close()
            except Exception:
                pass

        # Per-component individual CDFs
        for key, values in self.component_durations_seconds.items():
            _plot_cdf(
                values,
                title=f"{key} duration CDF",
                xlabel="seconds",
                outfile=durations_components_dir
                / f"comp_{_sanitize(key)}_duration_cdf.png",
            )
        for key, values in self.component_durations_per_char.items():
            _plot_cdf(
                values,
                title=f"{key} duration CDF (per char)",
                xlabel="seconds per char",
                outfile=durations_components_dir
                / f"comp_{_sanitize(key)}_per_char_duration_cdf.png",
            )

        # 2b) Horizontal stacked bars: end-to-end by component for different statistics
        if self.component_durations_seconds:
            try:
                import matplotlib.pyplot as plt  # type: ignore[no-redef]

                # Determine component order by numeric prefix in component id if present
                def _order_key(comp_key: str) -> int:
                    try:
                        # comp_key format: "type:NN-Name" -> extract NN
                        after_colon = comp_key.split(":", 1)[1]
                        idx_str = after_colon.split("-", 1)[0]
                        return int(idx_str)
                    except Exception:
                        return 10_000

                ordered_components = sorted(
                    self.component_durations_seconds.keys(), key=_order_key
                )

                # Compute stats per component
                comp_stats: Dict[str, Dict[str, float]] = {}
                for comp in ordered_components:
                    vals = self.component_durations_seconds.get(comp, [])
                    if not vals:
                        comp_stats[comp] = {
                            "min": 0.0,
                            "p50": 0.0,
                            "p95": 0.0,
                            "p99": 0.0,
                            "max": 0.0,
                            "mean": 0.0,
                        }
                        continue
                    stats_pack = self._compute_stats_pack(vals, [50, 95, 99])
                    comp_stats[comp] = {
                        "min": stats_pack.get("min", 0.0),
                        "p50": stats_pack.get("p50", 0.0),
                        "p95": stats_pack.get("p95", 0.0),
                        "p99": stats_pack.get("p99", 0.0),
                        "max": stats_pack.get("max", 0.0),
                        "mean": float(sum(vals)) / float(len(vals)),
                    }

                # Create stacked bars for each requested statistic
                stat_names = ["min", "p50", "p95", "p99", "max"]
                y_positions = list(range(len(stat_names)))

                plt.figure(figsize=(8, max(4, len(stat_names) * 0.8)))
                left = [0.0 for _ in stat_names]
                for comp in ordered_components:
                    widths = [comp_stats[comp][s] for s in stat_names]
                    plt.barh(y_positions, widths, left=left, label=comp)
                    left = [l + w for l, w in zip(left, widths)]

                plt.xlabel("seconds")
                plt.yticks(y_positions, stat_names)
                plt.title("Pipeline stacked component durations (per statistic)")
                plt.grid(True, axis="x", alpha=0.3)
                plt.legend(
                    fontsize="small",
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.1),
                    ncol=2,
                )
                plt.tight_layout()
                plt.savefig(durations_dir / "pipeline_stacked_component_durations.png")
                plt.close()
            except Exception:
                pass

        # 3) Entity totals bars – global and per component
        _plot_bars(
            {k: float(v) for k, v in self.detected_by_entity_type.items()},
            title="Detected spans by entity type (global)",
            xlabel="entity type",
            outfile=entity_spans_dir / "detected_spans_by_type.png",
        )
        _plot_bars(
            {k: float(v) for k, v in self.masked_by_entity_type.items()},
            title="Masked spans by entity type (global)",
            xlabel="entity type",
            outfile=entity_spans_dir / "masked_spans_by_type.png",
        )

        for key, raw_counts in self.component_span_counts_raw.items():
            _plot_bars(
                {k: float(v) for k, v in raw_counts.items()},
                title=f"{key} detected spans (raw totals)",
                xlabel="entity type",
                outfile=entity_spans_components_dir
                / f"comp_{_sanitize(key)}_span_totals.png",
            )
        for key, norm_counts in self.component_span_counts_per_char.items():
            _plot_bars(
                norm_counts,
                title=f"{key} detected spans (per char totals)",
                xlabel="entity type",
                outfile=entity_spans_components_dir
                / f"comp_{_sanitize(key)}_span_per_char_totals.png",
            )

        # 4) Quality (evaluation) plots
        def _derive(counts: Dict[str, int]) -> Dict[str, float]:
            tp = int(counts.get("tp", 0))
            fp = int(counts.get("fp", 0))
            fn = int(counts.get("fn", 0))
            precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
            recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall) > 0.0
                else 0.0
            )
            return {"precision": precision, "recall": recall, "f1": f1}

        # Helper: grouped bars (series overlay per group)
        def _plot_grouped_bars(
            group_labels: List[str],
            series_to_values: Dict[str, List[float]],
            title: str,
            xlabel: str,
            ylabel: str,
            outfile: Path,
        ) -> None:
            if not group_labels or not series_to_values:
                return
            try:
                import matplotlib.pyplot as plt  # type: ignore[no-redef]

                num_groups = len(group_labels)
                series_names = list(series_to_values.keys())
                num_series = len(series_names)
                x = list(range(num_groups))
                total_group_width = 0.8
                bar_width = total_group_width / max(1, num_series)
                offsets = [
                    (-total_group_width / 2) + (i + 0.5) * bar_width
                    for i in range(num_series)
                ]

                plt.figure(figsize=(max(6, len(group_labels) * 0.6), 4))
                for idx, sname in enumerate(series_names):
                    vals = series_to_values.get(sname, [])
                    if len(vals) != num_groups:
                        # pad or trim to fit
                        vals = (vals + [0.0] * num_groups)[:num_groups]
                    xs = [xi + offsets[idx] for xi in x]
                    plt.bar(xs, vals, width=bar_width, label=sname)

                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)
                plt.xticks(x, group_labels, rotation=45, ha="right")
                plt.grid(True, axis="y", alpha=0.3)
                plt.legend(fontsize="small")
                plt.tight_layout()
                plt.savefig(outfile)
                plt.close()
            except Exception:
                pass

        # Order variants: IoU thresholds ascending (by-id after type-only), then exact_by_id, then exact last
        def _variant_sort_key(vname: str) -> Tuple[int, float, int]:
            name = vname.lower()
            try:
                if name.startswith("iou@"):
                    suffix = name.split("@", 1)[1]
                    if "_by_id" in suffix:
                        thr = float(suffix.split("_", 1)[0])
                        return (0, thr, 1)
                    thr = float(suffix)
                    return (0, thr, 0)
            except Exception:
                pass
            if name == "exact_by_id":
                return (1, 1e8, 1)
            if name == "exact":
                return (1, 1e9, 0)
            return (2, 0.0, 0)

        variant_names = sorted(list(self.eval_variants.keys()), key=_variant_sort_key)
        # Split into type-only vs by-id groups
        type_only_variants = [
            v
            for v in variant_names
            if not v.lower().endswith("_by_id") and v != "exact_by_id"
        ]
        by_id_variants = [
            v
            for v in variant_names
            if v.lower().endswith("_by_id") or v == "exact_by_id"
        ]

        # Global by type, combined across variants
        # Build union of entity types
        all_types_set: set[str] = set()
        for v in variant_names:
            by_type_map = self.eval_variants.get(v, {}).get("global_by_type", {})
            all_types_set.update(list(by_type_map.keys()))
        type_labels = sorted(list(all_types_set))
        # For each metric, build series per variant
        for metric_name in ("f1", "precision", "recall"):
            # Type-only plot
            series: Dict[str, List[float]] = {}
            for v in type_only_variants:
                by_type_map = self.eval_variants.get(v, {}).get("global_by_type", {})
                vals: List[float] = []
                for et in type_labels:
                    counts = by_type_map.get(et)
                    if not counts:
                        vals.append(0.0)
                    else:
                        derived = _derive(counts)
                        vals.append(float(derived.get(metric_name, 0.0)))
                series[v] = vals
            _plot_grouped_bars(
                type_labels,
                series,
                title=f"Global {metric_name.upper()} by entity type (type-only)",
                xlabel="entity type",
                ylabel=metric_name,
                outfile=evaluation_aggregated_dir
                / f"global_{metric_name}_by_type_type_only.png",
            )
            # By-ID plot
            if by_id_variants:
                series_id: Dict[str, List[float]] = {}
                for v in by_id_variants:
                    by_type_map = self.eval_variants.get(v, {}).get(
                        "global_by_type", {}
                    )
                    vals: List[float] = []
                    for et in type_labels:
                        counts = by_type_map.get(et)
                        if not counts:
                            vals.append(0.0)
                        else:
                            derived = _derive(counts)
                            vals.append(float(derived.get(metric_name, 0.0)))
                    series_id[v] = vals
                _plot_grouped_bars(
                    type_labels,
                    series_id,
                    title=f"Global {metric_name.upper()} by entity type (by-id)",
                    xlabel="entity type",
                    ylabel=metric_name,
                    outfile=evaluation_aggregated_dir
                    / f"global_{metric_name}_by_type_by_id.png",
                )

        # Global all types: separate plots for type-only and by-id
        metrics = ("precision", "recall", "f1")
        group_labels = list(metrics)
        # Type-only
        series_values: Dict[str, List[float]] = {v: [] for v in type_only_variants}
        for v in type_only_variants:
            global_all = self.eval_variants.get(v, {}).get(
                "global_all_types", {"tp": 0, "fp": 0, "fn": 0}
            )
            derived = _derive(global_all)  # type: ignore[arg-type]
            for m in metrics:
                series_values[v].append(float(derived.get(m, 0.0)))
        _plot_grouped_bars(
            group_labels,
            series_values,
            title="Global quality metrics (type-only)",
            xlabel="metric",
            ylabel="value",
            outfile=evaluation_aggregated_dir
            / "global_quality_all_types_type_only.png",
        )
        # By-ID
        if by_id_variants:
            series_values_id: Dict[str, List[float]] = {v: [] for v in by_id_variants}
            for v in by_id_variants:
                global_all = self.eval_variants.get(v, {}).get(
                    "global_all_types", {"tp": 0, "fp": 0, "fn": 0}
                )
                derived = _derive(global_all)  # type: ignore[arg-type]
                for m in metrics:
                    series_values_id[v].append(float(derived.get(m, 0.0)))
            _plot_grouped_bars(
                group_labels,
                series_values_id,
                title="Global quality metrics (by-id)",
                xlabel="metric",
                ylabel="value",
                outfile=evaluation_aggregated_dir
                / "global_quality_all_types_by_id.png",
            )

        # Per component plots, combined across variants
        # 1) By type: F1, precision, recall
        comp_keys: set[str] = set()
        for v in variant_names:
            comp_keys.update(
                self.eval_variants.get(v, {}).get("per_component_by_type", {}).keys()
            )
        for comp in sorted(comp_keys):
            # union of types for this component
            type_labels_comp: set[str] = set()
            for v in variant_names:
                by_type = (
                    self.eval_variants.get(v, {})
                    .get("per_component_by_type", {})
                    .get(comp, {})
                )
                type_labels_comp.update(list(by_type.keys()))
            type_labels_list = sorted(list(type_labels_comp))
            for metric_name in ("f1", "precision", "recall"):
                # Type-only
                series: Dict[str, List[float]] = {}
                for v in type_only_variants:
                    by_type = (
                        self.eval_variants.get(v, {})
                        .get("per_component_by_type", {})
                        .get(comp, {})
                    )
                    vals: List[float] = []
                    for et in type_labels_list:
                        counts = by_type.get(et)
                        if not counts:
                            vals.append(0.0)
                        else:
                            derived = _derive(counts)
                            vals.append(float(derived.get(metric_name, 0.0)))
                    series[v] = vals
                _plot_grouped_bars(
                    type_labels_list,
                    series,
                    title=f"{comp} {metric_name.upper()} by entity type (type-only)",
                    xlabel="entity type",
                    ylabel=metric_name,
                    outfile=evaluation_components_dir
                    / f"comp_{_sanitize(comp)}_{metric_name}_by_type_type_only.png",
                )
                # By-ID
                if by_id_variants:
                    series_id: Dict[str, List[float]] = {}
                    for v in by_id_variants:
                        by_type = (
                            self.eval_variants.get(v, {})
                            .get("per_component_by_type", {})
                            .get(comp, {})
                        )
                        vals: List[float] = []
                        for et in type_labels_list:
                            counts = by_type.get(et)
                            if not counts:
                                vals.append(0.0)
                            else:
                                derived = _derive(counts)
                                vals.append(float(derived.get(metric_name, 0.0)))
                        series_id[v] = vals
                    _plot_grouped_bars(
                        type_labels_list,
                        series_id,
                        title=f"{comp} {metric_name.upper()} by entity type (by-id)",
                        xlabel="entity type",
                        ylabel=metric_name,
                        outfile=evaluation_components_dir
                        / f"comp_{_sanitize(comp)}_{metric_name}_by_type_by_id.png",
                    )

        # 2) Per component all types (group by metric, series = variants with exact last)
        comp_keys_all: set[str] = set()
        for v in variant_names:
            comp_keys_all.update(
                self.eval_variants.get(v, {}).get("per_component_all_types", {}).keys()
            )
        for comp in sorted(comp_keys_all):
            # Type-only
            series_values: Dict[str, List[float]] = {
                vn: [] for vn in type_only_variants
            }
            for v in type_only_variants:
                counts = (
                    self.eval_variants.get(v, {})
                    .get("per_component_all_types", {})
                    .get(comp, {"tp": 0, "fp": 0, "fn": 0})
                )
                derived = _derive(counts)
                for m in metrics:
                    # Will be appended per variant series for each metric group later
                    series_values[v].append(float(derived.get(m, 0.0)))
            # Transpose series_values to have groups=metrics, series=variants
            # series_values currently: variant -> [prec, rec, f1]
            # _plot_grouped_bars expects: series -> list aligned with group_labels
            _plot_grouped_bars(
                list(metrics),
                series_values,
                title=f"{comp} quality metrics (type-only)",
                xlabel="metric",
                ylabel="value",
                outfile=evaluation_components_dir
                / f"comp_{_sanitize(comp)}_quality_all_types_type_only.png",
            )
            # By-ID
            if by_id_variants:
                series_values_id: Dict[str, List[float]] = {
                    vn: [] for vn in by_id_variants
                }
                for v in by_id_variants:
                    counts = (
                        self.eval_variants.get(v, {})
                        .get("per_component_all_types", {})
                        .get(comp, {"tp": 0, "fp": 0, "fn": 0})
                    )
                    derived = _derive(counts)
                    for m in metrics:
                        # Will be appended per variant series for each metric group later
                        series_values_id[v].append(float(derived.get(m, 0.0)))
                _plot_grouped_bars(
                    list(metrics),
                    series_values_id,
                    title=f"{comp} quality metrics (by-id)",
                    xlabel="metric",
                    ylabel="value",
                    outfile=evaluation_components_dir
                    / f"comp_{_sanitize(comp)}_quality_all_types_by_id.png",
                )

    def _generate_evaluation_summary_docs(self) -> None:
        """Create small Markdown and HTML summaries of global evaluation metrics.

        Reads the persisted metrics.json to ensure summaries reflect saved output.
        Produces:
          - evaluation_summary.md
          - evaluation_summary.html
        in the run directory.
        """
        metrics_path = self.path_in_run_dir("metrics.json")
        if not metrics_path.exists():
            return
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return

        run_id = str(payload.get("run_id", ""))
        variants = payload.get("veil_eval_variants") or {}
        if not isinstance(variants, dict) or not variants:
            return

        def _vsort_key(vname: str) -> Tuple[int, float, int]:
            name = str(vname).lower()
            try:
                if name.startswith("iou@"):
                    suffix = name.split("@", 1)[1]
                    if "_by_id" in suffix:
                        thr = float(suffix.split("_", 1)[0])
                        return (0, thr, 1)
                    thr = float(suffix)
                    return (0, thr, 0)
            except Exception:
                pass
            if name == "exact_by_id":
                return (1, 1e8, 1)
            if name == "exact":
                return (1, 1e9, 0)
            return (2, 0.0, 0)

        ordered_variants = sorted(list(variants.keys()), key=_vsort_key)
        type_only_variants = [
            v
            for v in ordered_variants
            if not str(v).lower().endswith("_by_id") and str(v).lower() != "exact_by_id"
        ]
        by_id_variants = [
            v
            for v in ordered_variants
            if str(v).lower().endswith("_by_id") or str(v).lower() == "exact_by_id"
        ]

        # Collect union of entity types present across variants
        all_types: set[str] = set()
        for v in ordered_variants:
            by_type = (variants.get(v) or {}).get("global_by_type") or {}
            if isinstance(by_type, dict):
                all_types.update([str(k) for k in by_type.keys()])
        type_labels = sorted(all_types)

        def _num(x: Any, dflt: float = 0.0) -> float:
            try:
                return float(x)
            except Exception:
                return dflt

        # Build Markdown
        md_lines: List[str] = []
        md_lines.append(f"# Evaluation summary\n")
        if run_id:
            md_lines.append(f"Run: `{run_id}`\n")
        md_lines.append("\n")

        # Global F1 by entity type (type-only)
        md_lines.append("## Global F1 by entity type (type-only)\n")
        if type_only_variants and type_labels:
            headers = ["Entity type"] + [str(v) for v in type_only_variants]
            md_lines.append("| " + " | ".join(headers) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for et in type_labels:
                row: List[str] = [et]
                for v in type_only_variants:
                    f1 = _num(
                        ((variants.get(v) or {}).get("global_by_type") or {})
                        .get(et, {})
                        .get("f1", 0.0)
                    )
                    row.append(f"{f1:.3f}")
                md_lines.append("| " + " | ".join(row) + " |")
        else:
            md_lines.append("_No type-only variants available._\n")
        md_lines.append("\n")

        # Global F1 by entity type (by-id)
        md_lines.append("## Global F1 by entity type (by-id)\n")
        if by_id_variants and type_labels:
            headers = ["Entity type"] + [str(v) for v in by_id_variants]
            md_lines.append("| " + " | ".join(headers) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for et in type_labels:
                row = [et]
                for v in by_id_variants:
                    f1 = _num(
                        ((variants.get(v) or {}).get("global_by_type") or {})
                        .get(et, {})
                        .get("f1", 0.0)
                    )
                    row.append(f"{f1:.3f}")
                md_lines.append("| " + " | ".join(row) + " |")
        else:
            md_lines.append("_No by-id variants available._\n")
        md_lines.append("\n")

        # Global quality metrics (all types) - type-only
        md_lines.append("## Global quality metrics (all types, type-only)\n")
        if type_only_variants:
            md_lines.append("| Variant | TP | FP | FN | P | R | F1 |")
            md_lines.append("| --- | ---:| ---:| ---:| ---:| ---:| ---:|")
            for v in type_only_variants:
                counts = (variants.get(v) or {}).get("global_all_types") or {}
                tp = int(counts.get("tp", 0))
                fp = int(counts.get("fp", 0))
                fn = int(counts.get("fn", 0))
                p = _num(counts.get("precision", 0.0))
                r = _num(counts.get("recall", 0.0))
                f1 = _num(counts.get("f1", 0.0))
                md_lines.append(
                    f"| {v} | {tp} | {fp} | {fn} | {p:.3f} | {r:.3f} | {f1:.3f} |"
                )
        else:
            md_lines.append("_No type-only variants available._\n")
        md_lines.append("\n")

        # Global quality metrics (all types) - by-id
        md_lines.append("## Global quality metrics (all types, by-id)\n")
        if by_id_variants:
            md_lines.append("| Variant | TP | FP | FN | P | R | F1 |")
            md_lines.append("| --- | ---:| ---:| ---:| ---:| ---:| ---:|")
            for v in by_id_variants:
                counts = (variants.get(v) or {}).get("global_all_types") or {}
                tp = int(counts.get("tp", 0))
                fp = int(counts.get("fp", 0))
                fn = int(counts.get("fn", 0))
                p = _num(counts.get("precision", 0.0))
                r = _num(counts.get("recall", 0.0))
                f1 = _num(counts.get("f1", 0.0))
                md_lines.append(
                    f"| {v} | {tp} | {fp} | {fn} | {p:.3f} | {r:.3f} | {f1:.3f} |"
                )
        else:
            md_lines.append("_No by-id variants available._\n")
        md_lines.append("\n")

        md_text = "\n".join(md_lines) + "\n"
        md_path = self.path_in_run_dir("evaluation_summary.md")
        with md_path.open("w", encoding="utf-8") as f:
            f.write(md_text)

        # Basic HTML rendition
        def _html_escape(s: str) -> str:
            try:
                import html as _html

                return _html.escape(s)
            except Exception:
                return s

        def _md_table_to_html(md_table_lines: List[str]) -> str:
            rows: List[str] = []
            for idx, line in enumerate(md_table_lines):
                if not line.startswith("| ") or " |" not in line:
                    continue
                parts = [p.strip() for p in line.strip().strip("|").split("|")]
                if idx == 0:
                    # header
                    ths = "".join([f"<th>{_html_escape(p)}</th>" for p in parts])
                    rows.append(f"<thead><tr>{ths}</tr></thead>")
                elif idx == 1:
                    # separator row, skip
                    continue
                else:
                    tds = "".join([f"<td>{_html_escape(p)}</td>" for p in parts])
                    rows.append(f"<tr>{tds}</tr>")
            body_rows = [r for r in rows if not r.startswith("<thead>")]
            thead = next((r for r in rows if r.startswith("<thead>")), "")
            return f"<table>{thead}<tbody>{''.join(body_rows)}</tbody></table>"

        # Extract individual table blocks from markdown sections
        def _extract_section(title: str) -> List[str]:
            out: List[str] = []
            started = False
            for ln in md_lines:
                if ln.strip() == title:
                    started = True
                    continue
                if started and ln.startswith("## "):
                    break
                if started:
                    out.append(ln)
            return [l for l in out if l.strip()]

        html_parts: List[str] = []
        html_parts.append(
            "<html><head><meta charset='utf-8'><style>body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:16px}h1,h2{margin:0 0 8px}table{border-collapse:collapse;margin:8px 0}th,td{border:1px solid #ddd;padding:6px 8px;font-size:13px}th{background:#f7f7f7;text-align:left}</style></head><body>"
        )
        html_parts.append(f"<h1>Evaluation summary</h1>")
        if run_id:
            html_parts.append(
                f"<div><strong>Run:</strong> <code>{_html_escape(run_id)}</code></div>"
            )

        def _section_to_html(sec_title_md: str, heading_html: str) -> None:
            tbl = _extract_section(sec_title_md)
            if not any(l.startswith("| ") for l in tbl):
                html_parts.append(
                    f"<h2>{heading_html}</h2><div><em>No data.</em></div>"
                )
                return
            html_parts.append(f"<h2>{heading_html}</h2>")
            html_parts.append(_md_table_to_html([l for l in tbl if l.startswith("| ")]))

        _section_to_html(
            "## Global F1 by entity type (type-only)",
            "Global F1 by entity type (type-only)",
        )
        _section_to_html(
            "## Global F1 by entity type (by-id)", "Global F1 by entity type (by-id)"
        )
        _section_to_html(
            "## Global quality metrics (all types, type-only)",
            "Global quality metrics (all types, type-only)",
        )
        _section_to_html(
            "## Global quality metrics (all types, by-id)",
            "Global quality metrics (all types, by-id)",
        )

        html_parts.append("</body></html>")
        html_text = "".join(html_parts)
        html_path = self.path_in_run_dir("evaluation_summary.html")
        with html_path.open("w", encoding="utf-8") as f:
            f.write(html_text)

    # Utils
    def _write_json(self, path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

    # -----------------------------
    # Evaluation recording API
    # -----------------------------
    def _merge_confusion(self, a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
        return {
            "tp": int(a.get("tp", 0)) + int(b.get("tp", 0)),
            "fp": int(a.get("fp", 0)) + int(b.get("fp", 0)),
            "fn": int(a.get("fn", 0)) + int(b.get("fn", 0)),
        }

    def record_evaluation_variant(
        self,
        *,
        variant: str,
        per_component_by_type: Dict[str, Dict[str, Dict[str, int]]],
        per_component_all_types: Dict[str, Dict[str, int]],
        global_by_type: Dict[str, Dict[str, int]],
        global_all_types: Dict[str, int],
    ) -> None:
        """Record evaluation counts for a named variant (e.g., 'exact', 'iou@0.50')."""
        with self._lock:
            v = self.eval_variants.setdefault(
                str(variant),
                {
                    "per_component_by_type": {},
                    "per_component_all_types": {},
                    "global_by_type": {},
                    "global_all_types": {"tp": 0, "fp": 0, "fn": 0},
                },
            )

            # Per-component by-type
            v_pcbt = v["per_component_by_type"]  # type: ignore[index]
            for comp, mapping in per_component_by_type.items():
                comp_map = v_pcbt.setdefault(comp, {})  # type: ignore[assignment]
                for etype, counts in mapping.items():
                    existing = comp_map.get(etype, {"tp": 0, "fp": 0, "fn": 0})
                    comp_map[etype] = self._merge_confusion(existing, counts)

            # Per-component all-types
            v_pcat = v["per_component_all_types"]  # type: ignore[index]
            for comp, counts in per_component_all_types.items():
                existing = v_pcat.get(comp, {"tp": 0, "fp": 0, "fn": 0})
                v_pcat[comp] = self._merge_confusion(existing, counts)

            # Global by-type
            v_gbt = v["global_by_type"]  # type: ignore[index]
            for etype, counts in global_by_type.items():
                existing = v_gbt.get(etype, {"tp": 0, "fp": 0, "fn": 0})
                v_gbt[etype] = self._merge_confusion(existing, counts)

            # Global all-types
            v_gat = v["global_all_types"]  # type: ignore[index]
            v["global_all_types"] = self._merge_confusion(v_gat, global_all_types)  # type: ignore[index]
