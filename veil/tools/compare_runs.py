from __future__ import annotations

import argparse
import json
import re
import sys
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RunMetrics:
    run_id: str
    path: Path
    docs_total: int
    docs_failed: int
    spans_detected_total: int
    spans_masked_total: int
    overall_seconds: float
    mean_docs_per_second: float
    component_durations_raw: Dict[str, Dict[str, float]]
    component_durations_per_char: Dict[str, Dict[str, float]]
    variants: Dict[str, Dict[str, Any]]


def load_metrics_json(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {run_dir}")
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_run_metrics(run_dir: Path) -> RunMetrics:
    payload = load_metrics_json(run_dir)
    run_id = payload.get("run_id") or run_dir.name
    # Durations
    comp = payload.get("veil_pipeline_component_duration_seconds", {})
    comp_raw: Dict[str, Dict[str, float]] = {}
    comp_pc: Dict[str, Dict[str, float]] = {}
    for key, pack in comp.items():
        raw = pack.get("raw_seconds", {}) or {}
        per_char = pack.get("per_char_seconds", {}) or {}
        comp_raw[key] = {
            "min": float(raw.get("min", 0.0)),
            "p50": float(raw.get("p50", 0.0)),
            "p95": float(raw.get("p95", 0.0)),
            "p99": float(raw.get("p99", 0.0)),
            "max": float(raw.get("max", 0.0)),
        }
        comp_pc[key] = {
            "min": float(per_char.get("min", 0.0)),
            "p50": float(per_char.get("p50", 0.0)),
            "p95": float(per_char.get("p95", 0.0)),
            "p99": float(per_char.get("p99", 0.0)),
            "max": float(per_char.get("max", 0.0)),
        }

    # Variants
    variants: Dict[str, Dict[str, Any]] = payload.get("veil_eval_variants", {}) or {}

    return RunMetrics(
        run_id=str(run_id),
        path=run_dir,
        docs_total=int(payload.get("veil_pipeline_documents_processed_total", 0)),
        docs_failed=int(payload.get("veil_pipeline_documents_failed_total", 0)),
        spans_detected_total=int(payload.get("veil_pipeline_spans_detected_total", 0)),
        spans_masked_total=int(payload.get("veil_pipeline_spans_masked_total", 0)),
        overall_seconds=float(
            payload.get("veil_pipeline_overall_duration_seconds", 0.0)
        ),
        mean_docs_per_second=float(
            payload.get("veil_pipeline_mean_docs_per_second", 0.0)
        ),
        component_durations_raw=comp_raw,
        component_durations_per_char=comp_pc,
        variants=variants,
    )


def _fmt_delta(curr: Optional[float], base: Optional[float], suffix: str = "") -> str:
    if curr is None or base is None:
        return "n/a"
    d = curr - base
    pct = None
    if abs(base) > 1e-12:
        pct = (d / base) * 100.0
    sign = "+" if d >= 0 else ""
    base_str = f"{base:.4g}{suffix}"
    curr_str = f"{curr:.4g}{suffix}"
    if pct is None:
        return f"{base_str} -> {curr_str} ({sign}{d:.4g}{suffix})"
    return f"{base_str} -> {curr_str} ({sign}{d:.4g}{suffix}, {sign}{pct:.2f}%)"


def compare_two_runs(baseline: RunMetrics, candidate: RunMetrics) -> Dict[str, Any]:
    out: Dict[str, Any] = {"baseline": baseline.run_id, "candidate": candidate.run_id}

    # Top-level performance
    out["performance"] = {
        "docs_total": {
            "baseline": baseline.docs_total,
            "candidate": candidate.docs_total,
            "delta": candidate.docs_total - baseline.docs_total,
        },
        "overall_seconds": {
            "baseline": baseline.overall_seconds,
            "candidate": candidate.overall_seconds,
            "delta": candidate.overall_seconds - baseline.overall_seconds,
        },
        "mean_docs_per_second": {
            "baseline": baseline.mean_docs_per_second,
            "candidate": candidate.mean_docs_per_second,
            "delta": candidate.mean_docs_per_second - baseline.mean_docs_per_second,
        },
    }

    # Component durations (p50, p95)
    comp_keys = sorted(
        set(baseline.component_durations_raw.keys())
        | set(candidate.component_durations_raw.keys())
    )
    comp: Dict[str, Any] = {}
    for key in comp_keys:
        b = baseline.component_durations_raw.get(key)
        c = candidate.component_durations_raw.get(key)
        comp[key] = {
            "p50": {
                "baseline": (b or {}).get("p50"),
                "candidate": (c or {}).get("p50"),
            },
            "p95": {
                "baseline": (b or {}).get("p95"),
                "candidate": (c or {}).get("p95"),
            },
        }
    out["component_durations_seconds"] = comp

    # Quality: per-variant comparison (global all types and by type)
    def extract_variant(
        vmap: Dict[str, Any], v: str
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        pack = vmap.get(v, {})
        global_all = {
            k: float(val)
            for k, val in (pack.get("global_all_types", {}) or {}).items()
            if isinstance(val, (int, float))
        }
        by_type = {
            et: {
                k: float(val)
                for k, val in vals.items()
                if k in ("precision", "recall", "f1")
            }
            for et, vals in (pack.get("global_by_type", {}) or {}).items()
        }
        return global_all, by_type

    variants = sorted(set(baseline.variants.keys()) | set(candidate.variants.keys()))
    qa_fields = ("precision", "recall", "f1")
    variant_diffs: Dict[str, Any] = {}
    for v in variants:
        b_all, b_by = extract_variant(baseline.variants, v)
        c_all, c_by = extract_variant(candidate.variants, v)
        variant_diffs[v] = {
            "global_all": {
                f: {"baseline": b_all.get(f), "candidate": c_all.get(f)}
                for f in qa_fields
            },
            "global_by_type": {
                et: {
                    f: {
                        "baseline": b_by.get(et, {}).get(f),
                        "candidate": c_by.get(et, {}).get(f),
                    }
                    for f in qa_fields
                }
                for et in sorted(set(b_by.keys()) | set(c_by.keys()))
            },
        }
    out["quality_variants"] = variant_diffs

    # Choose a primary variant for legacy-style sections expected by write_html
    def choose_primary_variant(vmap: Dict[str, Any]) -> Optional[str]:
        if not vmap:
            return None
        if "exact" in vmap:
            return "exact"
        for key in ("iou@0.50", "iou@0.5", "iou@0.80", "iou@0.8"):
            if key in vmap:
                return key
        return sorted(vmap.keys())[0]

    primary_baseline = choose_primary_variant(baseline.variants)
    primary_candidate = choose_primary_variant(candidate.variants)
    # Prefer a common name if possible
    primary = None
    for k in ("exact", "iou@0.50", "iou@0.5", "iou@0.80", "iou@0.8"):
        if k in baseline.variants or k in candidate.variants:
            primary = k
            break
    if primary is None:
        primary = primary_baseline or primary_candidate

    # Build legacy-style sections from the chosen primary variant
    quality_global_all: Dict[str, Dict[str, Optional[float]]] = {
        f: {"baseline": None, "candidate": None} for f in qa_fields
    }
    quality_global_by_type: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    quality_per_component_all_types: Dict[str, Dict[str, Dict[str, float]]] = {}

    if primary:
        b_pack = baseline.variants.get(primary, {})
        c_pack = candidate.variants.get(primary, {})

        # Global all types
        b_global_all = b_pack.get("global_all_types", {}) or {}
        c_global_all = c_pack.get("global_all_types", {}) or {}
        for f in qa_fields:
            quality_global_all[f] = {
                "baseline": (
                    float(b_global_all.get(f))
                    if isinstance(b_global_all.get(f), (int, float))
                    else None
                ),
                "candidate": (
                    float(c_global_all.get(f))
                    if isinstance(c_global_all.get(f), (int, float))
                    else None
                ),
            }

        # Global by type
        b_by = b_pack.get("global_by_type", {}) or {}
        c_by = c_pack.get("global_by_type", {}) or {}
        all_types = sorted(set(b_by.keys()) | set(c_by.keys()))
        for et in all_types:
            b_vals = b_by.get(et, {})
            c_vals = c_by.get(et, {})
            quality_global_by_type[et] = {
                f: {
                    "baseline": (
                        float(b_vals.get(f))
                        if isinstance(b_vals.get(f), (int, float))
                        else None
                    ),
                    "candidate": (
                        float(c_vals.get(f))
                        if isinstance(c_vals.get(f), (int, float))
                        else None
                    ),
                }
                for f in qa_fields
            }

        # Per-component all types (include f1 and raw counts)
        b_comp = b_pack.get("per_component_all_types", {}) or {}
        c_comp = c_pack.get("per_component_all_types", {}) or {}
        comp_keys = sorted(set(b_comp.keys()) | set(c_comp.keys()))
        for comp in comp_keys:
            b_vals = b_comp.get(comp, {})
            c_vals = c_comp.get(comp, {})

            def num(d: Dict[str, Any], k: str) -> float:
                v = d.get(k)
                try:
                    return float(v)
                except Exception:
                    return 0.0

            quality_per_component_all_types[comp] = {
                "f1": {"baseline": num(b_vals, "f1"), "candidate": num(c_vals, "f1")},
                "tp": {"baseline": num(b_vals, "tp"), "candidate": num(c_vals, "tp")},
                "fp": {"baseline": num(b_vals, "fp"), "candidate": num(c_vals, "fp")},
                "fn": {"baseline": num(b_vals, "fn"), "candidate": num(c_vals, "fn")},
            }

    out["primary_variant"] = primary
    out["quality_global_all"] = quality_global_all
    out["quality_global_by_type"] = quality_global_by_type
    out["quality_per_component_all_types"] = quality_per_component_all_types
    return out


def _print_header(title: str) -> None:
    print("\n" + title)
    print("=" * len(title))


def _print_row(label: str, value: str, width: int = 26) -> None:
    print(f"{label.ljust(width)} {value}")


def render_console_report(diff: Dict[str, Any]) -> None:
    bl = diff["baseline"]
    cand = diff["candidate"]
    print(f"Baseline:  {bl}")
    print(f"Candidate: {cand}")

    _print_header("Performance")
    perf = diff["performance"]
    _print_row(
        "Docs processed",
        _fmt_delta(perf["docs_total"]["candidate"], perf["docs_total"]["baseline"]),
    )
    _print_row(
        "Overall duration (s)",
        _fmt_delta(
            perf["overall_seconds"]["candidate"],
            perf["overall_seconds"]["baseline"],
            suffix="s",
        ),
    )
    _print_row(
        "Mean docs/sec",
        _fmt_delta(
            perf["mean_docs_per_second"]["candidate"],
            perf["mean_docs_per_second"]["baseline"],
        ),
    )

    _print_header("Component durations (seconds)")
    for key, packs in sorted(diff["component_durations_seconds"].items()):
        p50 = packs["p50"]
        p95 = packs["p95"]
        p50s = _fmt_delta(p50.get("candidate"), p50.get("baseline"), suffix="s")
        p95s = _fmt_delta(p95.get("candidate"), p95.get("baseline"), suffix="s")
        print(f"- {key}")
        _print_row("  p50", p50s)
        _print_row("  p95", p95s)

    _print_header("Quality (variants)")
    for v, packs in diff.get("quality_variants", {}).items():
        print(f"- Variant: {v}")
        qa = packs["global_all"]
        for field in ("precision", "recall", "f1"):
            _print_row(
                "  " + field.capitalize(),
                _fmt_delta(qa[field]["candidate"], qa[field]["baseline"]),
            )

    # By type: show top 10 improvements and regressions by F1
    _print_header("Quality by type (top changes by F1 across variants)")
    changes: List[Tuple[str, str, float, float, float]] = (
        []
    )  # (variant, etype, base_f1, cand_f1, delta)
    for v, packs in diff.get("quality_variants", {}).items():
        for etype, vals in packs["global_by_type"].items():
            b = vals["f1"]["baseline"]
            c = vals["f1"]["candidate"]
            if b is None or c is None:
                continue
            changes.append((v, etype, float(b), float(c), float(c) - float(b)))
    changes.sort(key=lambda x: x[3], reverse=True)

    def _print_change_block(
        title: str, rows: List[Tuple[str, str, float, float, float]]
    ):
        print(title)
        print(
            "  "
            + "variant".ljust(12)
            + "etype".ljust(16)
            + "baseline".rjust(10)
            + "  ->  "
            + "candidate".ljust(10)
            + "  (delta)"
        )
        for v, etype, base_f1, cand_f1, delta in rows:
            print(
                "  "
                + v.ljust(12)
                + etype.ljust(16)
                + f"{base_f1:>10.3f}  ->  {cand_f1:<10.3f}  ({delta:+.3f})"
            )

    top_k = 10
    _print_change_block("  Improvements:", changes[:top_k])
    _print_change_block("  Regressions:", list(reversed(changes[-top_k:])))

    _print_header("Quality per component (all types) by variant")
    for v, packs in diff.get("quality_variants", {}).items():
        print(f"- Variant: {v}")
        # This summary focuses on global metrics; extend here if per-component variant summaries are needed.


def write_json(diff: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(diff, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_html(diff: Dict[str, Any], out_path: Path) -> None:
    # Minimal dependency-free HTML renderer
    def esc(s: Any) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def table(rows: List[List[Any]], header: Optional[List[str]] = None) -> str:
        th = ""
        if header:
            th = "<tr>" + "".join(f"<th>{esc(h)}</th>" for h in header) + "</tr>"
        # Cells can be:
        # - plain strings (rendered escaped)
        # - tuples (class_name, inner_html_string)
        tds_rows: List[str] = []
        for r in rows:
            cells: List[str] = []
            for c in r:
                if (
                    isinstance(c, (tuple, list))
                    and len(c) == 2
                    and isinstance(c[0], str)
                ):
                    cls, inner = c  # type: ignore[misc]
                    cls_attr = f" class='cell-{cls}'" if cls else ""
                    cells.append(f"<td{cls_attr}>{inner}</td>")
                else:
                    cells.append(f"<td>{esc(c)}</td>")
            tds_rows.append("<tr>" + "".join(cells) + "</tr>")
        trs = "".join(tds_rows)
        return f"<table>{th}{trs}</table>"

    styles = """
    <style>
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }
    h2 { margin-top: 28px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 14px; }
    th { background: #f7f7f7; text-align: left; }
    /* Colorblind-friendly cell backgrounds with icons */
    td.cell-pos { background: #e6f4ea; color: #0b4121; font-weight: 600; }
    td.cell-neg { background: #fde8e8; color: #5f0a0a; font-weight: 600; }
    td.cell-neutral { background: #f4f5f7; color: #333; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .arrow { display: inline-block; width: 1.2em; text-align: center; font-weight: 700; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    </style>
    """

    bl = diff["baseline"]
    cand = diff["candidate"]
    html: List[str] = ["<html><head><meta charset='utf-8'>", styles, "</head><body>"]
    html.append(
        f"<h1>Run comparison</h1><div class='mono'>Baseline: {esc(bl)}<br/>Candidate: {esc(cand)}</div>"
    )

    # Performance
    html.append("<h2>Performance</h2>")
    perf = diff["performance"]

    def delta_cell(
        curr: Optional[float],
        base: Optional[float],
        *,
        suffix: str = "",
        good_if_increase: bool = True,
    ) -> tuple[str, str]:
        if curr is None or base is None:
            return ("neutral", "n/a")
        d = curr - base
        eps = 1e-12
        if abs(d) <= eps:
            return ("neutral", "→ " + esc(_fmt_delta(curr, base, suffix)))
        good = (d > 0 and good_if_increase) or (d < 0 and not good_if_increase)
        cls = "pos" if good else "neg"
        arrow = "↑" if d > 0 else "↓"
        return (
            cls,
            f"<span class='arrow'>{arrow}</span> "
            + esc(_fmt_delta(curr, base, suffix)),
        )

    rows = [
        [
            "Docs processed",
            delta_cell(perf["docs_total"]["candidate"], perf["docs_total"]["baseline"]),
        ],
        [
            "Overall duration (s)",
            delta_cell(
                perf["overall_seconds"]["candidate"],
                perf["overall_seconds"]["baseline"],
                suffix="s",
                good_if_increase=False,
            ),
        ],
        [
            "Mean docs/sec",
            delta_cell(
                perf["mean_docs_per_second"]["candidate"],
                perf["mean_docs_per_second"]["baseline"],
                good_if_increase=True,
            ),
        ],
    ]
    html.append(table(rows, ["Metric", "Change"]))

    # Component durations
    html.append("<h2>Component durations (seconds)</h2>")
    comp_rows: List[List[str]] = []
    for key, packs in sorted(diff["component_durations_seconds"].items()):
        p50 = packs["p50"]
        p95 = packs["p95"]
        comp_rows.append(
            [
                key,
                delta_cell(
                    p50.get("candidate"),
                    p50.get("baseline"),
                    suffix="s",
                    good_if_increase=False,
                ),
                delta_cell(
                    p95.get("candidate"),
                    p95.get("baseline"),
                    suffix="s",
                    good_if_increase=False,
                ),
            ]
        )
    html.append(table(comp_rows, ["Component", "p50 change", "p95 change"]))

    # Quality global
    html.append("<h2>Quality (global, all types)</h2>")
    qa_rows = []
    qa = diff["quality_global_all"]
    for field in ("precision", "recall", "f1"):
        qa_rows.append(
            [
                field.capitalize(),
                delta_cell(
                    qa[field]["candidate"], qa[field]["baseline"], good_if_increase=True
                ),
            ]
        )
    html.append(table(qa_rows, ["Metric", "Change"]))

    # Quality by type
    html.append("<h2>Quality by type (sorted by F1 change)</h2>")
    changes: List[Tuple[str, float, float, float]] = []
    for etype, vals in diff["quality_global_by_type"].items():
        b = vals["f1"]["baseline"]
        c = vals["f1"]["candidate"]
        if b is None or c is None:
            continue
        changes.append((etype, float(b), float(c), float(c) - float(b)))
    changes.sort(key=lambda x: x[3], reverse=True)

    def dcell(val: float) -> tuple[str, str]:
        eps = 1e-12
        if abs(val) <= eps:
            return ("neutral", "→ +0.000")
        cls = "pos" if val > 0 else "neg"
        arrow = "↑" if val > 0 else "↓"
        return (cls, f"<span class='arrow'>{arrow}</span> " + esc(f"{val:+.3f}"))

    ch_rows = [[et, f"{b:.3f}", f"{c:.3f}", dcell(d)] for et, b, c, d in changes]
    html.append(table(ch_rows, ["Type", "Baseline F1", "Candidate F1", "ΔF1"]))

    # Per component
    html.append("<h2>Quality per component (all types)</h2>")
    pc_rows: List[List[str]] = []
    for key, v in sorted(diff["quality_per_component_all_types"].items()):
        pc_rows.append(
            [
                key,
                delta_cell(
                    v["f1"]["candidate"], v["f1"]["baseline"], good_if_increase=True
                ),
                delta_cell(float(v["tp"]["candidate"]), float(v["tp"]["baseline"])),
                delta_cell(
                    float(v["fp"]["candidate"]),
                    float(v["fp"]["baseline"]),
                    good_if_increase=False,
                ),
                delta_cell(
                    float(v["fn"]["candidate"]),
                    float(v["fn"]["baseline"]),
                    good_if_increase=False,
                ),
            ]
        )
    html.append(table(pc_rows, ["Component", "ΔF1", "ΔTP", "ΔFP", "ΔFN"]))

    html.append("</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(html), encoding="utf-8")


# ---- Multi-run support ----


def _extract_run_day(run_dir: Path) -> str:
    """Return YYYYMMDD for a run directory by parsing metrics.json->run_id or directory name.

    Falls back to filesystem mtime day if parsing fails.
    """
    day: Optional[str] = None
    try:
        payload = load_metrics_json(run_dir)
        run_id_val = payload.get("run_id") or run_dir.name
        m = re.match(r"^run-(\d{8})T\d{6}-", str(run_id_val))
        if m:
            day = m.group(1)
    except Exception:
        day = None
    if day is None:
        try:
            ts = run_dir.stat().st_mtime
            day = datetime.fromtimestamp(ts).strftime("%Y%m%d")
        except Exception:
            day = datetime.now().strftime("%Y%m%d")
    return day


def _runs_grouped_by_day(runs: List[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for p in runs:
        d = _extract_run_day(p)
        groups.setdefault(d, []).append(p)
    # sort each group by mtime ascending
    for d, lst in groups.items():
        lst.sort(key=lambda p: p.stat().st_mtime)
    return groups


def compare_multi_runs(
    baseline: RunMetrics,
    candidates: List[RunMetrics],
    *,
    primary_variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a multi-run comparison payload with deltas per candidate.

    Structure:
    {
      baseline: str,
      candidates: [str],
      performance: { metric: { candidate_id: {baseline, candidate, delta} } },
      comp_durations_p50: { component: { candidate_id: {baseline, candidate, delta} } },
      comp_durations_p95: { ... },
      quality: {
        variant: {
          global_all: { field: { candidate_id: {baseline, candidate, delta} } },
          by_type: { etype: { candidate_id: delta_f1 } }
        }
      }
    }
    """
    out: Dict[str, Any] = {
        "baseline": baseline.run_id,
        "baseline_path": str(baseline.path),
        "candidates": [c.run_id for c in candidates],
        "candidate_paths": {c.run_id: str(c.path) for c in candidates},
    }

    def _pct_delta(curr: Optional[float], base: Optional[float]) -> Optional[float]:
        if curr is None or base is None:
            return None
        if abs(float(base)) <= 1e-12:
            return None
        return (float(curr) - float(base)) / float(base) * 100.0

    def perf_block(cand: RunMetrics) -> Dict[str, Dict[str, float]]:
        return {
            "docs_total": {
                "baseline": baseline.docs_total,
                "candidate": cand.docs_total,
                "delta": cand.docs_total - baseline.docs_total,
                "pct": _pct_delta(cand.docs_total, baseline.docs_total),
            },
            "overall_seconds": {
                "baseline": baseline.overall_seconds,
                "candidate": cand.overall_seconds,
                "delta": cand.overall_seconds - baseline.overall_seconds,
                "pct": _pct_delta(cand.overall_seconds, baseline.overall_seconds),
            },
            "mean_docs_per_second": {
                "baseline": baseline.mean_docs_per_second,
                "candidate": cand.mean_docs_per_second,
                "delta": cand.mean_docs_per_second - baseline.mean_docs_per_second,
                "pct": _pct_delta(
                    cand.mean_docs_per_second, baseline.mean_docs_per_second
                ),
            },
        }

    out["performance"] = {cand.run_id: perf_block(cand) for cand in candidates}

    # Component durations
    comp_keys = sorted(
        set().union(
            baseline.component_durations_raw.keys(),
            *[c.component_durations_raw.keys() for c in candidates],
        )
    )
    comp_p50: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    comp_p95: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for comp in comp_keys:
        comp_p50[comp] = {}
        comp_p95[comp] = {}
        b = baseline.component_durations_raw.get(comp)
        b_p50 = (b or {}).get("p50")
        b_p95 = (b or {}).get("p95")
        for cand in candidates:
            c = cand.component_durations_raw.get(comp)
            c_p50 = (c or {}).get("p50")
            c_p95 = (c or {}).get("p95")

            # helper for pct
            def pct(curr: Optional[float], base: Optional[float]) -> Optional[float]:
                if curr is None or base is None:
                    return None
                if abs(float(base)) <= 1e-12:
                    return None
                return (float(curr) - float(base)) / float(base) * 100.0

            comp_p50[comp][cand.run_id] = {
                "baseline": b_p50,
                "candidate": c_p50,
                "delta": (
                    None
                    if (c_p50 is None or b_p50 is None)
                    else (float(c_p50) - float(b_p50))
                ),
                "pct": pct(c_p50, b_p50),
            }
            comp_p95[comp][cand.run_id] = {
                "baseline": b_p95,
                "candidate": c_p95,
                "delta": (
                    None
                    if (c_p95 is None or b_p95 is None)
                    else (float(c_p95) - float(b_p95))
                ),
                "pct": pct(c_p95, b_p95),
            }
    out["comp_durations_p50"] = comp_p50
    out["comp_durations_p95"] = comp_p95

    # Quality variants
    def extract_variant(
        vmap: Dict[str, Any], v: str
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        pack = vmap.get(v, {})
        global_all = {
            k: float(val)
            for k, val in (pack.get("global_all_types", {}) or {}).items()
            if isinstance(val, (int, float))
        }
        by_type = {
            et: {
                k: float(val)
                for k, val in vals.items()
                if k in ("precision", "recall", "f1")
            }
            for et, vals in (pack.get("global_by_type", {}) or {}).items()
        }
        return global_all, by_type

    # choose variants to compute (primary + IOU variants)
    all_variant_names: List[str] = sorted(
        set(baseline.variants.keys())
        | set().union(*(c.variants.keys() for c in candidates))
    )
    if primary_variant is None:
        chosen: Optional[str] = None
        if "exact" in all_variant_names:
            chosen = "exact"
        else:
            for key in ("iou@0.50", "iou@0.5", "iou@0.75", "iou@0.80", "iou@0.8"):
                if key in all_variant_names:
                    chosen = key
                    break
        primary_variant = chosen or (
            all_variant_names[0] if all_variant_names else None
        )

    # Order: primary first (if present), then all IOU variants (sorted), then any others
    iou_variants = [v for v in all_variant_names if v.lower().startswith("iou@")]
    other_variants = [
        v for v in all_variant_names if v not in iou_variants and v != primary_variant
    ]
    variant_order: List[str] = []
    if primary_variant:
        variant_order.append(primary_variant)
    variant_order.extend(sorted(iou_variants))
    variant_order.extend(sorted(other_variants))

    quality: Dict[str, Any] = {}
    field_names = ("precision", "recall", "f1")
    for variant_name in variant_order:
        b_all, b_by = extract_variant(baseline.variants, variant_name)
        # global all types deltas per candidate
        global_all_block: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
            f: {} for f in field_names
        }
        for cand in candidates:
            c_all, _ = extract_variant(cand.variants, variant_name)
            for f in field_names:
                b_val = b_all.get(f)
                c_val = c_all.get(f)
                delta_val = (
                    None
                    if (b_val is None or c_val is None)
                    else (float(c_val) - float(b_val))
                )
                pct_val = None
                if (
                    b_val is not None
                    and abs(float(b_val)) > 1e-12
                    and c_val is not None
                ):
                    pct_val = (float(c_val) - float(b_val)) / float(b_val) * 100.0
                global_all_block[f][cand.run_id] = {
                    "baseline": b_val,
                    "candidate": c_val,
                    "delta": delta_val,
                    "pct": pct_val,
                }

        # by type: delta f1 per candidate
        by_type_block: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
        all_types = sorted(
            set(b_by.keys())
            | set().union(
                *[
                    extract_variant(c.variants, variant_name)[1].keys()
                    for c in candidates
                ]
            )
        )
        for et in all_types:
            b_f1 = (b_by.get(et, {}) or {}).get("f1")
            by_type_block[et] = {}
            for cand in candidates:
                c_by = extract_variant(cand.variants, variant_name)[1]
                c_f1 = (c_by.get(et, {}) or {}).get("f1")
                delta_val = (
                    None
                    if (b_f1 is None or c_f1 is None)
                    else float(c_f1) - float(b_f1)
                )
                pct_val = None
                if b_f1 is not None and abs(float(b_f1)) > 1e-12 and c_f1 is not None:
                    pct_val = (float(c_f1) - float(b_f1)) / float(b_f1) * 100.0
                by_type_block[et][cand.run_id] = {
                    "baseline": b_f1,
                    "candidate": c_f1,
                    "delta": delta_val,
                    "pct": pct_val,
                }

        quality[variant_name] = {
            "global_all": global_all_block,
            "by_type": by_type_block,
        }

    out["primary_variant"] = primary_variant
    out["quality_variant_order"] = variant_order
    out["quality_multi"] = quality
    return out


def write_multi_json(report: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_multi_html(report: Dict[str, Any], out_path: Path) -> None:
    def esc(s: Any) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    styles = """
    <style>
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }
    h2 { margin-top: 28px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 14px; }
    th { background: #f7f7f7; text-align: left; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .arrow { display: inline-block; width: 1.2em; text-align: center; font-weight: 700; }
    </style>
    """

    bl = report["baseline"]
    bl_path = report.get("baseline_path", "")
    cands: List[str] = report.get("candidates", [])
    cand_paths: Dict[str, str] = report.get("candidate_paths", {})

    def pos_neg_cell(
        delta: Optional[float], fmt: str = "+.3f", good_if_increase: bool = True
    ) -> str:
        if delta is None:
            return "n/a"
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        good = (delta > 0 and good_if_increase) or (delta < 0 and not good_if_increase)
        cls = (
            "color: #0b4121; background:#e6f4ea;"
            if good
            else (
                "color:#5f0a0a; background:#fde8e8;"
                if delta != 0
                else "background:#f4f5f7;"
            )
        )
        return f"<span class='arrow'>{arrow}</span> <span style='{cls}'>{esc(format(delta, fmt))}</span>"

    def heat_cell(delta: Optional[float], max_abs: float) -> str:
        if delta is None or max_abs <= 1e-12:
            return "n/a"
        x = max(-1.0, min(1.0, float(delta) / max_abs))
        # map -1..1 to red..grey..green
        # red hsl(0, 80%, L), green hsl(145, 50%, L). Blend via x
        if x >= 0:
            hue = 145
            sat = 50
            light = int(96 - 40 * x)
        else:
            hue = 0
            sat = 80
            light = int(96 - 40 * (-x))
        style = f"background: hsl({hue}, {sat}%, {light}%); padding:2px 4px; border-radius:2px; display:inline-block; min-width:64px; text-align:right;"
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        return f"<span class='arrow'>{arrow}</span> <span style='{style}'>{float(delta):+.3f}</span>"

    html: List[str] = ["<html><head><meta charset='utf-8'>", styles, "</head><body>"]
    html.append(
        "<h1>Multi-run comparison</h1>"
        + f"<div class='mono'>Baseline: {esc(bl)}<br/><small>{esc(bl_path)}</small></div>"
        + "<h2>Candidates</h2>"
        + "<table>"
        + "<tr><th>Run</th><th>Path</th></tr>"
        + "".join(
            [
                f"<tr><td>{esc(c)}</td><td class='mono'>{esc(cand_paths.get(c, ''))}</td></tr>"
                for c in cands
            ]
        )
        + "</table>"
    )

    # Performance summary
    perf = report.get("performance", {})
    # Quality (GLOBAL) section first, as requested
    prim = report.get("primary_variant")
    q = report.get("quality_multi", {})
    order: List[str] = report.get(
        "quality_variant_order", ([] if prim is None else [prim])
    )
    show_variants = [v for v in order if v in q]
    if show_variants:
        # Render the first variant prominently
        v0 = show_variants[0]
        html.append(f"<h2>Quality (global, all types) — {esc(v0)}</h2>")
        global_all = q[v0].get("global_all", {})
        # header
        html.append("<table>")
        html.append(
            "<tr><th>Metric</th>"
            + "".join([f"<th>{esc(c)}</th>" for c in cands])
            + "</tr>"
        )
        for metric in ("Precision", "Recall", "F1"):
            key = metric.lower()
            cells = [f"<td>{metric}</td>"]
            for cand in cands:
                pack = (global_all.get(key, {}) or {}).get(cand, {})
                base_v = pack.get("baseline")
                cand_v = pack.get("candidate")
                delta = pack.get("delta")
                pct = pack.get("pct")
                base_str = "n/a" if base_v is None else f"{float(base_v):.3f}"
                cand_str = "n/a" if cand_v is None else f"{float(cand_v):.3f}"
                pct_str = "" if pct is None else f" <small>{float(pct):+.2f}%</small>"
                delta_str = pos_neg_cell(delta, fmt="+.3f", good_if_increase=True)
                cells.append(
                    f"<td><div class='mono'>{cand_str} (base {base_str}){pct_str}</div><div>{delta_str}</div></td>"
                )
            html.append("<tr>" + "".join(cells) + "</tr>")
        html.append("</table>")

        # Quality by type heatmap with pct
        html.append(f"<h2>Quality ΔF1 by type (variant: {esc(v0)})</h2>")
        by_type = q[v0].get("by_type", {})
        # compute max abs
        max_abs = 0.0
        for et, cand_map in by_type.items():
            for cand in cands:
                dval = (cand_map.get(cand, {}) or {}).get("delta")
                if isinstance(dval, (int, float)):
                    max_abs = max(max_abs, abs(float(dval)))
        rows: List[str] = []
        rows.append(
            "<tr>"
            + "<th>Type</th>"
            + "".join([f"<th>{esc(c)}</th>" for c in cands])
            + "</tr>"
        )
        for et, cand_map in sorted(by_type.items()):
            cells = [f"<td>{esc(et)}</td>"]
            for cand in cands:
                pack = cand_map.get(cand, {}) or {}
                delta = pack.get("delta")
                base_v = pack.get("baseline")
                cand_v = pack.get("candidate")
                pct = pack.get("pct")
                if delta is None:
                    cells.append("<td>n/a</td>")
                else:
                    base_str = "n/a" if base_v is None else f"{float(base_v):.3f}"
                    cand_str = "n/a" if cand_v is None else f"{float(cand_v):.3f}"
                    pct_str = (
                        "" if pct is None else f" <small>{float(pct):+.2f}%</small>"
                    )
                    cells.append(
                        "<td>"
                        + heat_cell(float(delta), max_abs)
                        + f"<div class='mono'>{cand_str} (base {base_str}){pct_str}</div>"
                        + "</td>"
                    )
            rows.append("<tr>" + "".join(cells) + "</tr>")
        html.append("<table>" + "".join(rows) + "</table>")

        # Additional variants (IOU thresholds etc.)
        for v in show_variants[1:]:
            html.append(f"<h2>Quality (global, all types) — {esc(v)}</h2>")
            global_all_v = q[v].get("global_all", {})
            html.append("<table>")
            html.append(
                "<tr><th>Metric</th>"
                + "".join([f"<th>{esc(c)}</th>" for c in cands])
                + "</tr>"
            )
            for metric in ("Precision", "Recall", "F1"):
                key = metric.lower()
                cells = [f"<td>{metric}</td>"]
                for cand in cands:
                    pack = (global_all_v.get(key, {}) or {}).get(cand, {})
                    base_v = pack.get("baseline")
                    cand_v = pack.get("candidate")
                    delta = pack.get("delta")
                    pct = pack.get("pct")
                    base_str = "n/a" if base_v is None else f"{float(base_v):.3f}"
                    cand_str = "n/a" if cand_v is None else f"{float(cand_v):.3f}"
                    pct_str = (
                        "" if pct is None else f" <small>{float(pct):+.2f}%</small>"
                    )
                    delta_str = pos_neg_cell(delta, fmt="+.3f", good_if_increase=True)
                    cells.append(
                        f"<td><div class='mono'>{cand_str} (base {base_str}){pct_str}</div><div>{delta_str}</div></td>"
                    )
                html.append("<tr>" + "".join(cells) + "</tr>")
            html.append("</table>")
            html.append(f"<h2>Quality ΔF1 by type (variant: {esc(v)})</h2>")
            by_type_v = q[v].get("by_type", {})
            max_abs = 0.0
            for et, cand_map in by_type_v.items():
                for cand in cands:
                    dval = (cand_map.get(cand, {}) or {}).get("delta")
                    if isinstance(dval, (int, float)):
                        max_abs = max(max_abs, abs(float(dval)))
            rows = []
            rows.append(
                "<tr>"
                + "<th>Type</th>"
                + "".join([f"<th>{esc(c)}</th>" for c in cands])
                + "</tr>"
            )
            for et, cand_map in sorted(by_type_v.items()):
                cells = [f"<td>{esc(et)}</td>"]
                for cand in cands:
                    pack = cand_map.get(cand, {}) or {}
                    delta = pack.get("delta")
                    base_v = pack.get("baseline")
                    cand_v = pack.get("candidate")
                    pct = pack.get("pct")
                    if delta is None:
                        cells.append("<td>n/a</td>")
                    else:
                        base_str = "n/a" if base_v is None else f"{float(base_v):.3f}"
                        cand_str = "n/a" if cand_v is None else f"{float(cand_v):.3f}"
                        pct_str = (
                            "" if pct is None else f" <small>{float(pct):+.2f}%</small>"
                        )
                        cells.append(
                            "<td>"
                            + heat_cell(float(delta), max_abs)
                            + f"<div class='mono'>{cand_str} (base {base_str}){pct_str}</div>"
                            + "</td>"
                        )
                rows.append("<tr>" + "".join(cells) + "</tr>")
            html.append("<table>" + "".join(rows) + "</table>")

    html.append("<h2>Performance (Δ vs baseline)</h2>")
    html.append("<table>")
    # header
    html.append(
        "<tr><th>Metric</th>" + "".join([f"<th>{esc(c)}</th>" for c in cands]) + "</tr>"
    )
    for metric, good_inc in (
        ("Docs processed", True),
        ("Overall duration (s)", False),
        ("Mean docs/sec", True),
    ):
        key = {
            "Docs processed": "docs_total",
            "Overall duration (s)": "overall_seconds",
            "Mean docs/sec": "mean_docs_per_second",
        }[metric]
        cells = [f"<td>{esc(metric)}</td>"]
        for cand in cands:
            block = perf.get(cand, {}).get(key, {})
            delta = block.get("delta")
            pct = block.get("pct")
            base_v = block.get("baseline")
            cand_v = block.get("candidate")
            val_fmt = "{:.3f}" if key != "docs_total" else "{:.0f}"
            val_str = f"{val_fmt.format(cand_v if cand_v is not None else 0)} (base {val_fmt.format(base_v if base_v is not None else 0)})"
            pct_str = "" if pct is None else f" &nbsp;<small>{pct:+.2f}%</small>"
            cells.append(
                f"<td><div>{val_str}</div><div>{pos_neg_cell(delta, fmt='+.3f' if key != 'docs_total' else '+.0f', good_if_increase=good_inc)}{pct_str}</div></td>"
            )
        html.append("<tr>" + "".join(cells) + "</tr>")
    html.append("</table>")

    # Component durations heatmaps
    for label, key in (
        ("Component durations p50 (s)", "comp_durations_p50"),
        ("Component durations p95 (s)", "comp_durations_p95"),
    ):
        html.append(f"<h2>{esc(label)}</h2>")
        block = report.get(key, {})
        # compute max abs delta across all
        max_abs = 0.0
        for comp, cand_map in block.items():
            for cand in cands:
                dval = cand_map.get(cand, {}).get("delta")
                if isinstance(dval, (int, float)):
                    max_abs = max(max_abs, abs(float(dval)))
        rows: List[str] = []
        # header
        rows.append(
            "<tr>"
            + "<th>Component</th>"
            + "".join([f"<th>{esc(c)}</th>" for c in cands])
            + "</tr>"
        )
        for comp, cand_map in sorted(block.items()):
            cells = [f"<td>{esc(comp)}</td>"]
            for cand in cands:
                delta = cand_map.get(cand, {}).get("delta")
                base_v = cand_map.get(cand, {}).get("baseline")
                cand_v = cand_map.get(cand, {}).get("candidate")
                pct = cand_map.get(cand, {}).get("pct")
                if delta is None:
                    cells.append("<td>n/a</td>")
                else:
                    base_str = "n/a" if base_v is None else f"{float(base_v):.3f}"
                    cand_str = "n/a" if cand_v is None else f"{float(cand_v):.3f}"
                    pct_str = (
                        "" if pct is None else f" <small>{float(pct):+.2f}%</small>"
                    )
                    cells.append(
                        "<td>"
                        + heat_cell(float(delta), max_abs)
                        + f"<div class='mono'>{cand_str} (base {base_str}){pct_str}</div>"
                        + "</td>"
                    )
            rows.append("<tr>" + "".join(cells) + "</tr>")
        html.append("<table>" + "".join(rows) + "</table>")

    # (The enhanced Quality by type heatmap is rendered above with pct and absolutes)

    html.append("</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(html), encoding="utf-8")


def find_run_dir(path_str: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    if p.is_dir() and (p / "metrics.json").exists():
        return p
    # Allow passing the metrics.json path directly
    if p.is_file() and p.name == "metrics.json":
        return p.parent
    raise FileNotFoundError(f"Not a run directory (no metrics.json): {p}")


def list_run_dirs(base_dir: Path) -> List[Path]:
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    runs: List[Path] = []
    for child in base_dir.iterdir():
        if child.is_dir() and (child / "metrics.json").exists():
            runs.append(child)
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Veil run metrics between runs."
    )
    parser.add_argument(
        "runs",
        nargs="*",
        help=(
            "Zero, one, two, or many run dirs (or metrics.json paths).\n"
            "- 0 args: if --same-day (default), compares baseline (oldest of latest-day) vs all other runs from that day; else falls back to latest two.\n"
            "- 1 arg: uses it as candidate; baseline is previous latest alongside.\n"
            "- 2 args: baseline vs candidate (legacy).\n"
            "- 3+ args: first is baseline, rest are candidates."
        ),
    )
    parser.add_argument(
        "--json-out",
        dest="json_out",
        default=None,
        help="Write diff JSON to this path (default: veil_runs/compare-<baseline>-vs-<candidate>.json)",
    )
    parser.add_argument(
        "--html-out",
        dest="html_out",
        default=None,
        help="Write an HTML report (default: veil_runs/compare-<baseline>-vs-<candidate>.html)",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default=None,
        help="Directory to place outputs when not explicitly set (default: veil_runs)",
    )
    parser.add_argument(
        "--open",
        dest="open_html",
        action="store_true",
        help="Open the generated HTML report in the default browser",
    )
    parser.add_argument(
        "--baseline",
        dest="baseline_path",
        default=None,
        help="Baseline run dir (or metrics.json) to compare against the latest run",
    )
    parser.add_argument(
        "--same-day",
        dest="same_day",
        action="store_true",
        help="When no runs are provided, compare all runs from the latest day (default)",
    )
    parser.add_argument(
        "--no-same-day",
        dest="same_day",
        action="store_false",
        help="Disable same-day default selection; fallback to latest two runs",
    )
    parser.set_defaults(same_day=True)
    args = parser.parse_args(argv)

    # Resolve runs for single or multi mode
    base_dir = Path.cwd() / "veil_runs"
    resolved_run_dirs: List[Path] = []
    if len(args.runs) == 0:
        runs = list_run_dirs(base_dir)
        if not runs:
            raise SystemExit("No runs found in veil_runs")
        if args.same_day:
            groups = _runs_grouped_by_day(runs)
            latest_day = sorted(groups.keys())[-1]
            same_day_runs = groups[latest_day]
            if len(same_day_runs) < 2:
                raise SystemExit("Not enough runs on the latest day to compare")
            resolved_run_dirs = same_day_runs
        else:
            if args.baseline_path:
                baseline_dir = find_run_dir(args.baseline_path)
                # choose latest run different from baseline as candidate
                others = [p for p in runs if p != baseline_dir]
                if not others:
                    raise SystemExit(
                        "Could not find a candidate run different from the baseline."
                    )
                resolved_run_dirs = [baseline_dir, others[0]]
            else:
                if len(runs) < 2:
                    raise SystemExit(
                        "Could not find at least two runs in veil_runs; please specify runs explicitly."
                    )
                resolved_run_dirs = [runs[1], runs[0]]
    else:
        # explicit args
        if len(args.runs) == 1:
            cand = find_run_dir(args.runs[0])
            if args.baseline_path:
                baseline_dir = find_run_dir(args.baseline_path)
                resolved_run_dirs = [baseline_dir, cand]
            else:
                runs = [p for p in list_run_dirs(cand.parent) if p != cand]
                if not runs:
                    raise SystemExit(
                        "Could not find a baseline run alongside the specified candidate run."
                    )
                resolved_run_dirs = [runs[0], cand]
        else:
            # 2+ args; if 2, legacy pairwise; if 3+, first is baseline, rest candidates
            resolved_run_dirs = [find_run_dir(p) for p in args.runs]

    # Decide mode
    if len(resolved_run_dirs) == 2:
        baseline = parse_run_metrics(resolved_run_dirs[0])
        candidate = parse_run_metrics(resolved_run_dirs[1])
        diff = compare_two_runs(baseline, candidate)
        # Console
        render_console_report(diff)
        # Outputs
        default_out_dir = (
            Path(args.out_dir)
            if args.out_dir
            else (
                candidate.path.parent
                if candidate.path.parent.name
                else Path.cwd() / "veil_runs"
            )
        )
        default_out_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"compare-{baseline.run_id}-vs-{candidate.run_id}"
        json_out = (
            Path(args.json_out)
            if args.json_out
            else default_out_dir / f"{base_name}.json"
        )
        html_out = (
            Path(args.html_out)
            if args.html_out
            else default_out_dir / f"{base_name}.html"
        )
        write_json(diff, json_out)
        write_html(diff, html_out)
        if args.open_html:
            try:
                webbrowser.open(html_out.resolve().as_uri())
            except Exception:
                pass
    else:
        # Multi-run: first is baseline, rest are candidates
        baseline = parse_run_metrics(resolved_run_dirs[0])
        candidates = [parse_run_metrics(p) for p in resolved_run_dirs[1:]]
        report = compare_multi_runs(baseline, candidates)
        # Outputs
        default_out_dir = (
            Path(args.out_dir)
            if args.out_dir
            else (
                baseline.path.parent
                if baseline.path.parent.name
                else Path.cwd() / "veil_runs"
            )
        )
        default_out_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"compare-multi-{baseline.run_id}"
        json_out = (
            Path(args.json_out)
            if args.json_out
            else default_out_dir / f"{base_name}.json"
        )
        html_out = (
            Path(args.html_out)
            if args.html_out
            else default_out_dir / f"{base_name}.html"
        )
        write_multi_json(report, json_out)
        write_multi_html(report, html_out)
        if args.open_html:
            try:
                webbrowser.open(html_out.resolve().as_uri())
            except Exception:
                pass

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
