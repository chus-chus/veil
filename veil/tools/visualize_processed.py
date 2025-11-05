from __future__ import annotations

import argparse
import html
import json
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DocRecord:
    doc_id: str
    original_text: str
    masked_text: str
    ground_truth: List[Dict[str, Any]]
    predicted: List[Dict[str, Any]]
    tp: int
    fp: int
    fn: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _calc_prf(
    tp: int, fp: int, fn: int
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    p = None
    r = None
    f1 = None
    denom_p = tp + fp
    denom_r = tp + fn
    if denom_p > 0:
        p = tp / float(denom_p)
    if denom_r > 0:
        r = tp / float(denom_r)
    if p is not None and r is not None and (p + r) > 0:
        f1 = 2.0 * p * r / (p + r)
    return p, r, f1


def find_latest_processed_file(base_dir: Optional[str]) -> Path:
    if base_dir:
        proc_dir = Path(base_dir).expanduser().resolve()
    else:
        proc_dir = Path.cwd() / "data" / "processed"
    if not proc_dir.exists() or not proc_dir.is_dir():
        raise FileNotFoundError(f"Processed directory not found: {proc_dir}")
    candidates: List[Path] = [
        p for p in proc_dir.iterdir() if p.is_file() and p.suffix == ".jsonl"
    ]
    if not candidates:
        raise FileNotFoundError(f"No .jsonl files found in {proc_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_records(jsonl_path: Path) -> List[DocRecord]:
    records: List[DocRecord] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            doc_id = str(obj.get("doc_id", ""))
            masked_text = str(obj.get("masked_text", ""))
            original_text = str(obj.get("original_text", ""))
            gt_list = obj.get("ground_truth_entities") or []
            pr_list = obj.get("predicted_entities") or []
            # Evaluation: read from variant-based structure if present
            eval_pack = obj.get("evaluation") or {}
            variants = eval_pack.get("variants") or {}
            chosen_pack = None
            if isinstance(variants, dict) and variants:
                # Preference order: exact, iou@0.50, iou@0.8/0.80, otherwise first by key
                if "exact" in variants:
                    chosen_pack = variants["exact"]
                else:
                    for key in ("iou@0.50", "iou@0.5", "iou@0.80", "iou@0.8"):
                        if key in variants:
                            chosen_pack = variants[key]
                            break
                    if chosen_pack is None:
                        first_key = sorted(variants.keys())[0]
                        chosen_pack = variants[first_key]
            else:
                # Legacy fallback
                chosen_pack = eval_pack

            global_all = (chosen_pack or {}).get("global_all_types") or {}
            tp = int(global_all.get("tp", 0))
            fp = int(global_all.get("fp", 0))
            fn = int(global_all.get("fn", 0))
            p, r, f1 = _calc_prf(tp, fp, fn)
            records.append(
                DocRecord(
                    doc_id=doc_id,
                    original_text=original_text,
                    masked_text=masked_text,
                    ground_truth=list(gt_list),
                    predicted=list(pr_list),
                    tp=int(tp),
                    fp=int(fp),
                    fn=int(fn),
                    precision=p,
                    recall=r,
                    f1=f1,
                )
            )
    return records


def pick_worst(
    records: List[DocRecord], *, top_n: int, sort_by: str, descending: bool
) -> List[DocRecord]:
    def key(rec: DocRecord):
        if sort_by == "f1":
            # None last when ascending (worst first)
            return (1 if rec.f1 is None else 0, rec.f1 if rec.f1 is not None else 1.0)
        if sort_by == "recall":
            return (
                1 if rec.recall is None else 0,
                rec.recall if rec.recall is not None else 1.0,
            )
        if sort_by == "precision":
            return (
                1 if rec.precision is None else 0,
                rec.precision if rec.precision is not None else 1.0,
            )
        if sort_by == "errors":
            return -(rec.fp + rec.fn)
        # default f1
        return (1 if rec.f1 is None else 0, rec.f1 if rec.f1 is not None else 1.0)

    if sort_by == "errors":
        # For errors we want more errors first; handle descending flag accordingly
        sorted_recs = sorted(records, key=lambda r: key(r), reverse=True)
        if descending:
            # Descending of errors means fewer errors last; no change needed since reverse=True already puts more first
            pass
    else:
        sorted_recs = sorted(records, key=lambda r: key(r), reverse=descending)

    return sorted_recs[: max(0, int(top_n))]


def _html_escape_json(obj: Any) -> str:
    return html.escape(json.dumps(obj, ensure_ascii=False))


def _server_render_with_highlights(
    text: str, spans: List[Dict[str, Any]], cls: str
) -> str:
    n = len(text)
    norm_spans: List[Tuple[int, int, str]] = []
    for s in spans or []:
        try:
            start = max(0, min(n, int(s.get("start", 0))))
            end = max(0, min(n, int(s.get("end", 0))))
        except Exception:
            continue
        if end <= start:
            continue
        et = s.get("entity_type")
        norm_spans.append((start, end, str(et) if et is not None else ""))
    norm_spans.sort(key=lambda t: (t[0], t[1]))
    out: List[str] = []
    cursor = 0
    for start, end, etype in norm_spans:
        if start > cursor:
            out.append(html.escape(text[cursor:start]))
        inner = html.escape(text[start:end])
        label = f" <small>({html.escape(etype)})</small>" if etype else ""
        out.append(f"<mark class='{cls}'>" + inner + label + "</mark>")
        cursor = end
    if cursor < n:
        out.append(html.escape(text[cursor:]))
    return "".join(out)


def build_html(records: List[DocRecord], *, title: str, source_name: str) -> str:
    # Prepare data payload for client-side rendering
    data: List[Dict[str, Any]] = []
    for r in records:
        data.append(
            {
                "doc_id": r.doc_id,
                "original_text": r.original_text,
                "masked_text": r.masked_text,
                "tp": r.tp,
                "fp": r.fp,
                "fn": r.fn,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "ground_truth": [
                    {
                        "start": int(s.get("start", 0)),
                        "end": int(s.get("end", 0)),
                        "entity_type": s.get("entity_type"),
                    }
                    for s in (r.ground_truth or [])
                ],
                "predicted": [
                    {
                        "start": int(s.get("start", 0)),
                        "end": int(s.get("end", 0)),
                        "entity_type": s.get("entity_type"),
                    }
                    for s in (r.predicted or [])
                ],
            }
        )

    # JSON payload to embed in a dedicated <script type="application/json"> tag
    payload_json = json.dumps(data, ensure_ascii=False)
    payload_json = payload_json.replace("</", "<\\/")
    title_esc = html.escape(title)
    source_esc = html.escape(source_name)

    styles = """
    <style>
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 16px; }
    h1 { margin: 0 0 8px 0; font-size: 20px; }
    .controls { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }
    .controls input[type='text'] { padding: 6px 8px; font-size: 14px; }
    .controls select, .controls input[type='number'] { padding: 4px 6px; font-size: 14px; }
    .summary { color: #555; margin-bottom: 12px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .doc { border: 1px solid #ddd; border-radius: 8px; margin: 10px 0; padding: 8px 10px; }
    .doc-header { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; }
    .doc-id { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 13px; color: #333; }
    .metrics { font-size: 13px; color: #333; display: flex; gap: 10px; flex-wrap: wrap; }
    .metric-badge { padding: 2px 6px; border-radius: 12px; background: #f4f5f7; }
    .text-panels { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }
    .panel { border: 1px solid #eee; border-radius: 6px; padding: 8px; }
    .panel h3 { margin: 0 0 6px 0; font-size: 14px; color: #444; }
    .content { white-space: pre-wrap; line-height: 1.4; font-size: 14px; }
    mark.gt { background: rgba(166, 227, 161, 0.55); padding: 0 2px; border-radius: 3px; }
    mark.pred { background: rgba(250, 179, 135, 0.55); padding: 0 2px; border-radius: 3px; }
    .legend { display: flex; gap: 8px; align-items: center; font-size: 12px; color: #666; margin-top: 6px; }
    .legend .swatch { width: 12px; height: 12px; display: inline-block; border-radius: 3px; margin-right: 4px; }
    .pager { display: flex; gap: 10px; align-items: center; margin: 10px 0; }
    .hidden { display: none; }
    @media (max-width: 880px) { .text-panels { grid-template-columns: 1fr; } }
    </style>
    """

    data_script = f"""
    <script id='data-json' type='application/json'>
    {payload_json}
    </script>
    """

    script = """
    <script>
    const DATA = JSON.parse(document.getElementById('data-json').textContent);

    function prfRow(d) {{
      const p = d.precision == null ? 'n/a' : d.precision.toFixed(3);
      const r = d.recall == null ? 'n/a' : d.recall.toFixed(3);
      const f = d.f1 == null ? 'n/a' : d.f1.toFixed(3);
      return `<span class='metric-badge'>TP ${d.tp}</span><span class='metric-badge'>FP ${d.fp}</span><span class='metric-badge'>FN ${d.fn}</span>` +
             `<span class='metric-badge'>P ${p}</span><span class='metric-badge'>R ${r}</span><span class='metric-badge'>F1 ${f}</span>`;
    }}

    function escapeHtml(s) {{
      const div = document.createElement('div');
      div.innerText = s;
      return div.innerHTML;
    }}

    function renderWithHighlights(text, spans, cls) {{
      if (!spans || spans.length === 0) return escapeHtml(text);
      const n = text.length;
      const sorted = spans
        .map(s => ({{start: Math.max(0, Math.min(n, parseInt(s.start)||0)), end: Math.max(0, Math.min(n, parseInt(s.end)||0)), type: s.entity_type || ''}}))
        .filter(s => s.end > s.start)
        .sort((a, b) => a.start - b.start || a.end - b.end);
      let out = [];
      let cursor = 0;
      for (const s of sorted) {{
        if (s.start > cursor) {{ out.push(escapeHtml(text.slice(cursor, s.start))); }}
        const inner = escapeHtml(text.slice(s.start, s.end));
        const label = s.type ? ` <small>(${s.type})</small>` : '';
        out.push(`<mark class='${cls}'>${inner}${label}</mark>`);
        cursor = s.end;
      }}
      if (cursor < n) out.push(escapeHtml(text.slice(cursor)));
      return out.join('');
    }}

    function renderDocs(docs, options) {{
      const container = document.getElementById('docs');
      container.innerHTML = '';
      const showGt = document.getElementById('toggle-gt').checked;
      const showPred = document.getElementById('toggle-pred').checked;

      docs.forEach(d => {{
        const el = document.createElement('div');
        el.className = 'doc';
        el.innerHTML = `
          <div class='doc-header'>
            <div class='doc-id'>${d.doc_id}</div>
            <div class='metrics'>${prfRow(d)}</div>
          </div>
          <div class='text-panels'>
            <div class='panel'>
              <h3>Original</h3>
              <div class='content'>${showGt ? renderWithHighlights(d.original_text, d.ground_truth, 'gt') : escapeHtml(d.original_text)}</div>
              <div class='legend'><span class='swatch' style='background: rgba(166, 227, 161, 0.55)'></span> Ground truth</div>
            </div>
            <div class='panel'>
              <h3>Masked</h3>
              <div class='content'>${showPred ? renderWithHighlights(d.original_text, d.predicted, 'pred') : escapeHtml(d.masked_text)}</div>
              <div class='legend'><span class='swatch' style='background: rgba(250, 179, 135, 0.55)'></span> Predicted (shown over original)</div>
            </div>
          </div>
        `;
        container.appendChild(el);
      }});
    }}

    function applyFilters() {{
      const sortBy = document.getElementById('sort-by').value;
      const order = document.getElementById('order').value;
      const q = document.getElementById('search').value.trim().toLowerCase();
      const pageSize = parseInt(document.getElementById('page-size').value) || 10;
      let filtered = DATA.slice();
      if (q) {{
        filtered = filtered.filter(d => (d.doc_id||'').toLowerCase().includes(q));
      }}

      const cmp = (a, b) => {{
        const as = (v) => v == null ? -1 : v; // nulls first
        if (sortBy === 'errors') {{
          const ae = (a.fp + a.fn);
          const be = (b.fp + b.fn);
          return ae - be;
        }} else if (sortBy === 'precision') {{
          return as(a.precision) - as(b.precision);
        }} else if (sortBy === 'recall') {{
          return as(a.recall) - as(b.recall);
        }}
        // default f1
        return as(a.f1) - as(b.f1);
      }};
      filtered.sort(cmp);
      if (order === 'desc') filtered.reverse();

      // Pagination
      const total = filtered.length;
      const pageCount = Math.max(1, Math.ceil(total / pageSize));
      const pageInput = document.getElementById('page');
      let page = parseInt(pageInput.value) || 1;
      if (page < 1) page = 1;
      if (page > pageCount) page = pageCount;
      pageInput.value = page;
      const start = (page - 1) * pageSize;
      const end = Math.min(total, start + pageSize);

      document.getElementById('summary').innerText = `${total} documents, showing ${start+1}-${end}`;
      renderDocs(filtered.slice(start, end), {{}});
    }}

    function init() {{
      // Populate doc select
      const docSelect = document.getElementById('doc-select');
      const ids = DATA.map(d => d.doc_id);
      const frag = document.createDocumentFragment();
      const emptyOpt = document.createElement('option');
      emptyOpt.value = '';
      emptyOpt.textContent = '(select a doc)';
      frag.appendChild(emptyOpt);
      ids.forEach(id => {{
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = id;
        frag.appendChild(opt);
      }});
      docSelect.appendChild(frag);
      docSelect.addEventListener('change', () => {{
        const v = docSelect.value;
        const search = document.getElementById('search');
        search.value = v || '';
        document.getElementById('page').value = '1';
        applyFilters();
      }});
      document.getElementById('sort-by').addEventListener('change', applyFilters);
      document.getElementById('order').addEventListener('change', applyFilters);
      document.getElementById('page-size').addEventListener('change', applyFilters);
      document.getElementById('page').addEventListener('change', applyFilters);
      document.getElementById('search').addEventListener('input', applyFilters);
      document.getElementById('toggle-gt').addEventListener('change', applyFilters);
      document.getElementById('toggle-pred').addEventListener('change', applyFilters);
      applyFilters();
    }}

    window.addEventListener('DOMContentLoaded', init);
    </script>
    """

    # Server-side pre-render (fallback in case JS is blocked/errors)
    prerender_blocks: List[str] = []
    for d in data:
        gt_html = _server_render_with_highlights(
            d["original_text"], d["ground_truth"], "gt"
        )
        pred_html = _server_render_with_highlights(
            d["original_text"], d["predicted"], "pred"
        )
        metrics = [
            f"<span class='metric-badge'>TP {d['tp']}</span>",
            f"<span class='metric-badge'>FP {d['fp']}</span>",
            f"<span class='metric-badge'>FN {d['fn']}</span>",
            f"<span class='metric-badge'>P {('n/a' if d['precision'] is None else f"{d['precision']:.3f}")}</span>",
            f"<span class='metric-badge'>R {('n/a' if d['recall'] is None else f"{d['recall']:.3f}")}</span>",
            f"<span class='metric-badge'>F1 {('n/a' if d['f1'] is None else f"{d['f1']:.3f}")}</span>",
        ]
        prerender_blocks.append(
            "".join(
                [
                    "<div class='doc'>",
                    f"<div class='doc-header'><div class='doc-id'>{html.escape(d['doc_id'])}</div>",
                    f"<div class='metrics'>{''.join(metrics)}</div></div>",
                    "<div class='text-panels'>",
                    "<div class='panel'><h3>Original</h3>",
                    f"<div class='content'>{gt_html}</div>",
                    "<div class='legend'><span class='swatch' style='background: rgba(166, 227, 161, 0.55)'></span> Ground truth</div></div>",
                    "<div class='panel'><h3>Masked</h3>",
                    f"<div class='content'>{pred_html}</div>",
                    "<div class='legend'><span class='swatch' style='background: rgba(250, 179, 135, 0.55)'></span> Predicted (shown over original)</div></div>",
                    "</div></div>",
                ]
            )
        )
    prerender_html = "".join(prerender_blocks)

    html_doc = f"""
    <html>
      <head>
        <meta charset='utf-8'/>
        <title>{title_esc}</title>
        {styles}
      </head>
      <body>
        <h1>{title_esc}</h1>
        <div class='summary'>Source: <span class='mono'>{source_esc}</span></div>
        <div class='controls'>
          <label>Sort by
            <select id='sort-by'>
              <option value='f1' selected>F1 (asc)</option>
              <option value='precision'>Precision (asc)</option>
              <option value='recall'>Recall (asc)</option>
              <option value='errors'>Errors FP+FN (asc)</option>
            </select>
          </label>
          <label>Order
            <select id='order'>
              <option value='asc' selected>Ascending</option>
              <option value='desc'>Descending</option>
            </select>
          </label>
          <label>Page size
            <select id='page-size'>
              <option>5</option>
              <option selected>10</option>
              <option>20</option>
              <option>50</option>
            </select>
          </label>
          <label>Page <input id='page' type='number' min='1' value='1' style='width: 72px;'/></label>
          <label>Search Doc ID <input id='search' type='text' placeholder='contains…'/></label>
          <label>Doc select
            <select id='doc-select' style='min-width: 240px;'>
            </select>
          </label>
          <label><input id='toggle-gt' type='checkbox' checked/> Show ground truth</label>
          <label><input id='toggle-pred' type='checkbox' checked/> Show predictions</label>
        </div>
        <div id='summary' class='summary'></div>
        <div id='docs'>{prerender_html}</div>
        {data_script}
        {script}
      </body>
    </html>
    """
    return html_doc


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize processed JSONL documents and their per-document evaluation."
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        default=None,
        help="Path to processed .jsonl. If omitted, uses the most recent in data/processed",
    )
    parser.add_argument(
        "--processed-dir",
        dest="processed_dir",
        default="data/output",
        help="Directory to search when --input is not given (default: data/output)",
    )
    parser.add_argument(
        "--top",
        dest="top_n",
        type=int,
        default=50,
        help="Number of worst documents to include in the HTML (default: 50)",
    )
    parser.add_argument(
        "--sort-by",
        dest="sort_by",
        default="f1",
        choices=["f1", "precision", "recall", "errors"],
        help="Sorting key for worst documents selection",
    )
    parser.add_argument(
        "--desc",
        dest="descending",
        action="store_true",
        help="Sort in descending order",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=None,
        help="Output HTML path. Default: alongside input as visualize-<name>.html",
    )
    parser.add_argument(
        "--open",
        dest="open_html",
        action="store_true",
        help="Open the generated HTML in the default browser",
    )
    args = parser.parse_args(argv)

    # Resolve input
    if args.input_path:
        input_path = Path(args.input_path).expanduser().resolve()
        if not input_path.exists():
            raise SystemExit(f"Input not found: {input_path}")
    else:
        input_path = find_latest_processed_file(args.processed_dir)

    records = load_records(input_path)
    if not records:
        raise SystemExit("No records found in input JSONL")

    # Choose worst docs
    worst = pick_worst(
        records, top_n=args.top_n, sort_by=args.sort_by, descending=args.descending
    )

    # Output path
    if args.output_path:
        out_path = Path(args.output_path).expanduser().resolve()
    else:
        base = input_path.stem
        out_path = input_path.parent / f"visualize-{base}.html"

    # Build HTML
    title = f"Worst {len(worst)} documents by {args.sort_by}"
    html_doc = build_html(worst, title=title, source_name=str(input_path.name))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {out_path}")

    if args.open_html:
        try:
            webbrowser.open(out_path.resolve().as_uri())
        except Exception:
            pass

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
