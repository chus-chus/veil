from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import Request, urlopen

import pytest


def _base_url() -> str:
    return os.environ.get("VEIL_API_BASE_URL", "http://localhost:8000").rstrip("/")


def _server_available() -> bool:
    try:
        with urlopen(_base_url() + "/openapi.json", timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


def _post_json(path: str, payload: dict, timeout: float = 5) -> Tuple[int, Dict]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        _base_url() + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


def _latest_metrics_json(veil_runs_dir: str = "veil_runs") -> Optional[Path]:
    base = Path(veil_runs_dir)
    if not base.exists():
        return None
    candidates = []
    for run_dir in base.glob("run-*"):
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            candidates.append(metrics_path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _read_metrics_snapshot() -> Tuple[Optional[int], Optional[float]]:
    p = _latest_metrics_json()
    if p is None:
        return None, None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return (
            int(data.get("veil_pipeline_documents_processed_total", 0)),
            float(data.get("veil_pipeline_overall_duration_seconds", 0.0)),
        )
    except Exception:
        return None, None


def _drive_load(rate_rps: int, duration_s: int) -> Tuple[int, float]:
    total_requests = rate_rps * duration_s
    payload = {"text": "Ann visited the HQ at 10 Main Street.", "doc_id": None}
    sent = 0
    ok = 0
    start = time.perf_counter()

    # Schedule requests roughly at the given rate using a thread pool
    with ThreadPoolExecutor(max_workers=max(4, rate_rps * 2)) as pool:
        futures = []
        for i in range(total_requests):
            # target send time relative to start
            target = start + (i / max(1, rate_rps))
            now = time.perf_counter()
            delay = target - now
            if delay > 0:
                time.sleep(delay)
            futures.append(pool.submit(_post_json, "/mask", payload))
            sent += 1

        for fut in as_completed(futures):
            try:
                status, _ = fut.result()
                if status == 200:
                    ok += 1
            except Exception:
                pass

    elapsed = max(1e-6, time.perf_counter() - start)
    rps = ok / elapsed
    return ok, rps


@pytest.mark.performance
def test_server_performance_throughput_prints_results():
    if not _server_available():
        pytest.skip("Veil API server is not running on localhost:8000")

    # Snapshot metrics before
    docs_before, _ = _read_metrics_snapshot()

    # Try a few rates and short durations to keep CI quick
    scenarios = [(1, 3), (5, 3), (10, 3)]  # (rps, seconds)
    results = []
    for rate, dur in scenarios:
        ok, rps = _drive_load(rate, dur)
        results.append((rate, dur, ok, rps))

    # Snapshot metrics after (best-effort; may be None if server metrics disabled)
    docs_after, _ = _read_metrics_snapshot()
    docs_delta = None
    if docs_before is not None and docs_after is not None:
        docs_delta = max(0, docs_after - docs_before)

    # Print a compact summary for human inspection
    print("\n--- Server throughput summary (client-measured) ---")
    for rate, dur, ok, rps in results:
        print(f"target={rate} rps, duration={dur}s -> ok={ok}, measured_rps={rps:.2f}")
    if docs_delta is not None:
        print(f"metrics.json documents_processed_total delta: {docs_delta}")

    # Minimal assertion: at least some requests succeed in all scenarios
    assert all(ok > 0 for (_, _, ok, _) in results)

