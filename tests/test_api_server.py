from __future__ import annotations

import json
import os
from urllib.error import URLError, HTTPError
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


def _post_json(path: str, payload: dict, timeout: float = 5):
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


@pytest.mark.e2e
def test_mask_endpoint_e2e_shape():
    if not _server_available():
        pytest.skip("Veil API server is not running on localhost:8000")

    text = "Anna lives on 10th Street."
    status, data = _post_json("/mask", {"text": text, "doc_id": "test-e2e-1"})
    assert status == 200
    assert set(["doc_id", "masked_text", "entities"]) <= set(data.keys())
    assert data["doc_id"] == "test-e2e-1"
    assert isinstance(data["masked_text"], str)
    assert isinstance(data["entities"], list)
    # If entities exist, they must have expected keys
    if data["entities"]:
        assert {"start", "end", "entity_type", "id", "confidence"} <= set(
            data["entities"][0].keys()
        )


@pytest.mark.e2e
def test_mask_endpoint_e2e_handles_minimal_payload():
    if not _server_available():
        pytest.skip("Veil API server is not running on localhost:8000")

    text = "Simple text without entities"
    status, data = _post_json("/mask", {"text": text})
    assert status == 200
    assert isinstance(data.get("masked_text"), str)
    assert isinstance(data.get("entities"), list)


@pytest.mark.e2e
def test_entity_cache_conversation_ids_consistent():
    if not _server_available():
        pytest.skip("Veil API server is not running on localhost:8000")

    # Simulated conversation with repeated mentions of the same person
    texts = [
        "Enter Louise, greeting the Apple team.",
        "I, Louise, spoke with Marcel.",
        "And Anna also spoke with Marcel.",
        "Louise was here today. She greeted Anna from Roche.",
    ]

    # entity_cache format: { entity_type: { id: set(aliases) } }
    entity_cache: dict[str, dict[int, set[str]]] = {}

    def cache_to_lists(cache: dict[str, dict[int, set[str]]]) -> dict:
        return {k: {int(i): sorted(list(v)) for i, v in inner.items()} for k, inner in cache.items()}

    for idx, text in enumerate(texts):
        payload = {
            "doc_id": f"conv-{idx}",
            "text": text,
            "entity_cache": cache_to_lists(entity_cache),
        }
        status, data = _post_json("/mask", payload)
        assert status == 200
        assert isinstance(data.get("entities"), list)

        for ent in data["entities"]:
            ent_type = ent.get("entity_type")
            ent_id = ent.get("id")
            if ent_type is None or ent_id is None:
                continue
            # Only handle integer ids
            try:
                ent_id_int = int(ent_id)
            except Exception:
                continue
            start = int(ent.get("start", 0))
            end = int(ent.get("end", 0))
            mention = text[start:end]
            if not mention:
                continue

            # If mention exists in cache under this type, assert id matches expected cached id
            cached_for_type = entity_cache.get(ent_type, {})
            expected_id: int | None = None
            for cid, aliases in cached_for_type.items():
                if mention in aliases:
                    expected_id = cid
                    break
            if expected_id is not None:
                assert ent_id_int == expected_id

            # Update cache with this observation
            entity_cache.setdefault(ent_type, {}).setdefault(ent_id_int, set()).add(mention)
        print(f"Cache: {cache_to_lists(entity_cache)}")


