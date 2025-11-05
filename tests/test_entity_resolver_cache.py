from __future__ import annotations

import pytest

from veil.config.entity_resolvers import EmbeddingsEntityResolverConfig
from veil.core.document import Document
from veil.core.span import Span
from veil.entity_resolvers.embeddings_resolver import EmbeddingsEntityResolver
from veil.entity_detectors.gliner.gliner_entity_type import GlinerEntityType


@pytest.mark.unit
def test_cache_exact_match_assigns_cached_id():
    doc = Document(text="Louise lives here.", doc_id="d1")
    spans = [
        Span(start=0, end=6, entity_type=GlinerEntityType.NAME),  # "Louise"
    ]

    # Cache contains NAME -> id 1 with aliases "Louise", "Louise Smith", "Louise"
    entity_cache = {"NAME": {1: {"Louise", "Louise Smith", "Louise"}}}

    resolver = EmbeddingsEntityResolver(EmbeddingsEntityResolverConfig(threshold=0.8))
    resolved = resolver.resolve(doc, spans, entity_cache=entity_cache)

    assert len(resolved) == 1
    assert resolved[0].id == 1


@pytest.mark.unit
def test_non_collision_new_ids_start_after_cache_max_per_type():
    text = "Pedro and Ana."
    doc = Document(text=text, doc_id="d2")
    # "Pedro" at 0..5, "Ana" at 8..11 (approx; include space counts accordingly)
    spans = [
        Span(start=0, end=5, entity_type=GlinerEntityType.NAME),
        Span(start=8, end=11, entity_type=GlinerEntityType.NAME),
    ]

    # Cache has id=3 for NAME, but mentions don't match cache; local ids should start at 4
    entity_cache = {"NAME": {3: {"Louise"}}}

    resolver = EmbeddingsEntityResolver(EmbeddingsEntityResolverConfig(threshold=0.8))
    resolved = resolver.resolve(doc, spans, entity_cache=entity_cache)

    assert len(resolved) == 2
    ids = [int(s.id) for s in resolved]
    assert ids == [4, 5]


@pytest.mark.unit
def test_cache_overrides_cluster_id_for_matching_component():
    text = "Tom and Tom"
    doc = Document(text=text, doc_id="d3")
    # Two mentions of the same name forming one component
    spans = [
        Span(start=0, end=3, entity_type=GlinerEntityType.NAME),
        Span(start=8, end=11, entity_type=GlinerEntityType.NAME),
    ]
    
    # Cache maps NAME -> id 2 for alias "Tom"
    entity_cache = {"NAME": {2: {"Tom"}}}

    resolver = EmbeddingsEntityResolver(EmbeddingsEntityResolverConfig(threshold=0.8))
    resolved = resolver.resolve(doc, spans, entity_cache=entity_cache)

    assert len(resolved) == 2
    assert resolved[0].id == 2 and resolved[1].id == 2


