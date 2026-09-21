"""Snippet rank: lexical + review boost. No FAISS."""

from __future__ import annotations

from ml.rag.index import RagChunk
from ml.rag.retriever import chunk_rank_score, lexical_overlap, tokenize


def test_overlap_is_query_coverage() -> None:
    assert tokenize("Hydrating Serum") == {"hydrating", "serum"}
    assert lexical_overlap("hydrating serum", "A hydrating vitamin serum") == 1.0
    assert lexical_overlap("hydrating serum", "gentle cleanser") == 0.0


def test_review_with_query_tokens_beats_unrelated_meta() -> None:
    query = "niacinamide serum"
    review = RagChunk("1", "B1", "review", "This niacinamide serum faded spots.")
    meta = RagChunk("0", "B1", "meta", "Generic beauty product kit.")
    assert chunk_rank_score(query, review, 0.1) > chunk_rank_score(query, meta, 0.4)
