"""Parallel explain keeps item order; OpenAI not called."""

from __future__ import annotations

from ml.rag.index import RagChunk
from ml.rag import llm as llm_mod


def test_explain_items_parallel_preserves_order(monkeypatch) -> None:
    calls: list[str] = []

    def fake_reason(client, model, item_id, title, snippets, query):
        calls.append(item_id)
        return f"reason-{item_id}", 1, 2

    monkeypatch.setattr(llm_mod, "require_api_key", lambda: "test-key")
    monkeypatch.setattr(llm_mod, "_reason_for_one", fake_reason)
    monkeypatch.setattr(llm_mod, "OpenAI", lambda api_key: object())

    ids = ["B1", "B2", "B3"]
    snippets = {
        iid: [RagChunk("0", iid, "meta", f"text {iid}")] for iid in ids
    }
    out = llm_mod.explain_items(
        ids,
        {iid: iid for iid in ids},
        snippets,
        "hydrating serum",
        model="gpt-4o-mini",
        select_k=0,
        max_workers=3,
    )
    assert [it.item_id for it in out.items] == ids
    assert [it.reason for it in out.items] == [f"reason-{i}" for i in ids]
    assert out.prompt_tokens == 3
    assert out.completion_tokens == 6
    assert set(calls) == set(ids)
