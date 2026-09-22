"""Query eval miss/fp split. No FAISS."""

from __future__ import annotations

from ml.eval.query_eval import evaluate
from ml.eval.query_label import QUERY_SPECS, blob_for_item, filter_ids_by_query_spec


def test_evaluate_counts_miss_and_fp() -> None:
    spec = next(s for s in QUERY_SPECS if s.query == "mascara")
    gold = [
        {
            "query": spec.query,
            "intent": spec.intent,
            "item_ids": ["G1", "G2"],
        }
    ]
    recs = {spec.query: ["G1", "FP1"]}
    blobs = {
        "G1": blob_for_item("Best mascara", None, ""),
        "G2": blob_for_item("Volume mascara", None, ""),
        "FP1": blob_for_item("Lipstick", None, ""),
    }
    summary, misses, fps = evaluate(gold, recs, blobs, k=10)
    assert summary["n_miss"] == 1
    assert summary["n_fp"] == 1
    assert misses[0]["item_id"] == "G2"
    assert fps[0]["item_id"] == "FP1"
    assert summary["mean_recall_at_k"] == 0.5
    assert summary["mean_precision_at_k"] == 0.5
    assert "product" in summary["by_intent"]
    assert summary["by_intent"]["product"]["n_fp"] == 1
    assert summary["by_intent"]["product"]["n_queries"] == 1


def test_by_intent_buckets_two_intents() -> None:
    gold = [
        {
            "query": "mascara",
            "intent": "product",
            "item_ids": ["G1"],
        },
        {
            "query": "niacinamide serum",
            "intent": "ingredient",
            "item_ids": ["G2"],
        },
    ]
    recs = {
        "mascara": ["G1"],
        "niacinamide serum": ["FP2"],
    }
    blobs = {
        "G1": blob_for_item("mascara", None, ""),
        "G2": blob_for_item("niacinamide serum", None, ""),
        "FP2": blob_for_item("random cream", None, ""),
    }
    summary, _, fps = evaluate(gold, recs, blobs, k=10)
    assert summary["by_intent"]["product"]["n_fp"] == 0
    assert summary["by_intent"]["ingredient"]["n_fp"] == 1
    assert fps[0]["intent"] == "ingredient"


def test_filter_ids_drops_non_serum() -> None:
    blobs = {
        "ok": blob_for_item("Hydrating Facial Serum", None, "moisturizing"),
        "bad": blob_for_item("Rich Night Cream", None, "hydrating cream"),
    }
    kept = filter_ids_by_query_spec(["ok", "bad"], "hydrating serum", blobs)
    assert kept == ["ok"]


def test_filter_ids_unknown_query_passthrough() -> None:
    blobs = {"a": "x"}
    assert filter_ids_by_query_spec(["a", "b"], "free form query", blobs) == ["a", "b"]
