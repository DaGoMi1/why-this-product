"""Query eval miss/fp split. No FAISS."""

from __future__ import annotations

from ml.eval.query_label import QUERY_SPECS, blob_for_item
from ml.eval.query_eval import evaluate


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
