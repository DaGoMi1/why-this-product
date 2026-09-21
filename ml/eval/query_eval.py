"""FAISS query list vs lexical gold: recall, miss, false positive."""

from __future__ import annotations

from ml.eval.metrics import ndcg_at_k, recall_at_k
from ml.eval.query_label import QUERY_SPECS, is_positive


def spec_for(query: str):
    for spec in QUERY_SPECS:
        if spec.query == query:
            return spec
    raise KeyError(query)


def evaluate(
    gold_rows: list[dict],
    recs_by_query: dict[str, list[str]],
    blobs: dict[str, str],
    k: int,
) -> tuple[dict, list[dict], list[dict]]:
    recalls: list[float] = []
    precisions: list[float] = []
    ndcgs: list[float] = []
    misses: list[dict] = []
    fps: list[dict] = []
    per_query: list[dict] = []
    for row in gold_rows:
        query = str(row["query"])
        gold_ids = [str(i) for i in row["item_ids"]]
        gold_set = set(gold_ids)
        recs = recs_by_query.get(query, [])[:k]
        rec_set = set(recs)
        spec = spec_for(query)
        r_at = recall_at_k(recs, gold_set, k) if gold_set else 0.0
        p_at = (sum(1 for i in recs if i in gold_set) / len(recs)) if recs else 0.0
        rel = {iid: 1.0 for iid in gold_set}
        n_at = ndcg_at_k(recs, rel, k)
        recalls.append(r_at)
        precisions.append(p_at)
        ndcgs.append(n_at)
        miss_ids = [iid for iid in gold_ids if iid not in rec_set]
        for iid in miss_ids:
            misses.append({"query": query, "intent": row["intent"], "item_id": iid, "kind": "miss"})
        fp_ids = []
        for iid in recs:
            blob = blobs.get(iid, "")
            if iid in gold_set:
                continue
            if is_positive(blob, spec):
                continue
            fp_ids.append(iid)
            fps.append({"query": query, "intent": row["intent"], "item_id": iid, "kind": "fp"})
        per_query.append(
            {
                "query": query,
                "intent": row["intent"],
                "n_gold": len(gold_set),
                "recall_at_k": round(r_at, 4),
                "precision_at_k": round(p_at, 4),
                "ndcg_at_k": round(n_at, 4),
                "n_miss": len(miss_ids),
                "n_fp": len(fp_ids),
            }
        )
    nq = len(gold_rows) or 1
    summary = {
        "k": k,
        "n_queries": len(gold_rows),
        "mean_recall_at_k": round(sum(recalls) / nq, 4),
        "mean_precision_at_k": round(sum(precisions) / nq, 4),
        "mean_ndcg_at_k": round(sum(ndcgs) / nq, 4),
        "n_miss": len(misses),
        "n_fp": len(fps),
        "strategy": "content+mmr",
        "per_query": per_query,
    }
    return summary, misses, fps
