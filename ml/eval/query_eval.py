"""FAISS query list vs lexical gold: recall, miss, false positive."""

from __future__ import annotations

from collections import defaultdict

from ml.eval.metrics import ndcg_at_k, recall_at_k
from ml.eval.query_label import QUERY_SPECS, is_positive


def spec_for(query: str):
    for spec in QUERY_SPECS:
        if spec.query == query:
            return spec
    raise KeyError(query)


def _roll_intent(
    buckets: dict[str, dict],
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for intent in sorted(buckets):
        b = buckets[intent]
        n = len(b["recalls"]) or 1
        out[intent] = {
            "n_queries": len(b["recalls"]),
            "mean_recall_at_k": round(sum(b["recalls"]) / n, 4),
            "mean_precision_at_k": round(sum(b["precisions"]) / n, 4),
            "mean_ndcg_at_k": round(sum(b["ndcgs"]) / n, 4),
            "n_miss": b["n_miss"],
            "n_fp": b["n_fp"],
        }
    return out


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
    buckets: dict[str, dict] = defaultdict(
        lambda: {
            "recalls": [],
            "precisions": [],
            "ndcgs": [],
            "n_miss": 0,
            "n_fp": 0,
        }
    )
    for row in gold_rows:
        query = str(row["query"])
        intent = str(row["intent"])
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
            misses.append({"query": query, "intent": intent, "item_id": iid, "kind": "miss"})
        fp_ids = []
        for iid in recs:
            blob = blobs.get(iid, "")
            if iid in gold_set:
                continue
            if is_positive(blob, spec):
                continue
            fp_ids.append(iid)
            fps.append({"query": query, "intent": intent, "item_id": iid, "kind": "fp"})
        buckets[intent]["recalls"].append(r_at)
        buckets[intent]["precisions"].append(p_at)
        buckets[intent]["ndcgs"].append(n_at)
        buckets[intent]["n_miss"] += len(miss_ids)
        buckets[intent]["n_fp"] += len(fp_ids)
        per_query.append(
            {
                "query": query,
                "intent": intent,
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
        "strategy": "content+mmr+lex",
        "by_intent": _roll_intent(buckets),
        "per_query": per_query,
    }
    return summary, misses, fps
