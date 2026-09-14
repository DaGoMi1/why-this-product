"""오프라인 평가 지표"""

from __future__ import annotations

import math


def recall_at_k(recommended: list[str], ground_truth: set[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    top = recommended[:k]
    hits = sum(1 for i in top if i in ground_truth)
    return hits / len(ground_truth)


def mean_recall_at_k(
    user_recs: dict[str, list[str]],
    user_truth: dict[str, set[str]],
    k: int,
) -> float:
    scores: list[float] = []
    for user_id, truth in user_truth.items():
        if not truth:
            continue
        recs = user_recs.get(user_id, [])
        scores.append(recall_at_k(recs, truth, k))
    return float(sum(scores) / len(scores)) if scores else 0.0


def _dcg(rels: list[float]) -> float:
    return sum((2.0 ** rel - 1.0) / math.log2(i + 1) for i, rel in enumerate(rels, start=1))


def ndcg_at_k(recommended: list[str], item_rel: dict[str, float], k: int) -> float:
    if not item_rel:
        return 0.0
    rec_rels = [float(item_rel.get(item_id, 0.0)) for item_id in recommended[:k]]
    ideal = sorted((float(r) for r in item_rel.values()), reverse=True)[:k]
    denom = _dcg(ideal)
    if denom <= 0:
        return 0.0
    return _dcg(rec_rels) / denom


def mean_ndcg_at_k(
    user_recs: dict[str, list[str]],
    user_rel: dict[str, dict[str, float]],
    k: int,
) -> float:
    scores: list[float] = []
    for user_id, item_rel in user_rel.items():
        if not item_rel:
            continue
        recs = user_recs.get(user_id, [])
        scores.append(ndcg_at_k(recs, item_rel, k))
    return float(sum(scores) / len(scores)) if scores else 0.0
