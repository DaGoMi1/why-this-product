"""오프라인 평가 지표"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


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


def coverage_at_k(user_recs: dict[str, list[str]], catalog_n: int, k: int) -> float:
    if catalog_n <= 0:
        return 0.0
    uniq: set[str] = set()
    for recs in user_recs.values():
        uniq.update(recs[:k])
    return float(len(uniq) / catalog_n)


def mean_unique_categories_at_k(
    user_recs: dict[str, list[str]],
    item_category: dict[str, str | None],
    k: int,
) -> float:
    scores: list[float] = []
    for recs in user_recs.values():
        cats = {item_category.get(item_id) for item_id in recs[:k] if item_category.get(item_id)}
        scores.append(float(len(cats)))
    return float(sum(scores) / len(scores)) if scores else 0.0


def ild_at_k(
    recommended: list[str],
    vector_of: Callable[[str], np.ndarray | None],
    k: int,
) -> float:
    vecs = [v for item_id in recommended[:k] if (v := vector_of(item_id)) is not None]
    n = len(vecs)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - float(vecs[i] @ vecs[j])
            pairs += 1
    return total / pairs if pairs else 0.0


def mean_ild_at_k(
    user_recs: dict[str, list[str]],
    vector_of: Callable[[str], np.ndarray | None],
    k: int,
) -> float:
    scores = [ild_at_k(recs, vector_of, k) for recs in user_recs.values()]
    return float(sum(scores) / len(scores)) if scores else 0.0
