"""오프라인 랭킹 메트릭."""

from __future__ import annotations


# 추천 목록에서 실제 관심 아이템이 포함된 비율
# Recall@K
def recall_at_k(recommended: list[str], ground_truth: set[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    top = recommended[:k]
    hits = sum(1 for i in top if i in ground_truth)
    return hits / len(ground_truth)

# 모든 사용자의 Recall@K 평균
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
