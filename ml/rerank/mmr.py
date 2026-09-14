"""MMR 다양성 리랭커."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ml.retrieval.content_faiss import ContentFaissRetriever


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class ItemSimilarity:
    """Cosine on content embeddings; same category = 1 if a vector is missing."""

    def __init__(
        self,
        content: ContentFaissRetriever | None,
        items_meta: dict[str, dict],
    ) -> None:
        self.content = content
        self.items_meta = items_meta
        self._cache: dict[str, np.ndarray | None] = {}

    def vec(self, item_id: str) -> np.ndarray | None:
        if item_id in self._cache:
            return self._cache[item_id]
        if self.content is None:
            self._cache[item_id] = None
            return None
        pos = self.content._id_to_pos.get(item_id)
        if pos is None:
            self._cache[item_id] = None
            return None
        raw = np.asarray(self.content.index.index.reconstruct(int(pos)), dtype=np.float32)
        norm = float(np.linalg.norm(raw)) + 1e-12
        vec = raw / norm
        self._cache[item_id] = vec
        return vec

    def __call__(self, left: str, right: str) -> float:
        va = self.vec(left)
        vb = self.vec(right)
        if va is not None and vb is not None:
            return float(np.clip(va @ vb, 0.0, 1.0))
        ca = (self.items_meta.get(left) or {}).get("category")
        cb = (self.items_meta.get(right) or {}).get("category")
        if ca and cb and ca == cb:
            return 1.0
        return 0.0


def mmr_rerank(
    candidates: list[tuple[str, float]],
    k: int,
    lambda_diversity: float,
    similarity: Callable[[str, str], float],
) -> list[tuple[str, float]]:
    """Greedy MMR. lambda_diversity is the penalty weight; λ_rel = 1 - lambda_diversity."""
    if k <= 0 or not candidates:
        return []
    lam_div = min(1.0, max(0.0, float(lambda_diversity)))
    lam_rel = 1.0 - lam_div
    ids = [item_id for item_id, _ in candidates]
    rels = _minmax([float(score) for _, score in candidates])
    remaining = list(range(len(ids)))
    chosen: list[int] = []
    while remaining and len(chosen) < k:
        best_i = remaining[0]
        best_score = float("-inf")
        for i in remaining:
            if not chosen:
                score = rels[i]
            else:
                max_sim = max(similarity(ids[i], ids[j]) for j in chosen)
                score = lam_rel * rels[i] - lam_div * max_sim
            if score > best_score:
                best_score = score
                best_i = i
        chosen.append(best_i)
        remaining.remove(best_i)
    return [(ids[i], float(candidates[i][1])) for i in chosen]
