"""랭커 피처. 전부 train 통계만 사용."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ml.retrieval.content_faiss import ContentFaissRetriever
from ml.retrieval.ials import IALSRetriever
from ml.retrieval.popularity import PopularityRetriever

FEATURE_NAMES = [
    "log_pop_count",
    "item_n",
    "item_mean_rating",
    "user_n",
    "pop_rank",
    "content_max_sim",
    "ials_score",
    "same_brand",
    "same_category",
]


class FeatureBuilder:
    def __init__(
        self,
        popularity: PopularityRetriever,
        item_n: dict[str, int],
        item_mean_rating: dict[str, float],
        items_meta: dict[str, dict],
        content: ContentFaissRetriever | None,
        ials: IALSRetriever | None,
    ) -> None:
        self.popularity = popularity                    # 인기도 레파지토리
        self.item_n = item_n                            # 아이템 개수
        self.item_mean_rating = item_mean_rating        # 아이템 평균 평점
        self.items_meta = items_meta                    # 아이템 메타데이터
        self.content = content                          # 콘텐츠 레파지토리
        self.ials = ials                                # 아이템 인자 레파지토리
        self._pop_rank = {iid: i + 1 for i, iid in enumerate(popularity._item_ids)}
        self._vec_cache: dict[str, np.ndarray | None] = {}

    @classmethod
    def from_train(
        cls,
        train: pd.DataFrame,
        popularity: PopularityRetriever,
        items_meta: dict[str, dict],
        content: ContentFaissRetriever | None,
        ials: IALSRetriever | None,
    ) -> FeatureBuilder:
        stats = train.groupby("item_id").agg(
            item_n=("rating", "size"),
            item_mean_rating=("rating", "mean"),
        )
        item_n = {str(i): int(n) for i, n in stats["item_n"].items()}
        item_mean = {str(i): float(m) for i, m in stats["item_mean_rating"].items()}
        return cls(
            popularity=popularity,
            item_n=item_n,
            item_mean_rating=item_mean,
            items_meta=items_meta,
            content=content,
            ials=ials,
        )

    def _item_vec(self, item_id: str) -> np.ndarray | None:
        if item_id in self._vec_cache:
            return self._vec_cache[item_id]
        if self.content is None:
            self._vec_cache[item_id] = None
            return None
        pos = self.content._id_to_pos.get(item_id)
        if pos is None:
            self._vec_cache[item_id] = None
            return None
        vec = np.asarray(self.content.index.index.reconstruct(int(pos)), dtype=np.float32)
        norm = float(np.linalg.norm(vec)) + 1e-12
        vec = vec / norm
        self._vec_cache[item_id] = vec
        return vec

    def _content_max_sim(self, seeds: list[str], item_id: str) -> float:
        cand = self._item_vec(item_id)
        if cand is None:
            return 0.0
        best = 0.0
        for seed in seeds:
            sv = self._item_vec(seed)
            if sv is None:
                continue
            best = max(best, float(sv @ cand))
        return best

    def matrix(
        self,
        user_id: str,
        history: list[str],
        candidate_ids: list[str],
    ) -> np.ndarray:
        seeds = history[-5:]
        last = seeds[-1] if seeds else None
        last_meta = self.items_meta.get(last, {}) if last else {}
        last_brand = last_meta.get("brand")
        last_cat = last_meta.get("category")
        user_n = float(len(history))
        rows: list[list[float]] = []
        for item_id in candidate_ids:
            count = float(self.popularity._scores.get(item_id, 0.0))
            meta = self.items_meta.get(item_id, {})
            rows.append(
                [
                    math.log1p(count),
                    float(self.item_n.get(item_id, 0)),
                    float(self.item_mean_rating.get(item_id, 0.0)),
                    user_n,
                    float(self._pop_rank.get(item_id, len(self._pop_rank) + 1)),
                    self._content_max_sim(seeds, item_id),
                    self.ials.score_item(user_id, item_id) if self.ials is not None else 0.0,
                    1.0 if last_brand and meta.get("brand") == last_brand else 0.0,
                    1.0 if last_cat and meta.get("category") == last_cat else 0.0,
                ]
            )
        return np.asarray(rows, dtype=np.float32)
