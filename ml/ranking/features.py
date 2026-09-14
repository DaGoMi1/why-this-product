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

    def _content_max_sims(self, seeds: list[str], candidate_ids: list[str]) -> np.ndarray:
        n = len(candidate_ids)
        out = np.zeros(n, dtype=np.float32)
        seed_vecs = [v for s in seeds if (v := self._item_vec(s)) is not None]
        if not seed_vecs:
            return out
        have: list[np.ndarray] = []
        pos: list[int] = []
        for i, item_id in enumerate(candidate_ids):
            vec = self._item_vec(item_id)
            if vec is None:
                continue
            have.append(vec)
            pos.append(i)
        if not have:
            return out
        sims = np.stack(seed_vecs) @ np.stack(have).T
        out[np.asarray(pos, dtype=np.int32)] = sims.max(axis=0)
        return out

    def _ials_scores(self, user_id: str, candidate_ids: list[str]) -> np.ndarray:
        n = len(candidate_ids)
        out = np.zeros(n, dtype=np.float32)
        if self.ials is None:
            return out
        uidx = self.ials._user_index.get(user_id)
        if uidx is None:
            return out
        user_f = self.ials.user_factors[uidx]
        idxs = np.fromiter(
            (self.ials._item_index.get(item_id, -1) for item_id in candidate_ids),
            dtype=np.int32,
            count=n,
        )
        valid = idxs >= 0
        if not np.any(valid):
            return out
        out[valid] = self.ials.item_factors[idxs[valid]] @ user_f
        return out

    def matrix(
        self,
        user_id: str,
        history: list[str],
        candidate_ids: list[str],
    ) -> np.ndarray:
        n = len(candidate_ids)
        if n == 0:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        seeds = history[-5:]
        last = seeds[-1] if seeds else None
        last_meta = self.items_meta.get(last, {}) if last else {}
        last_brand = last_meta.get("brand")
        last_cat = last_meta.get("category")
        pop_fallback = len(self._pop_rank) + 1
        log_pop = np.empty(n, dtype=np.float32)
        item_n = np.empty(n, dtype=np.float32)
        item_mean = np.empty(n, dtype=np.float32)
        pop_rank = np.empty(n, dtype=np.float32)
        same_brand = np.zeros(n, dtype=np.float32)
        same_cat = np.zeros(n, dtype=np.float32)
        for i, item_id in enumerate(candidate_ids):
            log_pop[i] = math.log1p(float(self.popularity._scores.get(item_id, 0.0)))
            item_n[i] = float(self.item_n.get(item_id, 0))
            item_mean[i] = float(self.item_mean_rating.get(item_id, 0.0))
            pop_rank[i] = float(self._pop_rank.get(item_id, pop_fallback))
            meta = self.items_meta.get(item_id, {})
            if last_brand and meta.get("brand") == last_brand:
                same_brand[i] = 1.0
            if last_cat and meta.get("category") == last_cat:
                same_cat[i] = 1.0
        return np.column_stack(
            (
                log_pop,
                item_n,
                item_mean,
                np.full(n, float(len(history)), dtype=np.float32),
                pop_rank,
                self._content_max_sims(seeds, candidate_ids),
                self._ials_scores(user_id, candidate_ids),
                same_brand,
                same_cat,
            )
        )
