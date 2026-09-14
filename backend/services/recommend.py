"""추천 서비스: 인기도 + 콘텐츠 FAISS + iALS + 부스팅 재정렬"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from ml.config import load_mvp_config, resolve_path
from ml.ranking import Ranker, load_ranker
from ml.ranking.features import FeatureBuilder
from ml.rerank.mmr import ItemSimilarity, mmr_rerank
from ml.retrieval.content_faiss import ContentFaissRetriever
from ml.retrieval.ials import IALSRetriever
from ml.retrieval.merge import merge_candidates, rrf_fuse
from ml.retrieval.popularity import PopularityRetriever

COLD_CONTENT_MAX = 20


@dataclass
class RecommendResult:
    items: list[dict]
    strategy: str


class RecommendService:
    def __init__(
        self,
        popularity: PopularityRetriever,        # 인기도 추천 서비스
        content: ContentFaissRetriever | None,  # 콘텐츠 추천 서비스
        ials: IALSRetriever | None,             # iALS CF 추천 서비스
        train_by_user: dict[str, list[str]],    # 사용자 히스토리 [user_id: [item_id, ...]]
        items_meta: dict[str, dict],            # 아이템 메타데이터 [item_id: {title: str, brand: str, category: str}]
        cfg: dict,                              # 설정
        features: FeatureBuilder | None = None, # 피처 빌더
        ranker: Ranker | None = None,   # 랭커
        train_item_ids: set[str] | None = None,
    ) -> None:
        self.popularity = popularity
        self.content = content
        self.ials = ials
        self.train_by_user = train_by_user
        self.items_meta = items_meta
        self.cfg = cfg
        self.features = features
        self.ranker = ranker
        self.train_item_ids = train_item_ids or set()
        self.similarity = ItemSimilarity(content, items_meta)

    # 처리된 데이터로 추천 서비스 초기화
    @classmethod
    def from_processed(cls) -> RecommendService:
        cfg = load_mvp_config()                           # MVP 설정 로드
        processed = resolve_path(cfg["data"]["processed_dir"])      # 처리된 데이터 경로 절대 경로 변환
        train_path = processed / "interactions_train.parquet"       # 학습 데이터 경로
        items_path = processed / "items.parquet"                    # 아이템 데이터 경로
        if not train_path.exists() or not items_path.exists():
            raise FileNotFoundError("처리된 데이터가 없습니다. download_data → prepare_splits 순서로 실행해주세요.")

        pop_n = int(cfg["retrieval"]["popularity_top_n"])
        pop_min = cfg["retrieval"].get("popularity_min_rating")
        pop = PopularityRetriever.from_train(
            train_path,
            top_n=pop_n,
            min_rating=float(pop_min) if pop_min is not None else None,
        )
        train = pd.read_parquet(train_path, columns=["user_id", "item_id", "rating", "timestamp"]) # 학습 데이터 로드
        train = train.sort_values("timestamp") # 학습 데이터 정렬
        train_by_user = (train.groupby("user_id")["item_id"].apply(lambda s: s.astype(str).tolist()).to_dict()) # 사용자 히스토리 생성
        train_item_ids = {str(i) for i in train["item_id"].astype(str).unique()}
        items = pd.read_parquet(items_path) # 아이템 데이터 로드
        items_meta: dict[str, dict] = {}

        # 아이템 메타데이터 생성
        for r in items.itertuples(index=False):
            brand = r.brand
            if brand is not None and isinstance(brand, float) and pd.isna(brand):
                brand = None
            category = r.category
            if category is not None and isinstance(category, float) and pd.isna(category):
                category = None
            title = r.title
            if title is not None and isinstance(title, float) and pd.isna(title):
                title = None
            items_meta[str(r.item_id)] = {
                "title": title,
                "brand": brand,
                "category": category,
            }

        content = None
        index_dir = resolve_path(cfg["rag"]["faiss_index_dir"]) # FAISS 인덱스 경로 절대 경로 변환
        if (index_dir / "index.faiss").exists(): # FAISS 인덱스 파일이 존재하면
            content = ContentFaissRetriever.load(index_dir, cfg["rag"]["embedding_model"]) # 콘텐츠 추천 서비스 초기화

        ials = None
        ials_cfg = cfg.get("ials", {})
        variant = str(ials_cfg.get("variant", "all"))
        ials_dir = resolve_path(ials_cfg.get("artifact_dir", "data/processed/ials")) / variant
        if (ials_dir / "user_factors.npy").exists():
            ials = IALSRetriever.load(ials_dir)

        features = FeatureBuilder.from_train(train, pop, items_meta, content, ials)
        rank_cfg = cfg.get("ranking", {})
        rank_dir = resolve_path(rank_cfg.get("artifact_dir", "data/processed/ranker"))
        ranker = load_ranker(rank_dir, str(rank_cfg.get("model", "lightgbm")))

        return cls(
            popularity=pop,
            content=content,
            ials=ials,
            train_by_user=train_by_user,
            items_meta=items_meta,
            cfg=cfg,
            features=features,
            ranker=ranker,
            train_item_ids=train_item_ids,
        )

    # 사용자 기반 추천
    def _content_hits(
        self,
        history: list[str],
        exclude: set[str],
        content_k: int,
    ) -> list[tuple[str, float]]:
        if self.content is None or not history:
            return []
        seeds = history[-5:]
        content_mode = str(self.cfg["retrieval"].get("content_query_mode", "mean"))
        return self.content.recommend_from_item_ids(
            seeds, k=content_k, exclude=exclude, mode=content_mode
        )

    def _ials_hits(
        self,
        user_id: str,
        exclude: set[str],
        final_k: int,
        ials: IALSRetriever | None = None,
    ) -> list[tuple[str, float]]:
        ials_model = ials if ials is not None else self.ials
        if ials_model is None or not ials_model.has_user(user_id):
            return []
        ials_k = int(self.cfg.get("ials", {}).get("top_k", 100))
        return ials_model.recommend(user_id, k=max(final_k, ials_k), exclude=exclude)

    def recommend_for_user(
        self,
        user_id: str,
        k: int | None = None,
        use_content: bool = False,
        use_ials: bool = False,
        use_hybrid: bool = False,
        use_ranker: bool = False,
        use_mmr: bool = True,
        ials: IALSRetriever | None = None,
        ranker: Ranker | None = None,
        hybrid_channels: list[str] | None = None,
    ) -> RecommendResult:
        final_k = k or int(self.cfg["rerank"]["final_k"])               # 최종 추천 상품 개수
        pop_n = int(self.cfg["retrieval"]["popularity_top_n"])          # 인기도 추천 상품 개수
        content_k = int(self.cfg["retrieval"]["content_faiss_top_k"])   # 콘텐츠 추천 상품 개수
        merge_k = int(self.cfg["retrieval"]["hybrid_merge_k"])          # 인기도와 콘텐츠 기반 추천 결과를 혼합하여 추천 상품 개수

        history = self.train_by_user.get(user_id, [])                   # 사용자 히스토리
        exclude = set(history)                                     # 제외할 아이템 목록
        pop_hits = self.popularity.recommend(k=pop_n, exclude=exclude)  # 인기도 추천 결과

        if use_hybrid:
            channels = hybrid_channels or list(
                self.cfg["retrieval"].get("hybrid_channels", ["popularity", "ials", "content"])
            )
            lists: list[list[tuple[str, float]]] = []
            names: list[str] = []
            if "popularity" in channels:
                lists.append(pop_hits)
                names.append("pop")
            if "ials" in channels:
                lists.append(self._ials_hits(user_id, exclude, final_k, ials=ials))
                names.append("ials")
            if "content" in channels:
                lists.append(self._content_hits(history, exclude, content_k))
                names.append("content")
            rrf_k = int(self.cfg["retrieval"].get("rrf_k", 60))
            merged = rrf_fuse(lists, rrf_k=rrf_k, merge_k=merge_k)
            return self._to_result(merged[:final_k], "rrf_" + "_".join(names))

        # use_ials가 켜지면 iALS만 쓴다 (hybrid가 아닐 때).
        ials_model = ials if ials is not None else self.ials
        if use_ials:
            hits = self._ials_hits(user_id, exclude, final_k, ials=ials)
            if hits:
                variant = ials_model.variant if ials_model is not None else "ials"
                return self._to_result(hits[:final_k], f"ials_{variant}")
            return self._to_result(pop_hits[:final_k], "popularity")

        active_ranker = ranker if ranker is not None else self.ranker
        if use_ranker and active_ranker is not None and self.features is not None and pop_hits:
            cand_ids = [item_id for item_id, _ in pop_hits]
            feat = self.features.matrix(user_id, history, cand_ids)
            scores = active_ranker.score(feat)
            order = scores.argsort()[::-1]
            ranked = [(cand_ids[int(i)], float(scores[int(i)])) for i in order]
            return self._to_result(ranked[:final_k], getattr(active_ranker, "name", "ranker_lgbm"))

        if use_content and self.content is not None and history:
            content_hits = self._content_hits(history, exclude, content_k)
            merged = merge_candidates(pop_hits, content_hits, merge_k=merge_k)
            return self._to_result(merged[:final_k], "popularity+content")

        pool = self.candidate_pool(user_id)
        if use_mmr and pool:
            lam = float(self.cfg.get("rerank", {}).get("lambda_diversity", 0.7))
            ranked = mmr_rerank(pool, final_k, lam, self.similarity)
            has_cold = any(iid not in self.train_item_ids for iid, _ in pool)
            strategy = "popularity+cold+mmr" if has_cold else "popularity+mmr"
            return self._to_result(ranked, strategy)
        return self._to_result(pool[:final_k], "popularity")

    def candidate_pool(self, user_id: str) -> list[tuple[str, float]]:
        """pop-200 + train에 없는 content 이웃 최대 20. rel은 log pop count."""
        history = self.train_by_user.get(user_id, [])
        exclude = set(history)
        pop_n = int(self.cfg["retrieval"]["popularity_top_n"])
        content_k = int(self.cfg["retrieval"]["content_faiss_top_k"])
        pop_hits = self.popularity.recommend(k=pop_n, exclude=exclude)
        seen = {item_id for item_id, _ in pop_hits}
        pool = [(item_id, math.log1p(score)) for item_id, score in pop_hits]
        if self.content is None or not history:
            return pool
        cold_n = 0
        for item_id, _score in self._content_hits(history, exclude, content_k):
            if item_id in seen or item_id in self.train_item_ids:
                continue
            pool.append((item_id, math.log1p(self.popularity.score(item_id))))
            seen.add(item_id)
            cold_n += 1
            if cold_n >= COLD_CONTENT_MAX:
                break
        return pool

    def _to_result(
        self,
        hits: list[tuple[str, float]],
        strategy: str,
    ) -> RecommendResult:
        items = []
        for item_id, score in hits:
            meta = self.items_meta.get(item_id, {})
            items.append(
                {
                    "item_id": item_id,
                    "score": round(float(score), 6),
                    "title": meta.get("title"),
                    "brand": meta.get("brand"),
                    "category": meta.get("category"),
                }
            )
        return RecommendResult(items=items, strategy=strategy)

    # 쿼리 기반 추천
    def recommend_from_query(
        self,
        query: str,
        k: int | None = None,
    ) -> RecommendResult:
        final_k = k or int(self.cfg["rerank"]["final_k"]) # 최종 추천 상품 개수
        if self.content is None:
            raise RuntimeError("Content FAISS index not built. Run scripts.build_faiss")
        content_k = int(self.cfg["retrieval"]["content_faiss_top_k"]) # 콘텐츠 추천 상품 개수
        merge_k = int(self.cfg["retrieval"]["hybrid_merge_k"]) # 인기도와 콘텐츠 기반 추천 결과를 혼합하여 추천 상품 개수

        pop_hits = self.popularity.recommend(k=int(self.cfg["retrieval"]["popularity_top_n"])) # 인기도 추천 결과
        content_hits = self.content.recommend_from_text(query, k=content_k) # 콘텐츠 추천 결과
        rrf_k = int(self.cfg["retrieval"].get("rrf_k", 60))
        merged = rrf_fuse(
            [pop_hits, content_hits], rrf_k=rrf_k, merge_k=merge_k
        )[:final_k]
        return self._to_result(merged, "rrf_pop_content_query")

# 추천 서비스 인스턴스 캐시
@lru_cache(maxsize=1)
def get_recommend_service() -> RecommendService:
    return RecommendService.from_processed()
