"""추천 서비스: 인기도 + 콘텐츠 FAISS + iALS (Phase 2 retrieve)"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from ml.config import load_mvp_config, resolve_path
from ml.retrieval.content_faiss import ContentFaissRetriever
from ml.retrieval.ials import IALSRetriever
from ml.retrieval.merge import merge_candidates
from ml.retrieval.popularity import PopularityRetriever


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
    ) -> None:
        self.popularity = popularity
        self.content = content
        self.ials = ials
        self.train_by_user = train_by_user
        self.items_meta = items_meta
        self.cfg = cfg

    # 처리된 데이터로 추천 서비스 초기화
    @classmethod
    def from_processed(cls) -> RecommendService:
        cfg = load_mvp_config()                           # MVP 설정 로드
        processed = resolve_path(cfg["data"]["processed_dir"])      # 처리된 데이터 경로 절대 경로 변환
        train_path = processed / "interactions_train.parquet"       # 학습 데이터 경로
        items_path = processed / "items.parquet"                    # 아이템 데이터 경로
        if not train_path.exists() or not items_path.exists():
            raise FileNotFoundError("처리된 데이터가 없습니다. download_data → prepare_splits 순서로 실행해주세요.")

        pop = PopularityRetriever.from_train(train_path, top_n=int(cfg["retrieval"]["popularity_top_n"])) # 인기도 추천 서비스 초기화
        train = pd.read_parquet(train_path, columns=["user_id", "item_id", "timestamp"]) # 학습 데이터 로드
        train = train.sort_values("timestamp") # 학습 데이터 정렬
        train_by_user = (train.groupby("user_id")["item_id"].apply(lambda s: s.astype(str).tolist()).to_dict()) # 사용자 히스토리 생성
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

        return cls(
            popularity=pop,
            content=content,
            ials=ials,
            train_by_user=train_by_user,
            items_meta=items_meta,
            cfg=cfg,
        )

    # 사용자 기반 추천
    def recommend_for_user(
        self,
        user_id: str,
        k: int | None = None,
        use_content: bool = True,
        use_ials: bool = False,
        ials: IALSRetriever | None = None,
    ) -> RecommendResult:
        final_k = k or int(self.cfg["rerank"]["final_k"])               # 최종 추천 상품 개수
        pop_n = int(self.cfg["retrieval"]["popularity_top_n"])          # 인기도 추천 상품 개수
        content_k = int(self.cfg["retrieval"]["content_faiss_top_k"])   # 콘텐츠 추천 상품 개수
        merge_k = int(self.cfg["retrieval"]["hybrid_merge_k"])          # 인기도와 콘텐츠 기반 추천 결과를 혼합하여 추천 상품 개수

        history = self.train_by_user.get(user_id, [])                   # 사용자 히스토리
        exclude = set(history)                                     # 제외할 아이템 목록
        pop_hits = self.popularity.recommend(k=pop_n, exclude=exclude)  # 인기도 추천 결과

        # 이 슬라이스는 CF와 content를 합치지 않는다. use_ials가 켜지면 iALS만 쓴다.
        ials_model = ials if ials is not None else self.ials
        if use_ials:
            ials_k = int(self.cfg.get("ials", {}).get("top_k", 100))
            hits: list[tuple[str, float]] = []
            if ials_model is not None and ials_model.has_user(user_id):
                hits = ials_model.recommend(
                    user_id, k=max(final_k, ials_k), exclude=exclude
                )
            if hits:
                return self._to_result(hits[:final_k], f"ials_{ials_model.variant}")
            return self._to_result(pop_hits[:final_k], "popularity")

        content_hits: list[tuple[str, float]] = []  # 콘텐츠 추천 결과
        strategy = "popularity"                     # 추천 전략
        if use_content and self.content is not None and history:
            seeds = history[-5:] # 콘텐츠 추천 시드
            content_hits = self.content.recommend_from_item_ids(
                seeds, k=content_k, exclude=exclude
            ) # 콘텐츠 추천 결과
            strategy = "popularity+content" # 추천 전략
            merged = merge_candidates(pop_hits, content_hits, merge_k=merge_k) # 인기도와 콘텐츠 기반 추천 결과를 혼합하여 추천 상품 개수
        else:
            merged = pop_hits[:merge_k] # 인기도 추천 결과

        return self._to_result(merged[:final_k], strategy)

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

        merged = merge_candidates(pop_hits, content_hits, merge_k=merge_k)[:final_k] # 인기도와 콘텐츠 기반 추천 결과를 혼합하여 추천 상품 개수
        items = [] # 추천 상품 목록
        for item_id, score in merged:
            meta = self.items_meta.get(item_id, {}) # 아이템 메타데이터
            items.append(
                {
                    "item_id": item_id,
                    "score": round(float(score), 6),
                    "title": meta.get("title"),
                    "brand": meta.get("brand"),
                    "category": meta.get("category"),
                }
            )
        return RecommendResult(items=items, strategy="popularity+content_query")

# 추천 서비스 인스턴스 캐시
@lru_cache(maxsize=1)
def get_recommend_service() -> RecommendService:
    return RecommendService.from_processed()
