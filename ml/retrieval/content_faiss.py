"""콘텐츠 기반 FAISS Retrieval 클래스"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.embeddings.encoder import Embedder
from ml.vectorstore.faiss_store import FaissItemIndex


class ContentFaissRetriever:
    def __init__(self, index: FaissItemIndex, embedder: Embedder) -> None:
        self.index = index
        self.embedder = embedder
        self._id_to_pos = {iid: i for i, iid in enumerate(index.item_ids)}

    # FAISS 인덱스와 임베더로 콘텐츠 기반 FAISS Retrieval 클래스 생성
    @classmethod
    def load(cls, index_dir: Path, model_name: str) -> ContentFaissRetriever:
        return cls(index=FaissItemIndex.load(index_dir), embedder=Embedder(model_name))

    # 아이템 ID 리스트로 콘텐츠 기반 추천
    # 목적: 최근에 본/산 상품들(시드)과 텍스트가 비슷한 상품 k개 찾기.
    def recommend_from_item_ids(
        self,
        seed_item_ids: list[str],                           # 최근에 본/산 상품들(시드)
        k: int,                                             # 추천할 상품 개수
        exclude: set[str] | None = None,                    # 제외할 상품 ID 집합
    ) -> list[tuple[str, float]]:
        exclude = exclude or set()
        vecs: list[np.ndarray] = []

        # 존재하는 아이템 ID로 벡터 재구성
        for iid in seed_item_ids:
            pos = self._id_to_pos.get(iid)                  # 아이템 ID에 대한 인덱스 조회
            if pos is None:
                continue
            vec = self.index.index.reconstruct(int(pos))    # 인덱스에서 벡터 재구성
            vecs.append(np.asarray(vec, dtype=np.float32))  # 벡터를 numpy 배열로 변환

        if not vecs:                                        # 벡터가 없으면 빈 리스트 반환
            return []

        query = np.mean(np.stack(vecs, axis=0), axis=0, keepdims=True)      # 벡터 평균
        norm = np.linalg.norm(query, axis=1, keepdims=True) + 1e-12         # 벡터 정규화
        query = (query / norm).astype(np.float32)                           # 벡터 정규화 결과를 numpy 배열로 변환
        hits = self.index.search(query, top_k=k + len(exclude) + len(seed_item_ids))[0]    # 검색 결과 반환

        out: list[tuple[str, float]] = []
        for item_id, score in hits:                                 # 검색 결과 순회
            if item_id in exclude or item_id in seed_item_ids:
                continue                                            # 제외할 아이템 ID이거나 시드 아이템 ID이면 건너뜀
            out.append((item_id, score))                            # 추천 아이템 추가
            if len(out) >= k:                                       # 추천 아이템 개수가 k개 이상이면 종료
                break
        return out                                                  # 추천 아이템 리스트 반환

    # 텍스트로 콘텐츠 기반 추천
    # 목적: 텍스트와 비슷한 상품 k개 찾기.
    def recommend_from_text(
        self,
        query_text: str,                                    # 쿼리 텍스트
        k: int,                                             # 추천할 상품 개수
        exclude: set[str] | None = None,                    # 제외할 상품 ID 집합
    ) -> list[tuple[str, float]]:
        exclude = exclude or set()

        q = self.embedder.encode([query_text])                      # 텍스트 임베딩
        hits = self.index.search(q, top_k=k + len(exclude))[0]      # 검색 결과 반환

        out: list[tuple[str, float]] = []
        for item_id, score in hits:                                 # 검색 결과 순회
            if item_id in exclude:
                continue                                            # 제외할 아이템 ID이면 건너뜀
            out.append((item_id, score))                            # 추천 아이템 추가
            if len(out) >= k:                                       # 추천 아이템 개수가 k개 이상이면 종료
                break
        return out                                                  # 추천 아이템 리스트 반환


# 콘텐츠 인덱스 생성
# 목적: 상품 데이터로부터 콘텐츠 기반 FAISS 인덱스 생성.
def build_content_index(items_path: Path, model_name: str, index_dir: Path) -> FaissItemIndex:
    items = pd.read_parquet(items_path, columns=["item_id", "doc_text"])    # 상품 데이터 로드
    texts = items["doc_text"].fillna("").astype(str).tolist()               # 텍스트 리스트 생성
    item_ids = items["item_id"].astype(str).tolist()                        # 아이템 ID 리스트 생성    
    embedder = Embedder(model_name)                                         # 임베더 생성
    embeddings = embedder.encode(texts)                                     # 텍스트 임베딩
    index = FaissItemIndex.build(embeddings, item_ids)                      # 콘텐츠 기반 FAISS 인덱스 생성
    index.save(index_dir)                                                   # 인덱스 저장
    return index                                                            # 콘텐츠 기반 FAISS 인덱스 반환
