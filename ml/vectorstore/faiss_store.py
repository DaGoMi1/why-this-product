"""FAISS 인덱스 저장/로드/검색 클래스"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class FaissItemIndex:
    def __init__(self, index: faiss.Index, item_ids: list[str]) -> None:
        if index.ntotal != len(item_ids):
            raise ValueError("index size and item_ids length mismatch")
        self.index = index
        self.item_ids = item_ids

    # 아이템 콘텐츠 벡터로 FAISS 인덱스 생성
    @classmethod
    def build(cls, embeddings: np.ndarray, item_ids: list[str]) -> FaissItemIndex:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return cls(index=index, item_ids=item_ids)

    # FAISS 인덱스 저장
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "item_ids.json").write_text(
            json.dumps(self.item_ids, ensure_ascii=False),
            encoding="utf-8",
        )

    # FAISS 인덱스 로드
    @classmethod
    def load(cls, directory: Path) -> FaissItemIndex:
        index = faiss.read_index(str(directory / "index.faiss"))
        item_ids = json.loads((directory / "item_ids.json").read_text(encoding="utf-8"))
        return cls(index=index, item_ids=item_ids)

    # FAISS 인덱스로 검색
    def search(self, query_vecs: np.ndarray, top_k: int) -> list[list[tuple[str, float]]]:
        if query_vecs.ndim == 1:
            query_vecs = query_vecs.reshape(1, -1)
        if query_vecs.dtype != np.float32:
            query_vecs = query_vecs.astype(np.float32)
        scores, indices = self.index.search(query_vecs, top_k)
        results: list[list[tuple[str, float]]] = []
        for row_scores, row_idx in zip(scores, indices, strict=True):
            hits: list[tuple[str, float]] = []
            for score, idx in zip(row_scores, row_idx, strict=True):
                if idx < 0:
                    continue
                hits.append((self.item_ids[int(idx)], float(score)))
            results.append(hits)
        return results
