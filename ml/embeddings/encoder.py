"""문장 임베딩 클래스"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    # 임베딩 차원 반환
    @property
    def dim(self) -> int:
        return int(self._model.get_embedding_dimension())

    # 문장 임베딩
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        emb = self._model.encode(
            texts,                                  # 임베딩할 문장 리스트
            batch_size=batch_size,                  # 배치 크기
            show_progress_bar=len(texts) > 100,     # 진행 바 표시 여부
            convert_to_numpy=True,                  # 결과를 numpy 배열로 변환
            normalize_embeddings=True,              # 임베딩 벡터 정규화
        )
        return np.asarray(emb, dtype=np.float32)    # 임베딩 결과를 numpy 배열로 반환
