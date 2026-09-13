"""iALS 협업 필터링 추천 서비스 (암시적 ALS 인자)"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 최소 평점에 따른 변형 매핑
VARIANT_BY_MIN_RATING = {
    0: "all",
    3: "rating_ge_3",
    4: "rating_ge_4",
    5: "rating_ge_5",
}


class IALSRetriever:
    def __init__(
        self,
        user_ids: list[str],            # 사용자 ID 목록
        item_ids: list[str],            # 아이템 ID 목록
        user_factors: np.ndarray,       # 사용자 인자 행렬
        item_factors: np.ndarray,       # 아이템 인자 행렬
        variant: str,                   # VARIANT_BY_MIN_RATING에 따른 변형
    ) -> None:
        if user_factors.shape[0] != len(user_ids):
            raise ValueError("user_factors rows must match user_ids") # 사용자 인자 행렬의 행 수가 사용자 ID 목록의 길이와 일치해야 함
        if item_factors.shape[0] != len(item_ids):
            raise ValueError("item_factors rows must match item_ids") # 아이템 인자 행렬의 행 수가 아이템 ID 목록의 길이와 일치해야 함
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.user_factors = np.asarray(user_factors, dtype=np.float32)
        self.item_factors = np.asarray(item_factors, dtype=np.float32)
        self.variant = variant
        self._user_index = {u: i for i, u in enumerate(user_ids)}
        self._item_index = {it: i for i, it in enumerate(item_ids)}

    def has_user(self, user_id: str) -> bool: # 사용자 ID가 존재하는지 확인
        return user_id in self._user_index

    def score_item(self, user_id: str, item_id: str) -> float:
        uidx = self._user_index.get(user_id)
        iidx = self._item_index.get(item_id)
        if uidx is None or iidx is None:
            return 0.0
        return float(self.user_factors[uidx] @ self.item_factors[iidx])

    def recommend(
        self,
        user_id: str,                       # 사용자 ID
        k: int,                             # 추천 상품 개수
        exclude: set[str] | None = None,    # 제외할 아이템 ID 목록
    ) -> list[tuple[str, float]]:
        uidx = self._user_index.get(user_id) # 사용자 ID 인덱스
        if uidx is None:
            return []
        exclude = exclude or set()

        scores = self.user_factors[uidx] @ self.item_factors.T  # 사용자 인자 행렬과 아이템 인자 행렬의 곱
        extra = len(exclude) + 16                               # 제외할 아이템 개수 + 16
        take = min(len(scores), k + extra)                      # 추천 상품 개수와 제외할 아이템 개수 + 16 중 작은 값
        top_idx = np.argpartition(-scores, take - 1)[:take]     # 추천 상품 인덱스
        top_idx = top_idx[np.argsort(-scores[top_idx])]         # 추천 상품 인덱스 정렬

        out: list[tuple[str, float]] = []
        for i in top_idx:
            item_id = self.item_ids[int(i)]                     # 아이템 ID
            if item_id in exclude:                              # 제외할 아이템 ID 목록에 포함되어 있으면 제외
                continue
            out.append((item_id, float(scores[int(i)])))        # 추천 상품 목록에 추가
            if len(out) >= k:                                   # 추천 상품 개수가 추천 상품 개수보다 크면 종료
                break
        return out

    # 학습이 끝난 벡터와 ID 목록을 저장
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "user_factors.npy", self.user_factors) # 사용자 인자 행렬 저장
        np.save(directory / "item_factors.npy", self.item_factors) # 아이템 인자 행렬 저장
        (directory / "user_ids.json").write_text(
            json.dumps(self.user_ids, ensure_ascii=False), encoding="utf-8"
        )
        (directory / "item_ids.json").write_text(
            json.dumps(self.item_ids, ensure_ascii=False), encoding="utf-8"
        )
        (directory / "meta.json").write_text(
            json.dumps({"variant": self.variant}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> IALSRetriever:
        user_factors = np.load(directory / "user_factors.npy") # 사용자 인자 행렬
        item_factors = np.load(directory / "item_factors.npy") # 아이템 인자 행렬
        user_ids = json.loads((directory / "user_ids.json").read_text(encoding="utf-8")) # 사용자 ID 목록
        item_ids = json.loads((directory / "item_ids.json").read_text(encoding="utf-8")) # 아이템 ID 목록

        meta_path = directory / "meta.json"
        variant = "all"
        if meta_path.exists():
            variant = json.loads(meta_path.read_text(encoding="utf-8")).get("variant", "all")
        return cls(
            user_ids=user_ids,
            item_ids=item_ids,
            user_factors=user_factors,
            item_factors=item_factors,
            variant=variant,
        )
