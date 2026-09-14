"""인기도 기준 Retrieval 클래스"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class PopularityRetriever:
    """Rank items by train interaction count (desc)."""

    def __init__(self, item_ids: list[str], scores: dict[str, float]) -> None:
        self._item_ids = item_ids
        self._scores = scores

    # train 데이터로부터 인기도 기준 Retrieval 클래스 생성
    @classmethod
    def from_train(
        cls,
        train_path: Path,
        top_n: int | None = None,
        min_rating: float | None = None,
    ) -> PopularityRetriever:
        if min_rating is None:
            train = pd.read_parquet(train_path, columns=["item_id"])
        else:
            train = pd.read_parquet(train_path, columns=["item_id", "rating"])
            train = train[train["rating"] >= float(min_rating)]
        counts = train["item_id"].value_counts()
        if top_n is not None:
            counts = counts.head(top_n)
        item_ids = counts.index.astype(str).tolist()
        scores = {str(i): float(c) for i, c in counts.items()}
        return cls(item_ids=item_ids, scores=scores)

    # 인기도 기준으로 추천 아이템 반환
    def recommend(self, k: int, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        exclude = exclude or set()
        out: list[tuple[str, float]] = []
        for item_id in self._item_ids:
            if item_id in exclude:
                continue
            out.append((item_id, self._scores[item_id]))
            if len(out) >= k:
                break
        return out

    # 아이템의 인기도 점수 반환
    def score(self, item_id: str) -> float:
        return self._scores.get(item_id, 0.0)
