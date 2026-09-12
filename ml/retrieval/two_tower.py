"""히스토리 mean-pool Two-Tower retrieve (유저 ID 임베딩 없음)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TwoTowerRetriever:
    def __init__(
        self,
        item_ids: list[str],
        item_factors: np.ndarray,
        last_n: int = 10,
    ) -> None:
        if item_factors.shape[0] != len(item_ids):
            raise ValueError("item_factors rows must match item_ids")
        self.item_ids = item_ids
        self.item_factors = np.asarray(item_factors, dtype=np.float32)
        self.last_n = int(last_n)
        self._item_index = {it: i for i, it in enumerate(item_ids)}

    # 추천 결과 반환
    def recommend(
        self,
        history: list[str],
        k: int,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        in_vocab = [item_id for item_id in history if item_id in self._item_index]
        use = in_vocab[-self.last_n :]
        if not use:
            return []

        idxs = [self._item_index[item_id] for item_id in use]
        user_vec = self.item_factors[idxs].mean(axis=0)
        norm = float(np.linalg.norm(user_vec))
        if norm <= 0:
            return []
        user_vec = user_vec / norm
        scores = self.item_factors @ user_vec

        exclude = exclude or set()
        extra = len(exclude) + 16
        take = min(len(scores), k + extra)
        top_idx = np.argpartition(-scores, take - 1)[:take]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        out: list[tuple[str, float]] = []
        for i in top_idx:
            item_id = self.item_ids[int(i)]
            if item_id in exclude:
                continue
            out.append((item_id, float(scores[int(i)])))
            if len(out) >= k:
                break
        return out

    # 모델 저장
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "item_factors.npy", self.item_factors)
        (directory / "item_ids.json").write_text(
            json.dumps(self.item_ids, ensure_ascii=False), encoding="utf-8"
        )
        (directory / "meta.json").write_text(
            json.dumps({"last_n": self.last_n}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 저장된 모델 로드
    @classmethod
    def load(cls, directory: Path) -> TwoTowerRetriever:
        item_factors = np.load(directory / "item_factors.npy")
        item_ids = json.loads((directory / "item_ids.json").read_text(encoding="utf-8"))
        last_n = 10
        meta_path = directory / "meta.json"
        if meta_path.exists():
            last_n = int(json.loads(meta_path.read_text(encoding="utf-8")).get("last_n", 10))
        return cls(item_ids=item_ids, item_factors=item_factors, last_n=last_n)
