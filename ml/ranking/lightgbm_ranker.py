"""LightGBM binary 랭커."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ml.ranking.features import FEATURE_NAMES


class LightGBMRanker:
    def __init__(self, booster: lgb.Booster, feature_names: list[str] | None = None) -> None:
        self.booster = booster
        self.feature_names = feature_names or list(FEATURE_NAMES)

    def score(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.asarray([], dtype=np.float32)
        preds = self.booster.predict(features)
        return np.asarray(preds, dtype=np.float32)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "model.txt"))
        (directory / "meta.json").write_text(
            json.dumps({"feature_names": self.feature_names}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> LightGBMRanker:
        booster = lgb.Booster(model_file=str(directory / "model.txt"))
        feature_names = list(FEATURE_NAMES)
        meta_path = directory / "meta.json"
        if meta_path.exists():
            feature_names = json.loads(meta_path.read_text(encoding="utf-8")).get(
                "feature_names", feature_names
            )
        return cls(booster=booster, feature_names=feature_names)
