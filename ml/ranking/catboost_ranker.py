"""CatBoost binary 랭커."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from ml.ranking.features import FEATURE_NAMES


class CatBoostRanker:
    name = "ranker_catboost"
    model_file = "model.cbm"

    def __init__(self, model: CatBoostClassifier, feature_names: list[str] | None = None) -> None:
        self.model = model
        self.feature_names = feature_names or list(FEATURE_NAMES)

    def score(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.asarray([], dtype=np.float32)
        preds = self.model.predict_proba(features)[:, 1]
        return np.asarray(preds, dtype=np.float32)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(directory / self.model_file))
        (directory / "meta.json").write_text(
            json.dumps({"feature_names": self.feature_names}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> CatBoostRanker:
        model = CatBoostClassifier()
        model.load_model(str(directory / cls.model_file))
        feature_names = list(FEATURE_NAMES)
        meta_path = directory / "meta.json"
        if meta_path.exists():
            feature_names = json.loads(meta_path.read_text(encoding="utf-8")).get(
                "feature_names", feature_names
            )
        return cls(model=model, feature_names=feature_names)
