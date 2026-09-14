"""Ranking: boosting rankers on popularity candidates."""

from pathlib import Path

from ml.ranking.catboost_ranker import CatBoostRanker
from ml.ranking.features import FEATURE_NAMES, FeatureBuilder
from ml.ranking.lightgbm_ranker import LightGBMRanker
from ml.ranking.xgboost_ranker import XGBoostRanker

RANKERS = {
    "lightgbm": LightGBMRanker,
    "xgboost": XGBoostRanker,
    "catboost": CatBoostRanker,
}

Ranker = LightGBMRanker | XGBoostRanker | CatBoostRanker


def load_ranker(root: Path, model: str) -> Ranker | None:
    cls = RANKERS.get(model)
    if cls is None:
        return None
    sub = root / model
    if (sub / cls.model_file).exists():
        return cls.load(sub)
    if model == "lightgbm" and (root / LightGBMRanker.model_file).exists():
        return LightGBMRanker.load(root)
    return None


__all__ = [
    "FEATURE_NAMES",
    "FeatureBuilder",
    "LightGBMRanker",
    "XGBoostRanker",
    "CatBoostRanker",
    "RANKERS",
    "Ranker",
    "load_ranker",
]
