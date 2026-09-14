"""popularity 후보 위 부스팅 랭커 학습 (LightGBM / XGBoost / CatBoost).

양성은 train `rating >= popularity_min_rating`(기본 5) 시퀀스의 마지막 아이템.
후보·drop 기준은 같은 임계의 popularity top-N. item_mean_rating은 전 평점.
"""

from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier

from ml.config import load_mvp_config, resolve_path
from ml.ranking.catboost_ranker import CatBoostRanker
from ml.ranking.features import FEATURE_NAMES, FeatureBuilder
from ml.ranking.lightgbm_ranker import LightGBMRanker
from ml.ranking.xgboost_ranker import XGBoostRanker
from ml.retrieval.content_faiss import ContentFaissRetriever
from ml.retrieval.ials import IALSRetriever
from ml.retrieval.popularity import PopularityRetriever


def _items_meta(items: pd.DataFrame) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for r in items.itertuples(index=False):
        brand = r.brand
        if brand is not None and isinstance(brand, float) and pd.isna(brand):
            brand = None
        category = r.category
        if category is not None and isinstance(category, float) and pd.isna(category):
            category = None
        meta[str(r.item_id)] = {"brand": brand, "category": category}
    return meta


def _build_table(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    processed = resolve_path(cfg["data"]["processed_dir"])
    train_path = processed / "interactions_train.parquet"
    items_path = processed / "items.parquet"
    if not train_path.exists() or not items_path.exists():
        raise FileNotFoundError(f"Missing {train_path} or {items_path}. Run prepare_splits first.")

    pop_n = int(cfg["retrieval"]["popularity_top_n"])
    pop_min = cfg["retrieval"].get("popularity_min_rating", 5)
    train = pd.read_parquet(train_path, columns=["user_id", "item_id", "rating", "timestamp"])
    train = train.sort_values("timestamp")
    items = pd.read_parquet(items_path)
    popularity = PopularityRetriever.from_train(
        train_path, top_n=pop_n, min_rating=float(pop_min)
    )

    content = None
    index_dir = resolve_path(cfg["rag"]["faiss_index_dir"])
    if (index_dir / "index.faiss").exists():
        content = ContentFaissRetriever.load(index_dir, cfg["rag"]["embedding_model"])

    ials = None
    ials_cfg = cfg.get("ials", {})
    ials_dir = resolve_path(ials_cfg.get("artifact_dir", "data/processed/ials")) / str(
        ials_cfg.get("variant", "rating_ge_5")
    )
    if (ials_dir / "user_factors.npy").exists():
        ials = IALSRetriever.load(ials_dir)

    builder = FeatureBuilder.from_train(
        train, popularity, _items_meta(items), content, ials
    )

    positives = train[train["rating"] >= float(pop_min)]
    by_user = (
        positives.groupby("user_id", sort=False)["item_id"]
        .apply(lambda s: s.astype(str).tolist())
        .to_dict()
    )
    n_train_users = int(train["user_id"].nunique())
    n_users = len(by_user)
    n_multi = sum(1 for hist in by_user.values() if len(hist) >= 2)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    kept = 0
    dropped_not_in_pop = 0

    for i, (user_id, hist) in enumerate(by_user.items(), start=1):
        if len(hist) < 2:
            continue
        history = hist[:-1]
        positive = hist[-1]
        exclude = set(history)
        candidates = [iid for iid, _ in popularity.recommend(k=pop_n, exclude=exclude)]
        if positive not in candidates:
            dropped_not_in_pop += 1
            continue
        feat = builder.matrix(user_id, history, candidates)
        labels = np.asarray([1.0 if iid == positive else 0.0 for iid in candidates], dtype=np.float32)
        xs.append(feat)
        ys.append(labels)
        kept += 1
        if i % 5000 == 0:
            print(f"  ... scanned {i}/{n_users} kept={kept}")

    print(
        f"[coverage] train_users={n_train_users:,} ge{int(pop_min)}_users={n_users:,} "
        f"multi_ge{int(pop_min)}={n_multi:,} ({100.0 * n_multi / max(n_users, 1):.1f}% of ge{int(pop_min)})"
    )
    print(
        f"[coverage] kept_users={kept:,} dropped_positive_not_in_pop={dropped_not_in_pop:,} "
        f"(of multi {n_multi:,} = {100.0 * dropped_not_in_pop / n_multi:.1f}% dropped)"
    )
    if not xs:
        raise ValueError("no ranker training rows")

    x = np.vstack(xs)
    y = np.concatenate(ys)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    print(f"[coverage] rows={len(y):,} positives={n_pos:,} negatives={n_neg:,}")
    return x, y


def _train_lightgbm(x: np.ndarray, y: np.ndarray, rank_cfg: dict) -> LightGBMRanker:
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    dtrain = lgb.Dataset(x, label=y, feature_name=list(FEATURE_NAMES))
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": int(rank_cfg.get("num_leaves", 31)),
        "learning_rate": float(rank_cfg.get("learning_rate", 0.05)),
        "verbose": -1,
        "seed": 42,
        "scale_pos_weight": (n_neg / n_pos) if n_pos else 1.0,
    }
    booster = lgb.train(params, dtrain, num_boost_round=int(rank_cfg.get("n_estimators", 200)))
    return LightGBMRanker(booster)


def _train_xgboost(x: np.ndarray, y: np.ndarray, rank_cfg: dict) -> XGBoostRanker:
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    dtrain = xgb.DMatrix(x, label=y, feature_names=list(FEATURE_NAMES))
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(rank_cfg.get("max_depth", 5)),
        "eta": float(rank_cfg.get("learning_rate", 0.05)),
        "seed": 42,
        "scale_pos_weight": (n_neg / n_pos) if n_pos else 1.0,
        "tree_method": "hist",
    }
    booster = xgb.train(params, dtrain, num_boost_round=int(rank_cfg.get("n_estimators", 200)))
    return XGBoostRanker(booster)


def _train_catboost(x: np.ndarray, y: np.ndarray, rank_cfg: dict) -> CatBoostRanker:
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    model = CatBoostClassifier(
        iterations=int(rank_cfg.get("n_estimators", 200)),
        learning_rate=float(rank_cfg.get("learning_rate", 0.05)),
        depth=int(rank_cfg.get("max_depth", 5)),
        loss_function="Logloss",
        scale_pos_weight=(n_neg / n_pos) if n_pos else 1.0,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x, y)
    return CatBoostRanker(model)


def main() -> None:
    cfg = load_mvp_config()
    rank_cfg = cfg.get("ranking", {})
    models = [str(m) for m in rank_cfg.get("models", ["lightgbm", "xgboost", "catboost"])]
    x, y = _build_table(cfg)
    out_root = resolve_path(rank_cfg.get("artifact_dir", "data/processed/ranker"))
    trainers = {
        "lightgbm": _train_lightgbm,
        "xgboost": _train_xgboost,
        "catboost": _train_catboost,
    }
    for name in models:
        trainer = trainers.get(name)
        if trainer is None:
            raise ValueError(f"unknown ranker model: {name}")
        print(f"[train] {name}")
        ranker = trainer(x, y, rank_cfg)
        out_dir = out_root / name
        ranker.save(out_dir)
        print(f"[done] {name} {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
