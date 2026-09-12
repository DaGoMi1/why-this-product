"""iALS 학습: 각 암시적 라벨 변형에 대해 train 상호작용을 학습"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from ml.config import load_mvp_config, resolve_path
from ml.retrieval.ials import VARIANT_BY_MIN_RATING, IALSRetriever


def _positives(train: pd.DataFrame, min_rating: float) -> pd.DataFrame:
    df = train
    if min_rating > 0:
        df = df[df["rating"] >= min_rating]
    agg = (
        df.groupby(["user_id", "item_id"], sort=False)
        .size()
        .reset_index(name="count")
    )
    agg["user_id"] = agg["user_id"].astype(str)
    agg["item_id"] = agg["item_id"].astype(str)
    return agg


def train_one(
    train: pd.DataFrame,
    min_rating: float,
    factors: int,
    regularization: float,
    iterations: int,
    alpha: float,
    out_dir: Path,
) -> IALSRetriever:
    from implicit.als import AlternatingLeastSquares

    agg = _positives(train, min_rating)
    if agg.empty:
        raise ValueError(f"no positives for min_rating={min_rating}")

    user_ids = sorted(agg["user_id"].unique().tolist())
    item_ids = sorted(agg["item_id"].unique().tolist())
    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {it: i for i, it in enumerate(item_ids)}

    rows = agg["user_id"].map(user_index).to_numpy()
    cols = agg["item_id"].map(item_index).to_numpy()
    data = agg["count"].to_numpy(dtype=np.float32)
    user_items = coo_matrix(
        (data, (rows, cols)), shape=(len(user_ids), len(item_ids))
    ).tocsr()

    print(
        f"[ials] variant={VARIANT_BY_MIN_RATING[int(min_rating)]} "
        f"users={len(user_ids):,} items={len(item_ids):,} nnz={user_items.nnz:,}"
    )
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
        random_state=42,
        num_threads=1,
    )
    model.fit(user_items)
    retriever = IALSRetriever(
        user_ids=user_ids,
        item_ids=item_ids,
        user_factors=np.asarray(model.user_factors, dtype=np.float32),
        item_factors=np.asarray(model.item_factors, dtype=np.float32),
        variant=VARIANT_BY_MIN_RATING[int(min_rating)],
    )
    retriever.save(out_dir)
    print(f"[done] {out_dir}")
    return retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-rating",
        type=int,
        default=None,
        choices=[0, 3, 4, 5],
        help="Train one variant. Default: train all four.",
    )
    args = parser.parse_args()

    cfg = load_mvp_config()
    ials_cfg = cfg.get("ials", {})
    processed = resolve_path(cfg["data"]["processed_dir"])
    train_path = processed / "interactions_train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Run prepare_splits first.")

    train = pd.read_parquet(train_path, columns=["user_id", "item_id", "rating"])
    artifact_root = resolve_path(ials_cfg.get("artifact_dir", "data/processed/ials"))
    ratings = [args.min_rating] if args.min_rating is not None else [0, 3, 4, 5]

    for min_rating in ratings:
        variant = VARIANT_BY_MIN_RATING[min_rating]
        train_one(
            train=train,
            min_rating=min_rating,
            factors=int(ials_cfg.get("factors", 64)),
            regularization=float(ials_cfg.get("regularization", 0.01)),
            iterations=int(ials_cfg.get("iterations", 15)),
            alpha=float(ials_cfg.get("alpha", 40)),
            out_dir=artifact_root / variant,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
