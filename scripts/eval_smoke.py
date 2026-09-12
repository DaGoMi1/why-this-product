"""오프라인 평가: 인기도 vs 인기도+콘텐츠 랭킹 평가"""

from __future__ import annotations

import sys

import pandas as pd

from backend.services.recommend import RecommendService
from ml.config import load_mvp_config, resolve_path
from ml.eval.metrics import mean_recall_at_k


def main() -> None:
    cfg = load_mvp_config()
    processed = resolve_path(cfg["data"]["processed_dir"])
    valid_path = processed / "interactions_valid.parquet"
    train_path = processed / "interactions_train.parquet"
    index_dir = resolve_path(cfg["rag"]["faiss_index_dir"])

    for path in (train_path, valid_path, processed / "items.parquet"):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run prepare_splits first.")
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"Missing FAISS index at {index_dir}. Run build_faiss first.")

    print("[load] recommend service...")
    service = RecommendService.from_processed()

    valid = pd.read_parquet(valid_path, columns=["user_id", "item_id"])
    # users that also appear in train (warm-start smoke)
    train_users = set(service.train_by_user.keys())
    valid = valid[valid["user_id"].isin(train_users)]
    truth = (
        valid.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(str))).to_dict()
    )
    # sample for smoke speed
    max_users = 200
    user_ids = list(truth.keys())[:max_users]
    truth = {u: truth[u] for u in user_ids}
    print(f"[eval] warm valid users sampled: {len(truth)}")

    k = 10
    pop_recs: dict[str, list[str]] = {}
    hybrid_recs: dict[str, list[str]] = {}
    for i, user_id in enumerate(user_ids, start=1):
        pop = service.recommend_for_user(user_id, k=k, use_content=False)
        hybrid = service.recommend_for_user(user_id, k=k, use_content=True)
        pop_recs[user_id] = [x["item_id"] for x in pop.items]
        hybrid_recs[user_id] = [x["item_id"] for x in hybrid.items]
        if i % 50 == 0:
            print(f"  ... {i}/{len(user_ids)}")

    pop_recall = mean_recall_at_k(pop_recs, truth, k)
    hybrid_recall = mean_recall_at_k(hybrid_recs, truth, k)
    print("--- smoke results ---")
    print(f"Recall@{k} popularity          : {pop_recall:.6f}")
    print(f"Recall@{k} popularity+content  : {hybrid_recall:.6f}")
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
