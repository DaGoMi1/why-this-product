"""오프라인 평가: popularity / content / iALS / two-tower / RRF / LightGBM Recall@10"""

from __future__ import annotations

import sys

import pandas as pd

from backend.services.recommend import RecommendService
from ml.config import load_mvp_config, resolve_path
from ml.eval.metrics import mean_recall_at_k
from ml.retrieval.ials import IALSRetriever
from ml.retrieval.two_tower import TwoTowerRetriever


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

    content_mean_recs: dict[str, list[str]] = {}
    content_per_seed_recs: dict[str, list[str]] = {}
    multi_seed_users: list[str] = []
    if service.content is None:
        raise RuntimeError("Content FAISS is required for content mean/per_seed eval")
    for user_id in user_ids:
        history = service.train_by_user.get(user_id, [])
        seeds = history[-5:]
        exclude = set(history)
        if len(seeds) >= 2:
            multi_seed_users.append(user_id)
        mean_hits = service.content.recommend_from_item_ids(
            seeds, k=k, exclude=exclude, mode="mean"
        )
        per_hits = service.content.recommend_from_item_ids(
            seeds, k=k, exclude=exclude, mode="per_seed"
        )
        content_mean_recs[user_id] = [item_id for item_id, _ in mean_hits]
        content_per_seed_recs[user_id] = [item_id for item_id, _ in per_hits]

    pop_recall = mean_recall_at_k(pop_recs, truth, k)
    hybrid_recall = mean_recall_at_k(hybrid_recs, truth, k)
    content_mean_recall = mean_recall_at_k(content_mean_recs, truth, k)
    content_per_seed_recall = mean_recall_at_k(content_per_seed_recs, truth, k)
    multi_truth = {u: truth[u] for u in multi_seed_users}
    multi_mean = mean_recall_at_k(
        {u: content_mean_recs[u] for u in multi_seed_users}, multi_truth, k
    )
    multi_per_seed = mean_recall_at_k(
        {u: content_per_seed_recs[u] for u in multi_seed_users}, multi_truth, k
    )
    print("--- smoke results ---")
    print(f"Recall@{k} popularity          : {pop_recall:.6f}")
    print(f"Recall@{k} popularity+content  : {hybrid_recall:.6f}")
    print(f"Recall@{k} content_mean        : {content_mean_recall:.6f}")
    print(f"Recall@{k} content_per_seed    : {content_per_seed_recall:.6f}")
    print(f"[content] multi-seed users (history[-5:] >= 2): {len(multi_seed_users)}/{len(user_ids)}")
    print(f"Recall@{k} content_mean multi  : {multi_mean:.6f}")
    print(f"Recall@{k} content_per_seed multi: {multi_per_seed:.6f}")

    ials_cfg = cfg.get("ials", {})
    ials_root = resolve_path(ials_cfg.get("artifact_dir", "data/processed/ials"))
    variants = ["all", "rating_ge_3", "rating_ge_4", "rating_ge_5"]
    for variant in variants:
        art = ials_root / variant
        if not (art / "user_factors.npy").exists():
            print(f"Recall@{k} ials_{variant:<14}: SKIP (run python -m scripts.train_ials)")
            continue
        retriever = IALSRetriever.load(art)
        ials_recs: dict[str, list[str]] = {}
        for user_id in user_ids:
            result = service.recommend_for_user(
                user_id, k=k, use_ials=True, ials=retriever
            )
            ials_recs[user_id] = [x["item_id"] for x in result.items]
        ials_recall = mean_recall_at_k(ials_recs, truth, k)
        print(f"Recall@{k} ials_{variant:<14}: {ials_recall:.6f}")

    tt_cfg = cfg.get("two_tower", {})
    tt_dir = resolve_path(tt_cfg.get("artifact_dir", "data/processed/two_tower"))
    if not (tt_dir / "item_factors.npy").exists():
        print(f"Recall@{k} two_tower        : SKIP (run python -m scripts.train_two_tower)")
    else:
        retriever = TwoTowerRetriever.load(tt_dir)
        tt_recs: dict[str, list[str]] = {}
        for user_id in user_ids:
            history = service.train_by_user.get(user_id, [])
            exclude = set(history)
            hits = retriever.recommend(history, k=k, exclude=exclude)
            if hits:
                tt_recs[user_id] = [item_id for item_id, _score in hits]
            else:
                pop = service.recommend_for_user(user_id, k=k, use_content=False)
                tt_recs[user_id] = [x["item_id"] for x in pop.items]
        tt_recall = mean_recall_at_k(tt_recs, truth, k)
        print(f"Recall@{k} two_tower        : {tt_recall:.6f}")

    rrf_specs = [
        ("rrf_pop_content", ["popularity", "content"]),
        ("rrf_pop_ials", ["popularity", "ials"]),
        ("rrf_pop_ials_content", ["popularity", "ials", "content"]),
    ]
    for label, channels in rrf_specs:
        recs: dict[str, list[str]] = {}
        for user_id in user_ids:
            result = service.recommend_for_user(
                user_id, k=k, use_hybrid=True, hybrid_channels=channels
            )
            recs[user_id] = [x["item_id"] for x in result.items]
        print(f"Recall@{k} {label:<22}: {mean_recall_at_k(recs, truth, k):.6f}")

    rank_dir = resolve_path(cfg.get("ranking", {}).get("artifact_dir", "data/processed/ranker"))
    if service.ranker is None or not (rank_dir / "model.txt").exists():
        print(f"Recall@{k} ranker_lgbm          : SKIP (run python -m scripts.train_ranker)")
    else:
        rank_recs: dict[str, list[str]] = {}
        for user_id in user_ids:
            result = service.recommend_for_user(
                user_id, k=k, use_content=False, use_ranker=True
            )
            rank_recs[user_id] = [x["item_id"] for x in result.items]
        print(f"Recall@{k} ranker_lgbm          : {mean_recall_at_k(rank_recs, truth, k):.6f}")
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
