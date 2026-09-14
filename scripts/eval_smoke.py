"""오프라인 평가: rating>=5 통일. retrieve 단독 / 채널×랭커 / 카탈로그 부스팅 단독."""

from __future__ import annotations

import sys

import pandas as pd

from backend.services.recommend import RecommendService
from ml.config import load_mvp_config, resolve_path
from ml.eval.metrics import mean_ndcg_at_k, mean_recall_at_k
from ml.ranking import RANKERS, Ranker, load_ranker
from ml.retrieval.ials import IALSRetriever
from ml.retrieval.merge import rrf_fuse
from ml.retrieval.two_tower import TwoTowerRetriever

POOL_K = 200
FINAL_K = 10
RELEVANT_MIN = 5.0
RANK_CHANNELS = [
    "popularity",
    "ials_rating_ge_5",
    "content_per_seed",
    "rrf_pop_content",
    "rrf_pop_ials",
    "rrf_pop_ials_content",
]


def _ids(hits: list[tuple[str, float]]) -> list[str]:
    return [item_id for item_id, _ in hits]


def _print_pair(kind: str, k: int, label: str, recall: float, ndcg: float) -> None:
    print(f"{kind}@{k} {label:<28}: Recall={recall:.6f}  NDCG={ndcg:.6f}")


def main() -> None:
    cfg = load_mvp_config()
    processed = resolve_path(cfg["data"]["processed_dir"])
    valid_path = processed / "interactions_valid.parquet"
    train_path = processed / "interactions_train.parquet"
    items_path = processed / "items.parquet"
    index_dir = resolve_path(cfg["rag"]["faiss_index_dir"])

    for path in (train_path, valid_path, items_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run prepare_splits first.")
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"Missing FAISS index at {index_dir}. Run build_faiss first.")

    print("[load] recommend service...")
    service = RecommendService.from_processed()
    catalog_ids = list(service.items_meta.keys())
    print(f"[eval] catalog items: {len(catalog_ids):,}")

    valid = pd.read_parquet(valid_path, columns=["user_id", "item_id", "rating"])
    train_users = set(service.train_by_user.keys())
    valid = valid[valid["user_id"].isin(train_users)]
    valid = valid.groupby(["user_id", "item_id"], as_index=False)["rating"].max()

    rel: dict[str, dict[str, float]] = {}
    for r in valid.itertuples(index=False):
        rel.setdefault(str(r.user_id), {})[str(r.item_id)] = float(r.rating)
    truth = {
        user_id: {iid for iid, rating in items.items() if rating >= RELEVANT_MIN}
        for user_id, items in rel.items()
    }
    truth = {u: s for u, s in truth.items() if s}
    max_users = 200
    user_ids = list(truth.keys())[:max_users]
    truth = {u: truth[u] for u in user_ids}
    rel = {u: rel[u] for u in user_ids}
    print(f"[eval] warm valid users with rating>={RELEVANT_MIN:.0f}: {len(truth)}")

    rrf_k = int(cfg["retrieval"].get("rrf_k", 60))
    ials_cfg = cfg.get("ials", {})
    ials_root = resolve_path(ials_cfg.get("artifact_dir", "data/processed/ials"))
    ials_ge5 = None
    ials_ge5_dir = ials_root / "rating_ge_5"
    if (ials_ge5_dir / "user_factors.npy").exists():
        ials_ge5 = IALSRetriever.load(ials_ge5_dir)

    tt_cfg = cfg.get("two_tower", {})
    tt_dir = resolve_path(tt_cfg.get("artifact_dir", "data/processed/two_tower"))
    two_tower = None
    if (tt_dir / "item_factors.npy").exists():
        two_tower = TwoTowerRetriever.load(tt_dir)

    pools: dict[str, dict[str, list[str]]] = {name: {} for name in [*RANK_CHANNELS, "two_tower"]}

    for i, user_id in enumerate(user_ids, start=1):
        history = service.train_by_user.get(user_id, [])
        exclude = set(history)
        pop_hits = service.popularity.recommend(k=POOL_K, exclude=exclude)
        pools["popularity"][user_id] = _ids(pop_hits)

        ials_hits: list[tuple[str, float]] = []
        if ials_ge5 is not None and ials_ge5.has_user(user_id):
            ials_hits = ials_ge5.recommend(user_id, k=POOL_K, exclude=exclude)
        pools["ials_rating_ge_5"][user_id] = _ids(ials_hits)

        content_hits: list[tuple[str, float]] = []
        if service.content is not None and history:
            content_hits = service.content.recommend_from_item_ids(
                history[-5:], k=POOL_K, exclude=exclude, mode="per_seed"
            )
        pools["content_per_seed"][user_id] = _ids(content_hits)

        pools["rrf_pop_content"][user_id] = _ids(
            rrf_fuse([pop_hits, content_hits], rrf_k=rrf_k, merge_k=POOL_K)
        )
        pools["rrf_pop_ials"][user_id] = _ids(
            rrf_fuse([pop_hits, ials_hits], rrf_k=rrf_k, merge_k=POOL_K)
        )
        pools["rrf_pop_ials_content"][user_id] = _ids(
            rrf_fuse([pop_hits, ials_hits, content_hits], rrf_k=rrf_k, merge_k=POOL_K)
        )

        tt_hits: list[tuple[str, float]] = []
        if two_tower is not None:
            tt_hits = two_tower.recommend(history, k=POOL_K, exclude=exclude)
        pools["two_tower"][user_id] = _ids(tt_hits)

        if i % 50 == 0:
            print(f"  ... retrieve {i}/{len(user_ids)}")

    print(f"--- retrieve relevant>={RELEVANT_MIN:.0f} Recall@{POOL_K} ---")
    retrieve_scores: dict[str, float] = {}
    for label, recs in pools.items():
        if label == "ials_rating_ge_5" and ials_ge5 is None:
            print(f"Recall@{POOL_K} {label:<28}: SKIP (run python -m scripts.train_ials)")
            continue
        if label == "two_tower" and two_tower is None:
            print(f"Recall@{POOL_K} {label:<28}: SKIP (run python -m scripts.train_two_tower)")
            continue
        score = mean_recall_at_k(recs, truth, POOL_K)
        retrieve_scores[label] = score
        print(f"Recall@{POOL_K} {label:<28}: {score:.6f}")

    print(f"--- retrieve-only relevant>={RELEVANT_MIN:.0f} @{FINAL_K} ---")
    for label in RANK_CHANNELS:
        recs10 = {u: recs[:FINAL_K] for u, recs in pools[label].items()}
        _print_pair(
            "single",
            FINAL_K,
            label,
            mean_recall_at_k(recs10, truth, FINAL_K),
            mean_ndcg_at_k(recs10, rel, FINAL_K),
        )

    rank_dir = resolve_path(cfg.get("ranking", {}).get("artifact_dir", "data/processed/ranker"))
    loaded_rankers: list[tuple[str, Ranker]] = []
    for model_name, cls in RANKERS.items():
        loaded = load_ranker(rank_dir, model_name)
        if loaded is None:
            print(f"[ranker] {cls.name}: SKIP (run python -m scripts.train_ranker)")
            continue
        loaded_rankers.append((cls.name, loaded))

    print(f"--- retrieve x ranker relevant>={RELEVANT_MIN:.0f} @{FINAL_K} ---")
    funnel_recs: dict[tuple[str, str], dict[str, list[str]]] = {
        (ch, rname): {} for ch in RANK_CHANNELS for rname, _ in loaded_rankers
    }
    for ch in RANK_CHANNELS:
        for user_id in user_ids:
            cand = pools[ch].get(user_id, [])
            if not cand or service.features is None:
                for rname, _ in loaded_rankers:
                    funnel_recs[(ch, rname)][user_id] = []
                continue
            history = service.train_by_user.get(user_id, [])
            feat = service.features.matrix(user_id, history, cand)
            for rname, ranker in loaded_rankers:
                order = ranker.score(feat).argsort()[::-1][:FINAL_K]
                funnel_recs[(ch, rname)][user_id] = [cand[int(i)] for i in order]
    for ch in RANK_CHANNELS:
        for rname, _ in loaded_rankers:
            recs = funnel_recs[(ch, rname)]
            _print_pair(
                "funnel",
                FINAL_K,
                f"{ch}+{rname}",
                mean_recall_at_k(recs, truth, FINAL_K),
                mean_ndcg_at_k(recs, rel, FINAL_K),
            )

    print(f"--- boost-solo catalog relevant>={RELEVANT_MIN:.0f} @{FINAL_K} ---")
    solo_recs: dict[str, dict[str, list[str]]] = {rname: {} for rname, _ in loaded_rankers}
    for i, user_id in enumerate(user_ids, start=1):
        history = service.train_by_user.get(user_id, [])
        cand = [iid for iid in catalog_ids if iid not in set(history)]
        if not cand or service.features is None:
            for rname, _ in loaded_rankers:
                solo_recs[rname][user_id] = []
            continue
        feat = service.features.matrix(user_id, history, cand)
        for rname, ranker in loaded_rankers:
            order = ranker.score(feat).argsort()[::-1][:FINAL_K]
            solo_recs[rname][user_id] = [cand[int(i)] for i in order]
        if i % 10 == 0:
            print(f"  ... catalog {i}/{len(user_ids)}")
    for rname, _ in loaded_rankers:
        _print_pair(
            "solo",
            FINAL_K,
            f"catalog+{rname}",
            mean_recall_at_k(solo_recs[rname], truth, FINAL_K),
            mean_ndcg_at_k(solo_recs[rname], rel, FINAL_K),
        )
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
