"""오프라인 평가: pop vs MMR 람다 + cold 세그먼트."""

from __future__ import annotations

import sys

import pandas as pd

from backend.services.recommend import RecommendService
from ml.config import load_mvp_config, resolve_path
from ml.eval.metrics import (
    coverage_at_k,
    mean_ild_at_k,
    mean_ndcg_at_k,
    mean_recall_at_k,
    mean_unique_categories_at_k,
)
from ml.rerank.mmr import mmr_rerank

POOL_K = 200
FINAL_K = 10
RELEVANT_MIN = 5.0
MMR_LAMBDAS = (0.3, 0.5, 0.7)
MAX_USERS = 200


def _ids(hits: list[tuple[str, float]]) -> list[str]:
    return [item_id for item_id, _ in hits]


def _relevant_maps(
    frame: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], dict[str, set[str]]]:
    rel: dict[str, dict[str, float]] = {}
    for r in frame.itertuples(index=False):
        rel.setdefault(str(r.user_id), {})[str(r.item_id)] = float(r.rating)
    truth = {
        user_id: {iid for iid, rating in items.items() if rating >= RELEVANT_MIN}
        for user_id, items in rel.items()
    }
    truth = {u: s for u, s in truth.items() if s}
    rel = {u: rel[u] for u in truth}
    return rel, truth


def _cap(truth: dict[str, set[str]], rel: dict[str, dict[str, float]], n: int):
    user_ids = list(truth.keys())[:n]
    return {u: truth[u] for u in user_ids}, {u: rel[u] for u in user_ids}, user_ids


def _print_row(
    label: str,
    recs: dict[str, list[str]],
    truth: dict[str, set[str]],
    rel: dict[str, dict[str, float]],
    catalog_n: int,
    item_category: dict[str, str | None],
    vector_of,
) -> dict[str, float]:
    recall = mean_recall_at_k(recs, truth, FINAL_K)
    ndcg = mean_ndcg_at_k(recs, rel, FINAL_K)
    cov = coverage_at_k(recs, catalog_n, FINAL_K)
    ild = mean_ild_at_k(recs, vector_of, FINAL_K)
    uniq = mean_unique_categories_at_k(recs, item_category, FINAL_K)
    print(
        f"{label:<22} n={len(truth):<4}  "
        f"Recall={recall:.6f}  NDCG={ndcg:.6f}  "
        f"Cov={cov:.6f}  ILD={ild:.6f}  uniqCat={uniq:.3f}"
    )
    return {"recall": recall, "ndcg": ndcg, "cov": cov, "ild": ild, "uniq": uniq}


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
    catalog_n = len(service.items_meta)
    item_category = {
        iid: (meta.get("category") if isinstance(meta.get("category"), str) else None)
        for iid, meta in service.items_meta.items()
    }
    vector_of = service.similarity.vec
    print(f"[eval] catalog items: {catalog_n:,}  train items: {len(service.train_item_ids):,}")

    valid = pd.read_parquet(valid_path, columns=["user_id", "item_id", "rating"])
    valid = valid.groupby(["user_id", "item_id"], as_index=False)["rating"].max()
    train_users = set(service.train_by_user.keys())

    warm_df = valid[valid["user_id"].isin(train_users)]
    cold_user_df = valid[~valid["user_id"].isin(train_users)]
    warm_rel, warm_truth = _relevant_maps(warm_df)
    warm_truth, warm_rel, warm_ids = _cap(warm_truth, warm_rel, MAX_USERS)
    print(f"[eval] warm valid users with rating>={RELEVANT_MIN:.0f}: {len(warm_truth)}")

    pop_recs: dict[str, list[str]] = {}
    cold_pool_recs: dict[str, list[str]] = {}
    mmr_recs: dict[float, dict[str, list[str]]] = {lam: {} for lam in MMR_LAMBDAS}
    for i, user_id in enumerate(warm_ids, start=1):
        history = service.train_by_user.get(user_id, [])
        exclude = set(history)
        pop_ids = _ids(service.popularity.recommend(k=POOL_K, exclude=exclude))
        pop_recs[user_id] = pop_ids[:FINAL_K]
        pool = service.candidate_pool(user_id)
        cold_pool_recs[user_id] = _ids(pool)[:FINAL_K]
        for lam in MMR_LAMBDAS:
            mmr_recs[lam][user_id] = _ids(mmr_rerank(pool, FINAL_K, lam, service.similarity))
        if i % 50 == 0:
            print(f"  ... warm {i}/{len(warm_ids)}")

    print(f"--- warm relevant>={RELEVANT_MIN:.0f} @{FINAL_K} ---")
    scores = {
        "pop": _print_row("pop", pop_recs, warm_truth, warm_rel, catalog_n, item_category, vector_of),
        "pop+cold": _print_row(
            "pop+cold", cold_pool_recs, warm_truth, warm_rel, catalog_n, item_category, vector_of
        ),
    }
    for lam in MMR_LAMBDAS:
        scores[f"mmr_{lam}"] = _print_row(
            f"mmr_div={lam}",
            mmr_recs[lam],
            warm_truth,
            warm_rel,
            catalog_n,
            item_category,
            vector_of,
        )

    pop_r = scores["pop"]["recall"]
    pop_ild = scores["pop"]["ild"]
    pop_uniq = scores["pop"]["uniq"]
    eligible: list[tuple[float, float]] = []
    for lam in MMR_LAMBDAS:
        row = scores[f"mmr_{lam}"]
        ild_up = row["ild"] > pop_ild or row["uniq"] > pop_uniq
        recall_ok = (pop_r - row["recall"]) < 0.04 and row["recall"] >= 0.15
        print(
            f"[rule] mmr_div={lam}: ild_or_cat_up={ild_up} recall_ok={recall_ok} "
            f"(drop={pop_r - row['recall']:.4f})"
        )
        if ild_up and recall_ok:
            eligible.append((lam, row["ild"]))
    if not eligible:
        print("[serve] MMR off (rule not met)")
        chosen = None
    else:
        chosen = max(eligible, key=lambda t: t[1])[0]
        print(f"[serve] MMR on lambda_diversity={chosen}")

    cold_rel, cold_truth = _relevant_maps(cold_user_df)
    cold_truth, cold_rel, cold_ids = _cap(cold_truth, cold_rel, MAX_USERS)
    print(f"--- cold-user relevant>={RELEVANT_MIN:.0f} @{FINAL_K} n={len(cold_truth)} ---")
    if cold_ids:
        c_pop: dict[str, list[str]] = {}
        c_mmr: dict[str, list[str]] = {}
        default_lam = chosen if chosen is not None else 0.7
        for user_id in cold_ids:
            pool = service.candidate_pool(user_id)
            c_pop[user_id] = _ids(pool)[:FINAL_K]
            c_mmr[user_id] = _ids(mmr_rerank(pool, FINAL_K, default_lam, service.similarity))
        _print_row("cold-user pop", c_pop, cold_truth, cold_rel, catalog_n, item_category, vector_of)
        _print_row(
            f"cold-user mmr={default_lam}",
            c_mmr,
            cold_truth,
            cold_rel,
            catalog_n,
            item_category,
            vector_of,
        )
    else:
        print("cold-user: none")

    item_truth = {
        u: {iid for iid in items if iid not in service.train_item_ids}
        for u, items in warm_truth.items()
    }
    item_truth = {u: s for u, s in item_truth.items() if s}
    item_rel = {u: {iid: warm_rel[u][iid] for iid in s} for u, s in item_truth.items()}
    item_ids = list(item_truth.keys())[:MAX_USERS]
    item_truth = {u: item_truth[u] for u in item_ids}
    item_rel = {u: item_rel[u] for u in item_ids}
    print(f"--- cold-item GT (warm users) @{FINAL_K} n={len(item_truth)} ---")
    if item_ids:
        _print_row(
            "cold-item pop",
            {u: pop_recs[u] for u in item_ids},
            item_truth,
            item_rel,
            catalog_n,
            item_category,
            vector_of,
        )
        _print_row(
            "cold-item pop+cold",
            {u: cold_pool_recs[u] for u in item_ids},
            item_truth,
            item_rel,
            catalog_n,
            item_category,
            vector_of,
        )
        for lam in MMR_LAMBDAS:
            _print_row(
                f"cold-item mmr={lam}",
                {u: mmr_recs[lam][u] for u in item_ids},
                item_truth,
                item_rel,
                catalog_n,
                item_category,
                vector_of,
            )
    else:
        print("cold-item GT: none")
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
