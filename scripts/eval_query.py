"""Query retrieve vs lexical gold: recall, misses, false positives."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from backend.services.recommend import RecommendService
from ml.config import load_mvp_config, resolve_path
from ml.eval.query_eval import evaluate
from ml.eval.query_label import GOLD_JSONL, blob_for_item
from scripts.build_query_gold import load_gold

EVAL_K = 10
DUMP_PATH = "data/eval/query_eval.json"
MISS_CSV = "data/eval/query_misses.csv"
FP_CSV = "data/eval/query_false_positives.csv"


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["query", "intent", "item_id", "kind"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = load_mvp_config()
    gold_path = resolve_path(GOLD_JSONL)
    if not gold_path.exists():
        raise FileNotFoundError(f"Missing {gold_path}. Run python -m scripts.build_query_gold")
    gold_rows = load_gold(gold_path)
    items_path = resolve_path(cfg["data"]["processed_dir"]) / "items.parquet"
    items = pd.read_parquet(items_path, columns=["item_id", "title", "brand", "doc_text"])
    blobs: dict[str, str] = {}
    for r in items.itertuples(index=False):
        blobs[str(r.item_id)] = blob_for_item(
            None if pd.isna(r.title) else str(r.title),
            None if pd.isna(r.brand) else str(r.brand),
            None if pd.isna(r.doc_text) else str(r.doc_text),
        )
    rec = RecommendService.from_processed()
    recs_by_query: dict[str, list[str]] = {}
    for row in gold_rows:
        query = str(row["query"])
        result = rec.recommend_from_query(query, k=EVAL_K, use_mmr=True)
        recs_by_query[query] = [it["item_id"] for it in result.items]
        print(f"[query] {query!r} n={len(recs_by_query[query])} strategy={result.strategy}")
    summary, misses, fps = evaluate(gold_rows, recs_by_query, blobs, k=EVAL_K)
    dump = resolve_path(DUMP_PATH)
    dump.parent.mkdir(parents=True, exist_ok=True)
    dump.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(resolve_path(MISS_CSV), misses)
    _write_csv(resolve_path(FP_CSV), fps)
    print("--- query eval ---")
    print(
        f"R@{EVAL_K}={summary['mean_recall_at_k']} "
        f"P@{EVAL_K}={summary['mean_precision_at_k']} "
        f"NDCG@{EVAL_K}={summary['mean_ndcg_at_k']}"
    )
    print(f"miss={summary['n_miss']} fp={summary['n_fp']} queries={summary['n_queries']}")
    for intent, row in (summary.get("by_intent") or {}).items():
        print(
            f"  intent={intent} n={row['n_queries']} "
            f"R={row['mean_recall_at_k']} P={row['mean_precision_at_k']} "
            f"NDCG={row['mean_ndcg_at_k']} miss={row['n_miss']} fp={row['n_fp']}"
        )
    print(f"wrote {dump}")
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
