"""Build lexical query gold from items.parquet. Not human labels."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

from ml.config import load_mvp_config, resolve_path
from ml.eval.query_label import (
    GOLD_CSV,
    GOLD_JSONL,
    MAX_POSITIVES,
    QUERY_SPECS,
    assert_specs_match_eval_queries,
    blob_for_item,
    matched_tokens,
)


def _empty(val: object) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    return str(val).strip() == ""


def _str(val: object) -> str:
    if _empty(val):
        return ""
    return str(val).strip()


def build_gold(items: pd.DataFrame, max_positives: int = MAX_POSITIVES) -> list[dict]:
    assert_specs_match_eval_queries()
    rows: list[dict] = []
    for spec in QUERY_SPECS:
        scored: list[tuple[int, str, str, list[str]]] = []
        for r in items.itertuples(index=False):
            item_id = str(r.item_id)
            title = _str(getattr(r, "title", ""))
            brand = _str(getattr(r, "brand", ""))
            doc = _str(getattr(r, "doc_text", ""))
            blob = blob_for_item(title, brand, doc)
            tokens = matched_tokens(blob, spec)
            if tokens is None:
                continue
            scored.append((len(title or item_id), item_id, title or item_id, tokens))
        scored.sort(key=lambda x: (x[0], x[1]))
        kept = scored[:max_positives]
        rows.append(
            {
                "query": spec.query,
                "intent": spec.intent,
                "must": [list(g) for g in spec.must],
                "n_match": len(scored),
                "n_kept": len(kept),
                "item_ids": [item_id for _, item_id, _, _ in kept],
                "items": [
                    {"item_id": item_id, "title": title, "tokens": tokens}
                    for _, item_id, title, tokens in kept
                ],
            }
        )
    return rows


def write_gold(rows: list[dict], jsonl_path: Path, csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "intent", "item_id", "title", "matched_tokens", "n_kept"],
        )
        writer.writeheader()
        for row in rows:
            n_kept = int(row["n_kept"])
            for item in row["items"]:
                writer.writerow(
                    {
                        "query": row["query"],
                        "intent": row["intent"],
                        "item_id": item["item_id"],
                        "title": item["title"],
                        "matched_tokens": "|".join(item["tokens"]),
                        "n_kept": n_kept,
                    }
                )


def load_gold(jsonl_path: Path) -> list[dict]:
    out: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> None:
    cfg = load_mvp_config()
    items_path = resolve_path(cfg["data"]["processed_dir"]) / "items.parquet"
    if not items_path.exists():
        raise FileNotFoundError(f"Missing {items_path}. Run prepare_splits first.")
    items = pd.read_parquet(
        items_path, columns=["item_id", "title", "brand", "doc_text"]
    )
    rows = build_gold(items)
    jsonl_path = resolve_path(GOLD_JSONL)
    csv_path = resolve_path(GOLD_CSV)
    write_gold(rows, jsonl_path, csv_path)
    print(f"[gold] queries={len(rows)} max_positives={MAX_POSITIVES}")
    for row in rows:
        print(
            f"  {row['intent']:11} {row['query']!r:32} "
            f"match={row['n_match']:<5} kept={row['n_kept']}"
        )
    print(f"[done] {jsonl_path}")
    print(f"[done] {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
