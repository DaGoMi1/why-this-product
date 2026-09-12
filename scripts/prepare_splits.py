"""아마존 리뷰 2018 All_Beauty를 정규화하고 시간 순서 train/valid/test 분할을 생성"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from ml.config import load_mvp_config, resolve_path


def _parse_json_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Some amazon_v2 meta lines use invalid JSON literals (NaN / Infinity)
            line = line.replace(": NaN", ": null").replace(": Infinity", ": null")
            line = line.replace(":-Infinity", ": null")
            yield json.loads(line)


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, list):
        parts = [_as_text(v) for v in value]
        parts = [p for p in parts if p]
        return " ".join(parts) if parts else None
    text = str(value).strip()
    return text or None


def _parse_price(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"\d+(?:\.\d+)?", str(value))
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def load_interactions(raw_reviews: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for obj in _parse_json_gz(raw_reviews):
        user_id = obj.get("reviewerID")
        item_id = obj.get("asin")
        ts = obj.get("unixReviewTime")
        if not user_id or not item_id or ts is None:
            continue
        rows.append(
            {
                "user_id": str(user_id),
                "item_id": str(item_id),
                "rating": float(obj.get("overall", 0.0)),
                "timestamp": int(ts),
                "review_text": _as_text(obj.get("reviewText")),
            }
        )
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["user_id", "item_id", "timestamp"], keep="first")
    return df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def load_items(raw_meta: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for obj in _parse_json_gz(raw_meta):
        item_id = obj.get("asin")
        if not item_id:
            continue
        title = _as_text(obj.get("title")) or ""
        brand = _as_text(obj.get("brand"))
        category = _as_text(obj.get("category") or obj.get("main_cat"))
        description = _as_text(obj.get("description"))
        doc_parts = [p for p in (title, brand, description) if p]
        rows.append(
            {
                "item_id": str(item_id),
                "title": title,
                "brand": brand,
                "category": category,
                "description": description,
                "price": _parse_price(obj.get("price")),
                "doc_text": ". ".join(doc_parts) if doc_parts else str(item_id),
            }
        )
    items = pd.DataFrame(rows)
    items = items.drop_duplicates(subset=["item_id"], keep="first").reset_index(drop=True)
    return items


def temporal_split(
    interactions: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(interactions)
    if n < 10:
        raise ValueError(f"too few interactions to split: {n}")
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    train = interactions.iloc[:train_end].copy()
    valid = interactions.iloc[train_end:valid_end].copy()
    test = interactions.iloc[valid_end:].copy()
    return train, valid, test


def main() -> None:
    cfg = load_mvp_config()
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    out_dir = resolve_path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    reviews_path = raw_dir / "All_Beauty.json.gz"
    meta_path = raw_dir / "meta_All_Beauty.json.gz"
    if not reviews_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing raw files. Run: python -m scripts.download_data\n"
            f"Expected: {reviews_path} and {meta_path}"
        )

    print("[load] reviews...")
    interactions = load_interactions(reviews_path)
    print(f"  interactions: {len(interactions):,}")

    print("[load] metadata...")
    items = load_items(meta_path)
    print(f"  items: {len(items):,}")

    # Keep interactions whose items exist in meta when possible; still keep orphans
    # for CF stats but content retrieve needs meta — filter to known items for MVP.
    known = set(items["item_id"])
    before = len(interactions)
    interactions = interactions[interactions["item_id"].isin(known)].reset_index(drop=True)
    print(f"  interactions with meta: {len(interactions):,} (dropped {before - len(interactions):,})")

    train_ratio = float(cfg["data"]["train_ratio"])
    valid_ratio = float(cfg["data"]["valid_ratio"])
    train, valid, test = temporal_split(interactions, train_ratio, valid_ratio)
    print(
        f"[split] train={len(train):,} valid={len(valid):,} test={len(test):,} "
        f"(ratios {train_ratio}/{valid_ratio}/{cfg['data']['test_ratio']})"
    )

    train.to_parquet(out_dir / "interactions_train.parquet", index=False)
    valid.to_parquet(out_dir / "interactions_valid.parquet", index=False)
    test.to_parquet(out_dir / "interactions_test.parquet", index=False)
    items.to_parquet(out_dir / "items.parquet", index=False)
    print(f"[done] wrote parquet under {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
