"""train 메타+리뷰 청크로 설명용 FAISS 인덱스를 만든다."""

from __future__ import annotations

import sys

from ml.config import load_mvp_config, resolve_path
from ml.rag.index import build_rag_index


def main() -> None:
    cfg = load_mvp_config()
    processed = resolve_path(cfg["data"]["processed_dir"])
    items_path = processed / "items.parquet"
    train_path = processed / "interactions_train.parquet"
    if not items_path.exists() or not train_path.exists():
        raise FileNotFoundError(f"Missing {items_path} or {train_path}. Run prepare_splits first.")
    rag = cfg.get("rag", {})
    index_dir = resolve_path(rag.get("rag_index_dir", "data/processed/rag_index"))
    model_name = str(rag.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    max_reviews = int(rag.get("max_reviews_per_item", 5))
    chunk_chars = int(rag.get("chunk_chars", 512))
    print(f"[build] model={model_name}")
    print(f"[build] out={index_dir} max_reviews={max_reviews} chunk_chars={chunk_chars}")
    index = build_rag_index(
        items_path,
        train_path,
        model_name,
        index_dir,
        max_reviews=max_reviews,
        chunk_chars=chunk_chars,
    )
    print(f"[done] ntotal={index.index.ntotal:,} dim={index.index.d}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
