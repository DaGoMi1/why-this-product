from __future__ import annotations

import sys

from ml.config import load_mvp_config, resolve_path
from ml.retrieval.content_faiss import build_content_index


def main() -> None:
    cfg = load_mvp_config()
    processed = resolve_path(cfg["data"]["processed_dir"])
    items_path = processed / "items.parquet"
    if not items_path.exists():
        raise FileNotFoundError(
            f"Missing {items_path}. Run: python -m scripts.prepare_splits"
        )
    index_dir = resolve_path(cfg["rag"]["faiss_index_dir"])
    model_name = cfg["rag"]["embedding_model"]
    print(f"[build] model={model_name}")
    print(f"[build] items={items_path}")
    print(f"[build] out={index_dir}")
    index = build_content_index(items_path, model_name, index_dir)
    print(f"[done] ntotal={index.index.ntotal:,} dim={index.index.d}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
