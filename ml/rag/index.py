"""설명용 청크 FAISS. retrieve 인덱스와 분리. train 메타·리뷰만."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ml.embeddings.encoder import Embedder
from ml.vectorstore.faiss_store import FaissItemIndex


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    item_id: str
    source: str
    text: str


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    cut = text[:n]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut or text[:n]


def build_chunks(
    items: pd.DataFrame,
    train: pd.DataFrame,
    max_reviews: int,
    chunk_chars: int,
) -> list[RagChunk]:
    reviews = train.dropna(subset=["review_text"]).copy()
    reviews["item_id"] = reviews["item_id"].astype(str)
    reviews["review_text"] = reviews["review_text"].astype(str)
    reviews = reviews.sort_values("timestamp")
    by_item = (
        reviews.groupby("item_id", sort=False)["review_text"]
        .apply(lambda s: [t for t in s.tolist() if t.strip()][-max_reviews:])
        .to_dict()
    )
    chunks: list[RagChunk] = []
    n = 0
    for r in items.itertuples(index=False):
        item_id = str(r.item_id)
        parts = [str(getattr(r, name) or "").strip() for name in ("title", "brand", "description")]
        meta = _clip(". ".join(p for p in parts if p) or item_id, chunk_chars)
        chunks.append(RagChunk(str(n), item_id, "meta", meta))
        n += 1
        for text in by_item.get(item_id, []):
            clipped = _clip(text, chunk_chars)
            if not clipped:
                continue
            chunks.append(RagChunk(str(n), item_id, "review", clipped))
            n += 1
    return chunks


def save_chunks(chunks: list[RagChunk], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "chunks.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")


def load_chunks(directory: Path) -> list[RagChunk]:
    path = directory / "chunks.jsonl"
    out: list[RagChunk] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                RagChunk(
                    chunk_id=str(obj["chunk_id"]),
                    item_id=str(obj["item_id"]),
                    source=str(obj["source"]),
                    text=str(obj["text"]),
                )
            )
    return out


def build_rag_index(
    items_path: Path,
    train_path: Path,
    model_name: str,
    index_dir: Path,
    max_reviews: int,
    chunk_chars: int,
) -> FaissItemIndex:
    items = pd.read_parquet(items_path, columns=["item_id", "title", "brand", "description"])
    train = pd.read_parquet(train_path, columns=["item_id", "review_text", "timestamp"])
    chunks = build_chunks(items, train, max_reviews=max_reviews, chunk_chars=chunk_chars)
    if not chunks:
        raise ValueError("no RAG chunks")
    embedder = Embedder(model_name)
    embeddings = embedder.encode([c.text for c in chunks], batch_size=256)
    print(f"[build] chunks={len(chunks):,}")
    index = FaissItemIndex.build(embeddings, [c.chunk_id for c in chunks])
    index.save(index_dir)
    save_chunks(chunks, index_dir)
    return index
