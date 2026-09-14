"""후보 item_id로 필터한 설명 스니펫 검색."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ml.embeddings.encoder import Embedder
from ml.rag.index import RagChunk, load_chunks
from ml.vectorstore.faiss_store import FaissItemIndex


class RagSnippetRetriever:
    def __init__(self, index: FaissItemIndex, chunks: list[RagChunk], embedder: Embedder) -> None:
        if len(index.item_ids) != len(chunks):
            raise ValueError("index size and chunks length mismatch")
        self.index = index
        self.chunks = chunks
        self.embedder = embedder
        self._by_id = {c.chunk_id: c for c in chunks}
        self._by_item: dict[str, list[RagChunk]] = defaultdict(list)
        for c in chunks:
            self._by_item[c.item_id].append(c)

    @classmethod
    def load(cls, index_dir: Path, model_name: str) -> RagSnippetRetriever:
        return cls(
            index=FaissItemIndex.load(index_dir),
            chunks=load_chunks(index_dir),
            embedder=Embedder(model_name),
        )

    def snippets_for_items(
        self,
        item_ids: list[str],
        query: str | None,
        per_item: int = 3,
    ) -> dict[str, list[RagChunk]]:
        allowed = [iid for iid in item_ids if iid in self._by_item]
        out: dict[str, list[RagChunk]] = {iid: [] for iid in allowed}
        if not allowed:
            return out
        qtext = (query or "").strip() or "why this beauty product"
        top_k = min(len(self.chunks), max(80, len(allowed) * 8))
        vec = self.embedder.encode([qtext])
        hits = self.index.search(vec, top_k=top_k)[0]
        seen: dict[str, set[str]] = {iid: set() for iid in allowed}
        for chunk_id, _score in hits:
            ch = self._by_id.get(chunk_id)
            if ch is None or ch.item_id not in out:
                continue
            if ch.chunk_id in seen[ch.item_id]:
                continue
            if len(out[ch.item_id]) >= per_item:
                continue
            out[ch.item_id].append(ch)
            seen[ch.item_id].add(ch.chunk_id)
        for iid in allowed:
            if out[iid]:
                continue
            meta = next((c for c in self._by_item[iid] if c.source == "meta"), None)
            fallback = meta or self._by_item[iid][0]
            out[iid] = [fallback]
        return out
