"""후보 item_id로 필터한 설명 스니펫 검색."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from ml.embeddings.encoder import Embedder
from ml.rag.index import RagChunk, load_chunks
from ml.vectorstore.faiss_store import FaissItemIndex

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


def lexical_overlap(query: str, text: str) -> float:
    q = tokenize(query)
    if not q:
        return 0.0
    return len(q & tokenize(text)) / len(q)


def chunk_rank_score(query: str, chunk: RagChunk, dense_score: float) -> float:
    lex = lexical_overlap(query, chunk.text)
    score = float(dense_score) + lex
    if chunk.source == "review" and lex > 0:
        score += 0.25
    return score


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
        dense = {chunk_id: float(score) for chunk_id, score in hits}
        for iid in allowed:
            ranked = sorted(
                self._by_item[iid],
                key=lambda ch: chunk_rank_score(qtext, ch, dense.get(ch.chunk_id, 0.0)),
                reverse=True,
            )
            picked: list[RagChunk] = []
            for ch in ranked:
                if len(picked) >= per_item:
                    break
                if chunk_rank_score(qtext, ch, dense.get(ch.chunk_id, 0.0)) <= 0 and picked:
                    continue
                picked.append(ch)
            if not picked:
                review = next(
                    (
                        c
                        for c in self._by_item[iid]
                        if c.source == "review" and lexical_overlap(qtext, c.text) > 0
                    ),
                    None,
                )
                meta = next((c for c in self._by_item[iid] if c.source == "meta"), None)
                picked = [review or meta or self._by_item[iid][0]]
            out[iid] = picked[:per_item]
        return out
