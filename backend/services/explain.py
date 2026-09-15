"""후보 ASIN 설명: RAG 스니펫 + OpenAI."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from ml.config import load_mvp_config, resolve_path
from ml.rag.llm import ExplainResult, explain_items
from ml.rag.retriever import RagSnippetRetriever


class ExplainService:
    def __init__(self, retriever: RagSnippetRetriever, cfg: dict) -> None:
        self.retriever = retriever
        self.cfg = cfg
        items_path = resolve_path(cfg["data"]["processed_dir"]) / "items.parquet"
        titles: dict[str, str | None] = {}
        if items_path.exists():
            import pandas as pd

            items = pd.read_parquet(items_path, columns=["item_id", "title"])
            for r in items.itertuples(index=False):
                title = r.title
                if title is not None and isinstance(title, float) and pd.isna(title):
                    title = None
                titles[str(r.item_id)] = str(title) if title else None
        self.titles = titles

    @classmethod
    def from_processed(cls) -> ExplainService:
        cfg = load_mvp_config()
        rag = cfg.get("rag", {})
        index_dir = resolve_path(rag.get("rag_index_dir", "data/processed/rag_index"))
        if not (Path(index_dir) / "index.faiss").exists():
            raise FileNotFoundError(
                f"Missing RAG index at {index_dir}. Run python -m scripts.build_rag_index"
            )
        retriever = RagSnippetRetriever.load(index_dir, str(rag.get("embedding_model")))
        return cls(retriever=retriever, cfg=cfg)

    def explain(
        self,
        item_ids: list[str],
        query: str | None = None,
        select_k: int = 0,
    ) -> ExplainResult:
        known = [iid for iid in item_ids if iid in self.retriever._by_item]
        if not known:
            raise ValueError("item_ids not in RAG index")
        rag = self.cfg.get("rag", {})
        per_item = int(rag.get("snippets_per_item", 3))
        t0 = time.perf_counter()
        snippets = self.retriever.snippets_for_items(known, query, per_item=per_item)
        result = explain_items(
            known,
            self.titles,
            snippets,
            query,
            model=str(rag.get("llm_model", "gpt-4o-mini")),
            select_k=select_k,
        )
        rag_ms = round((time.perf_counter() - t0) * 1000, 1)
        result.timings_ms = {"rag": rag_ms, "total": rag_ms}
        return result


@lru_cache(maxsize=1)
def get_explain_service() -> ExplainService:
    return ExplainService.from_processed()
