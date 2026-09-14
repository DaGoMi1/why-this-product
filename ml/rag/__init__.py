"""RAG explanations and OpenAI LLM selection within candidates only (API key required)."""

from ml.rag.index import RagChunk, build_rag_index
from ml.rag.llm import ExplainResult, MissingOpenAIKeyError, explain_items
from ml.rag.retriever import RagSnippetRetriever

__all__ = [
    "RagChunk",
    "RagSnippetRetriever",
    "ExplainResult",
    "MissingOpenAIKeyError",
    "explain_items",
    "build_rag_index",
]
