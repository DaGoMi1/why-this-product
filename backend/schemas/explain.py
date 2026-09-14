from __future__ import annotations

from pydantic import BaseModel, Field


class ExplainRequest(BaseModel):
    item_ids: list[str] = Field(min_length=1, max_length=10)
    query: str | None = None
    select_k: int = Field(default=0, ge=0, le=10)


class ExplainSnippet(BaseModel):
    text: str
    source: str


class ExplainItem(BaseModel):
    item_id: str
    reason: str
    snippets: list[ExplainSnippet]


class ExplainResponse(BaseModel):
    items: list[ExplainItem]
    selected_ids: list[str]
    model: str
    prompt_tokens: int
    completion_tokens: int
