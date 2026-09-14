from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class RecommendRequest(BaseModel):
    user_id: str | None = None
    query: str | None = None
    k: int = Field(default=10, ge=1, le=50)
    use_content: bool = False
    use_ials: bool = False
    use_hybrid: bool = False
    use_ranker: bool = False
    use_mmr: bool = True

    @model_validator(mode="after")
    def require_user_or_query(self) -> RecommendRequest:
        user_id = self.user_id.strip() if self.user_id else None
        query = self.query.strip() if self.query else None
        if not user_id and not query:
            raise ValueError("user_id 또는 query 중 하나는 필요합니다")
        self.user_id = user_id
        self.query = query
        return self


class RecommendItem(BaseModel):
    item_id: str
    score: float
    title: str | None = None
    brand: str | None = None
    category: str | None = None


class RecommendResponse(BaseModel):
    items: list[RecommendItem]
    strategy: str
    k: int
