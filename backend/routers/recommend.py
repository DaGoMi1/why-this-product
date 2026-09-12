"""추천 라우트"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.recommend import RecommendRequest, RecommendResponse
from backend.services.recommend import get_recommend_service

router = APIRouter(prefix="/api", tags=["recommend"])


@router.post("/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest) -> RecommendResponse:
    try:
        service = get_recommend_service()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if body.query:
        result = service.recommend_from_query(body.query, k=body.k)
    else:
        assert body.user_id is not None  # schema: user_id 또는 query 필수
        result = service.recommend_for_user(
            body.user_id,
            k=body.k,
            use_content=body.use_content,
            use_ials=body.use_ials,
            use_hybrid=body.use_hybrid,
        )
    return RecommendResponse(items=result.items, strategy=result.strategy, k=body.k)
