"""설명 라우트. OpenAI 키 필수."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.explain import ExplainItem, ExplainRequest, ExplainResponse, ExplainSnippet
from backend.services.explain import get_explain_service
from ml.rag.llm import MissingOpenAIKeyError

router = APIRouter(prefix="/api", tags=["explain"])


@router.post("/explain", response_model=ExplainResponse)
def explain(body: ExplainRequest) -> ExplainResponse:
    try:
        service = get_explain_service()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        result = service.explain(body.item_ids, query=body.query, select_k=body.select_k)
    except MissingOpenAIKeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if body.select_k > 0 and not result.selected_ids:
        raise HTTPException(status_code=400, detail="no selected_ids remained inside candidates")
    return ExplainResponse(
        items=[
            ExplainItem(
                item_id=it.item_id,
                reason=it.reason,
                snippets=[ExplainSnippet(text=s["text"], source=s["source"]) for s in it.snippets],
            )
            for it in result.items
        ],
        selected_ids=result.selected_ids,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
