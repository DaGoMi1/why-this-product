"""Why This Product API — health + recommend"""

from __future__ import annotations

from fastapi import FastAPI

from backend.routers.recommend import router as recommend_router

app = FastAPI(
    title="Why This Product",
    description="E-commerce RecSys + RAG 쇼핑 어시스턴트",
    version="0.1.0-phase1",
)
app.include_router(recommend_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "phase": "phase1"}
