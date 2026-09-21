"""Why This Product API — health + recommend"""

from __future__ import annotations

from fastapi import FastAPI

from backend.routers.explain import router as explain_router
from backend.routers.recommend import router as recommend_router

app = FastAPI(
    title="Why This Product",
    description="E-commerce RecSys + RAG 쇼핑 어시스턴트",
    version="0.1.0-phase7",
)
app.include_router(recommend_router)
app.include_router(explain_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "phase": "phase7"}
