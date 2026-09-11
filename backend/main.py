"""Why This Product API — health check and future recommend/explain routes."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="Why This Product",
    description="E-commerce RecSys + RAG shopping assistant",
    version="0.1.0-mvp-scaffold",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "phase": "scaffold"}
