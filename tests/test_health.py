"""스모크 테스트: FastAPI 앱 가져오기와 /health 응답"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["phase"] == "phase6"


def test_explain_validation() -> None:
    client = TestClient(app)
    response = client.post("/api/explain", json={})
    assert response.status_code == 422


def test_explain_missing_openai_key(monkeypatch) -> None:
    from backend.routers import explain as explain_mod
    from ml.rag.llm import MissingOpenAIKeyError

    class _Svc:
        def explain(self, *args, **kwargs):
            raise MissingOpenAIKeyError("OPENAI_API_KEY is required for /api/explain")

    monkeypatch.setattr(explain_mod, "get_explain_service", lambda: _Svc())
    client = TestClient(app)
    response = client.post("/api/explain", json={"item_ids": ["B000000000"]})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]
