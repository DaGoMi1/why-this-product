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
    assert body["phase"] == "phase1"


def test_recommend_validation() -> None:
    client = TestClient(app)
    response = client.post("/api/recommend", json={})
    assert response.status_code == 422
