import importlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPORT_AGENT_API_KEYS", "test-key")
    monkeypatch.setenv("SUPPORT_AGENT_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("SUPPORT_AGENT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("SUPPORT_AGENT_RATE_LIMIT_REQUESTS", "3")
    monkeypatch.setenv("SUPPORT_AGENT_RATE_LIMIT_WINDOW_SECONDS", "60")

    import support_agent.api as api_module

    importlib.reload(api_module)
    # Skip real ingest during lifespan; agent methods are mocked per-test.
    monkeypatch.setattr(api_module.agent, "is_ready", lambda: True)
    with TestClient(api_module.app) as test_client:
        yield test_client, api_module


def _headers():
    return {"X-API-Key": "test-key"}


def test_health_returns_ok(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_answer_without_api_key_returns_401(client):
    test_client, _ = client
    response = test_client.post("/answer", json={"question": "hi", "user_id": "user-1"})
    assert response.status_code == 401


def test_answer_with_invalid_api_key_returns_403(client):
    test_client, _ = client
    response = test_client.post(
        "/answer",
        json={"question": "hi", "user_id": "user-1"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 403


def test_answer_empty_question_returns_400(client):
    test_client, _ = client
    response = test_client.post(
        "/answer",
        json={"question": "   ", "user_id": "user-1"},
        headers=_headers(),
    )
    assert response.status_code == 400


def test_answer_valid_request_returns_200(client):
    test_client, api_module = client
    from support_agent.agent import AnswerResult

    fake = AnswerResult(
        answer="refunds take 3-5 business days",
        sources=["rag_knowledge_base.md#1"],
        confidence=0.82,
        escalated=False,
        tool_name=None,
        session_id="sess-1",
        trace_id="trace-abc",
        guardrail_reason=None,
    )
    with patch.object(api_module.agent, "answer_with_metadata", return_value=fake):
        response = test_client.post(
            "/answer",
            json={"question": "how long for a refund", "user_id": "user-1"},
            headers=_headers(),
        )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "refunds take 3-5 business days"
    assert body["confidence"] == 0.82
    assert body["session_id"] == "sess-1"


def test_rate_limit_returns_429(client):
    test_client, api_module = client
    from support_agent.agent import AnswerResult

    fake = AnswerResult(
        answer="ok",
        sources=[],
        confidence=1.0,
        escalated=False,
        session_id="s",
        trace_id="t",
    )
    with patch.object(api_module.agent, "answer_with_metadata", return_value=fake):
        # Limit is 3; 4th request should be rejected.
        for _ in range(3):
            ok = test_client.post(
                "/answer",
                json={"question": "hi", "user_id": "user-1"},
                headers=_headers(),
            )
            assert ok.status_code == 200
        rejected = test_client.post(
            "/answer",
            json={"question": "hi", "user_id": "user-1"},
            headers=_headers(),
        )
    assert rejected.status_code == 429
    assert "Retry-After" in rejected.headers


def test_create_ticket_returns_ticket_id(client):
    test_client, _ = client
    response = test_client.post(
        "/tickets",
        json={"user_id": "user-1", "message": "my order is broken"},
        headers=_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["ticket_id"], int)
    assert body["status"] == "open"


def test_get_ticket_not_found_returns_404(client):
    test_client, _ = client
    response = test_client.get("/tickets/999999", headers=_headers())
    assert response.status_code == 404
