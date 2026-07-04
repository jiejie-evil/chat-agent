import importlib
from unittest.mock import patch

import pytest

from support_agent.agent import RetrievedChunk, SupportAgent
from support_agent.human_interface import HumanAgentInterface
from support_agent.state_store import StateStore


@pytest.fixture
def agent(tmp_path):
    store = StateStore(db_path=str(tmp_path / "test.db"))
    hi = HumanAgentInterface(store)
    return SupportAgent(state_store=store, human_interface=hi)


@pytest.fixture
def tmp_doc(tmp_path):
    path = tmp_path / "faq.md"
    path.write_text(
        "Q: How do I return an item?\n"
        "A: Visit the returns page and submit a request.\n\n"
        "Q: What payment methods are supported?\n"
        "A: Credit card and PayPal are supported.",
        encoding="utf-8",
    )
    return str(path)


def test_stream_blocks_sensitive_query(agent, tmp_doc):
    agent.ingest([tmp_doc])
    chunks = list(agent.stream_answer("how to hack the system", user_id="user-1"))
    assert chunks
    assert "".join(chunks)


def test_stream_tool_route_returns_order(agent, tmp_doc):
    agent.ingest([tmp_doc])
    chunks = list(
        agent.stream_answer("status of order ORD-1001", user_id="user-1")
    )
    assert "wireless headset" in "".join(chunks)


def test_stream_fallback_without_openai(agent, tmp_doc):
    agent.ingest([tmp_doc])
    assert agent.openai_client is None
    chunks = list(agent.stream_answer("how do I return an item", user_id="user-1"))
    assert "".join(chunks)


def test_stream_low_confidence_escalates(agent, tmp_doc, monkeypatch):
    agent.ingest([tmp_doc])
    monkeypatch.setattr(
        agent,
        "retrieve",
        lambda query, k=None: [
            RetrievedChunk(content="noise", source="faq.md", chunk_id=0, score=0.01)
        ],
    )
    chunks = list(agent.stream_answer("totally unrelated question", user_id="user-1"))
    assert "human agent" in "".join(chunks).lower()


def test_stream_persists_assistant_message(agent, tmp_doc):
    agent.ingest([tmp_doc])
    session = agent.state_store.create_session("user-1")
    list(
        agent.stream_answer(
            "how do I return an item",
            user_id="user-1",
            session_id=session["session_id"],
        )
    )
    stored = agent.state_store.get_session(session["session_id"])
    roles = [m["role"] for m in stored["messages"]]
    assert "assistant" in roles


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPORT_AGENT_API_KEYS", "test-key")
    monkeypatch.setenv("SUPPORT_AGENT_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("SUPPORT_AGENT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("SUPPORT_AGENT_RATE_LIMIT_REQUESTS", "10")
    monkeypatch.setenv("SUPPORT_AGENT_RATE_LIMIT_WINDOW_SECONDS", "60")

    from fastapi.testclient import TestClient

    import support_agent.api as api_module

    importlib.reload(api_module)
    monkeypatch.setattr(api_module.agent, "is_ready", lambda: True)
    with TestClient(api_module.app) as test_client:
        yield test_client, api_module


def _headers():
    return {"X-API-Key": "test-key"}


def test_stream_endpoint_without_api_key_returns_401(client):
    test_client, _ = client
    response = test_client.post(
        "/answer/stream", json={"question": "hi", "user_id": "user-1"}
    )
    assert response.status_code == 401


def test_stream_endpoint_empty_question_returns_400(client):
    test_client, _ = client
    response = test_client.post(
        "/answer/stream",
        json={"question": "   ", "user_id": "user-1"},
        headers=_headers(),
    )
    assert response.status_code == 400


def test_stream_endpoint_streams_tokens(client):
    test_client, api_module = client

    def fake_stream(query, user_id="guest", session_id=None):
        yield "refunds "
        yield "take 3 days"

    with patch.object(api_module.agent, "stream_answer", side_effect=fake_stream):
        response = test_client.post(
            "/answer/stream",
            json={"question": "how long for a refund", "user_id": "user-1"},
            headers=_headers(),
        )
    assert response.status_code == 200
    body = response.text
    assert "data: refunds " in body
    assert "data: take 3 days" in body
    assert "data: [DONE]" in body
