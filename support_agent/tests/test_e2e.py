import os
import tempfile

from fastapi.testclient import TestClient

from support_agent.agent import SupportAgent
from support_agent.api import app
from support_agent.human_interface import HumanAgentInterface
from support_agent.state_store import StateStore


REFUND_QUERY = "\u5982\u4f55\u7533\u8bf7\u9000\u6b3e\uff1f"
PAYMENT_QUERY = "\u652f\u4ed8\u65b9\u5f0f\u6709\u54ea\u4e9b\uff1f"
LOW_CONF_QUERY = "\u8bf7\u5e2e\u6211\u91cd\u7f6e\u4f01\u4e1a SSO \u7684 SAML \u7b7e\u540d\u8bc1\u4e66"
ORDER_QUERY = "\u8bf7\u5e2e\u6211\u67e5\u8be2\u8ba2\u5355 ORD-1001 \u7684\u72b6\u6001"
SHIP_QUERY = "\u8ba2\u5355 ORD-1002 \u7684\u7269\u6d41\u5230\u54ea\u4e86"
TICKET_QUERY = "\u6211\u8981\u6295\u8bc9\u5e76\u521b\u5efa\u5de5\u5355"
SENSITIVE_QUERY = "My credit card number is 4111 1111 1111 1111 and cvv is 123"


def _build_agent() -> SupportAgent:
    temp_dir = tempfile.mkdtemp(prefix="support-agent-tests-")
    return _build_agent_with_store(StateStore(db_path=os.path.join(temp_dir, "state.db")))


def _build_agent_with_store(state_store: StateStore) -> SupportAgent:
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, "data")
    doc_paths = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".md")
    ]
    agent = SupportAgent(
        state_store=state_store,
        human_interface=HumanAgentInterface(state_store),
    )
    agent.ingest(doc_paths)
    return agent


def test_rag_answer_returns_grounded_text():
    agent = _build_agent()
    answer = agent.answer(REFUND_QUERY)
    assert isinstance(answer, str)
    assert "Answer grounded by retrieved knowledge" in answer or len(answer) > 0


def test_retrieve_returns_ranked_chunks():
    agent = _build_agent()
    results = agent.retrieve(PAYMENT_QUERY)
    assert results
    assert results[0].score > 0
    assert "PayPal" in results[0].content or "Q:" in results[0].content


def test_low_confidence_question_escalates():
    agent = _build_agent()
    result = agent.answer_with_metadata(LOW_CONF_QUERY)
    assert result.escalated is True
    assert result.confidence < agent.min_score
    assert result.guardrail_reason == "low_confidence"


def test_sensitive_content_is_blocked_before_retrieval():
    agent = _build_agent()
    result = agent.answer_with_metadata(SENSITIVE_QUERY, user_id="user-1")
    assert result.escalated is True
    assert result.guardrail_reason == "sensitive_content"
    assert "sensitive" in result.answer.lower() or "secure channel" in result.answer.lower()


def test_order_lookup_tool_returns_order_status():
    agent = _build_agent()
    result = agent.answer_with_metadata(ORDER_QUERY, user_id="user-1")
    assert result.tool_name == "lookup_order"
    assert result.escalated is False
    assert "ORD-1001" in result.answer
    assert "paid" in result.answer


def test_shipping_lookup_tool_returns_tracking_info():
    agent = _build_agent()
    result = agent.answer_with_metadata(SHIP_QUERY, user_id="user-2")
    assert result.tool_name == "lookup_shipping"
    assert result.escalated is False
    assert "SF123456789HK" in result.answer


def test_guest_cannot_access_order_data():
    agent = _build_agent()
    result = agent.answer_with_metadata(ORDER_QUERY, user_id="guest")
    assert result.tool_name == "lookup_order"
    assert result.escalated is True
    assert result.guardrail_reason == "unauthorized_access"
    assert "sign in" in result.answer.lower()


def test_user_cannot_access_other_users_order():
    agent = _build_agent()
    result = agent.answer_with_metadata(ORDER_QUERY, user_id="user-2")
    assert result.tool_name == "lookup_order"
    assert result.escalated is True
    assert result.guardrail_reason == "unauthorized_access"
    assert "does not belong" in result.answer.lower()


def test_ticket_tool_creates_human_ticket():
    temp_dir = tempfile.mkdtemp(prefix="support-agent-tests-")
    store = StateStore(db_path=os.path.join(temp_dir, "state.db"))
    interface = HumanAgentInterface(store)
    agent = _build_agent_with_store(store)

    result = agent.answer_with_metadata(TICKET_QUERY, user_id="user-3")
    assert result.tool_name == "create_ticket"
    assert result.escalated is True
    assert interface.list_open()


def test_api_returns_metadata():
    with TestClient(app) as client:
        response = client.post("/answer", json={"question": PAYMENT_QUERY, "user_id": "user-1"})
    data = response.json()
    assert response.status_code == 200
    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data
    assert "guardrail_reason" in data


def test_api_returns_tool_metadata_for_order_lookup():
    with TestClient(app) as client:
        response = client.post("/answer", json={"question": ORDER_QUERY, "user_id": "user-1"})
    data = response.json()
    assert response.status_code == 200
    assert data["tool_name"] == "lookup_order"
    assert data["confidence"] == 1.0


def test_human_ticket_flow():
    temp_dir = tempfile.mkdtemp(prefix="support-agent-tests-")
    store = StateStore(db_path=os.path.join(temp_dir, "state.db"))
    interface = HumanAgentInterface(store)
    ticket_id = interface.create_ticket("user-1", "I need a refund")
    interface.add_agent_response(ticket_id, "Escalated to human support")
    open_tickets = interface.list_open()

    assert ticket_id in open_tickets
    assert open_tickets[ticket_id]["status"] == "open"


def test_session_history_persists_across_store_reloads(tmp_path):
    db_path = tmp_path / "state.db"
    store = StateStore(db_path=str(db_path))
    agent = _build_agent_with_store(store)

    result = agent.answer_with_metadata(PAYMENT_QUERY, user_id="user-9", session_id="session-a")
    assert result.session_id == "session-a"

    reloaded_store = StateStore(db_path=str(db_path))
    session = reloaded_store.get_session("session-a")
    assert session["user_id"] == "user-9"
    assert len(session["messages"]) == 2
    assert session["messages"][0]["role"] == "user"
    assert session["messages"][1]["role"] == "assistant"
    trace = reloaded_store.get_trace(result.trace_id)
    assert trace["session_id"] == "session-a"
    assert trace["query"] == PAYMENT_QUERY
    assert trace["events"]


def test_user_identity_can_be_saved_and_queried(tmp_path):
    db_path = tmp_path / "state.db"
    store = StateStore(db_path=str(db_path))

    user = store.update_user(
        "user-88",
        display_name="Casey",
        email="casey@example.com",
        tier="vip",
    )

    reloaded_store = StateStore(db_path=str(db_path))
    fetched = reloaded_store.get_user("user-88")
    assert user["user_id"] == "user-88"
    assert fetched["display_name"] == "Casey"
    assert fetched["email"] == "casey@example.com"
    assert fetched["tier"] == "vip"


def test_ticket_flow_persists_and_can_be_recovered(tmp_path):
    db_path = tmp_path / "state.db"
    store = StateStore(db_path=str(db_path))
    interface = HumanAgentInterface(store)

    ticket_id = interface.create_ticket(
        user_id="user-66",
        initial_message="Please refund my order",
        session_id="session-z",
    )
    interface.add_agent_response(ticket_id, "Transferred to billing queue")
    interface.close_ticket(ticket_id)

    reloaded_store = StateStore(db_path=str(db_path))
    ticket = reloaded_store.get_ticket(ticket_id)
    assert ticket["user_id"] == "user-66"
    assert ticket["session_id"] == "session-z"
    assert ticket["status"] == "closed"
    assert len(ticket["messages"]) == 2


def test_api_exposes_persistent_state_queries(tmp_path):
    from support_agent import api as api_module

    db_path = tmp_path / "api-state.db"
    api_module.state_store = StateStore(db_path=str(db_path))
    api_module.human_interface = HumanAgentInterface(api_module.state_store)
    api_module.agent = SupportAgent(
        state_store=api_module.state_store,
        human_interface=api_module.human_interface,
    )

    with TestClient(api_module.app) as client:
        answer_response = client.post(
            "/answer",
            json={"question": PAYMENT_QUERY, "user_id": "user-55", "session_id": "session-api"},
        )
        assert answer_response.status_code == 200
        answer_data = answer_response.json()
        assert answer_data["session_id"] == "session-api"
        assert answer_data["trace_id"]

        session_response = client.get("/sessions/session-api")
        traces_response = client.get("/traces", params={"session_id": "session-api"})
        user_response = client.put(
            "/users/user-55",
            json={"display_name": "Taylor", "email": "taylor@example.com", "tier": "vip"},
        )
        ticket_response = client.post(
            "/tickets",
            json={
                "user_id": "user-55",
                "message": "Need human support",
                "session_id": "session-api",
            },
        )

        assert session_response.status_code == 200
        assert traces_response.status_code == 200
        assert user_response.status_code == 200
        assert ticket_response.status_code == 200

        trace_id = answer_data["trace_id"]
        trace_detail_response = client.get(f"/traces/{trace_id}")
        assert trace_detail_response.status_code == 200
        assert trace_detail_response.json()["events"]

        ticket_id = ticket_response.json()["ticket_id"]
        get_ticket_response = client.get(f"/tickets/{ticket_id}")
        close_ticket_response = client.patch(f"/tickets/{ticket_id}", json={"status": "closed"})

        assert get_ticket_response.status_code == 200
        assert close_ticket_response.status_code == 200
        assert close_ticket_response.json()["status"] == "closed"


def test_api_returns_guardrail_reason_for_sensitive_content(tmp_path):
    from support_agent import api as api_module

    db_path = tmp_path / "api-guardrails.db"
    api_module.state_store = StateStore(db_path=str(db_path))
    api_module.human_interface = HumanAgentInterface(api_module.state_store)
    api_module.agent = SupportAgent(
        state_store=api_module.state_store,
        human_interface=api_module.human_interface,
    )

    with TestClient(api_module.app) as client:
        response = client.post(
            "/answer",
            json={"question": SENSITIVE_QUERY, "user_id": "user-77", "session_id": "session-guard"},
        )
    data = response.json()
    assert response.status_code == 200
    assert data["guardrail_reason"] == "sensitive_content"
    assert data["escalated"] is True


def test_full_support_flow_runs_end_to_end(tmp_path):
    from support_agent import api as api_module

    db_path = tmp_path / "api-full-flow.db"
    api_module.state_store = StateStore(db_path=str(db_path))
    api_module.human_interface = HumanAgentInterface(api_module.state_store)
    api_module.agent = SupportAgent(
        state_store=api_module.state_store,
        human_interface=api_module.human_interface,
    )

    with TestClient(api_module.app) as client:
        user_response = client.put(
            "/users/user-200",
            json={"display_name": "Morgan", "email": "morgan@example.com", "tier": "vip"},
        )
        assert user_response.status_code == 200

        rag_response = client.post(
            "/answer",
            json={"question": REFUND_QUERY, "user_id": "user-200", "session_id": "session-full"},
        )
        assert rag_response.status_code == 200
        rag_data = rag_response.json()
        assert rag_data["session_id"] == "session-full"
        assert rag_data["trace_id"]
        assert rag_data["guardrail_reason"] is None

        tool_response = client.post(
            "/answer",
            json={"question": ORDER_QUERY, "user_id": "user-1", "session_id": "session-order"},
        )
        assert tool_response.status_code == 200
        tool_data = tool_response.json()
        assert tool_data["tool_name"] == "lookup_order"
        assert tool_data["trace_id"]

        ticket_response = client.post(
            "/tickets",
            json={
                "user_id": "user-200",
                "message": "I still need manual help",
                "session_id": "session-full",
            },
        )
        assert ticket_response.status_code == 200
        ticket_id = ticket_response.json()["ticket_id"]

        session_response = client.get("/sessions/session-full")
        traces_response = client.get("/traces", params={"session_id": "session-full"})
        trace_detail_response = client.get(f"/traces/{rag_data['trace_id']}")
        ticket_detail_response = client.get(f"/tickets/{ticket_id}")
        ticket_close_response = client.patch(f"/tickets/{ticket_id}", json={"status": "closed"})

        assert session_response.status_code == 200
        assert len(session_response.json()["messages"]) >= 2
        assert traces_response.status_code == 200
        assert traces_response.json()["traces"]
        assert trace_detail_response.status_code == 200
        assert trace_detail_response.json()["events"]
        assert ticket_detail_response.status_code == 200
        assert ticket_detail_response.json()["status"] == "open"
        assert ticket_close_response.status_code == 200
        assert ticket_close_response.json()["status"] == "closed"
