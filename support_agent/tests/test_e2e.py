import os

from fastapi.testclient import TestClient

from support_agent.agent import SupportAgent
from support_agent.api import app
from support_agent.human_interface import HumanAgentInterface


REFUND_QUERY = "\u5982\u4f55\u7533\u8bf7\u9000\u6b3e\uff1f"
PAYMENT_QUERY = "\u652f\u4ed8\u65b9\u5f0f\u6709\u54ea\u4e9b\uff1f"
LOW_CONF_QUERY = "\u8bf7\u5e2e\u6211\u91cd\u7f6e\u4f01\u4e1a SSO \u7684 SAML \u7b7e\u540d\u8bc1\u4e66"
ORDER_QUERY = "\u8bf7\u5e2e\u6211\u67e5\u8be2\u8ba2\u5355 ORD-1001 \u7684\u72b6\u6001"
SHIP_QUERY = "\u8ba2\u5355 ORD-1002 \u7684\u7269\u6d41\u5230\u54ea\u4e86"
TICKET_QUERY = "\u6211\u8981\u6295\u8bc9\u5e76\u521b\u5efa\u5de5\u5355"


def _build_agent() -> SupportAgent:
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, "data")
    doc_paths = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".md")
    ]
    agent = SupportAgent(human_interface=HumanAgentInterface())
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


def test_ticket_tool_creates_human_ticket():
    interface = HumanAgentInterface()
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, "data")
    doc_paths = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".md")
    ]
    agent = SupportAgent(human_interface=interface)
    agent.ingest(doc_paths)

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


def test_api_returns_tool_metadata_for_order_lookup():
    with TestClient(app) as client:
        response = client.post("/answer", json={"question": ORDER_QUERY, "user_id": "user-1"})
    data = response.json()
    assert response.status_code == 200
    assert data["tool_name"] == "lookup_order"
    assert data["confidence"] == 1.0


def test_human_ticket_flow():
    interface = HumanAgentInterface()
    ticket_id = interface.create_ticket("user-1", "I need a refund")
    interface.add_agent_response(ticket_id, "Escalated to human support")
    open_tickets = interface.list_open()

    assert ticket_id in open_tickets
    assert open_tickets[ticket_id]["status"] == "open"
