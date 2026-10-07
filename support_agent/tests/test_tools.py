import pytest

from support_agent.human_interface import HumanAgentInterface
from support_agent.state_store import StateStore
from support_agent.tools import SupportToolbox


@pytest.fixture
def toolbox(tmp_path):
    store = StateStore(db_path=str(tmp_path / "tools.db"))
    store.ensure_seed_data()
    human = HumanAgentInterface(store)
    return SupportToolbox(human, store)


def test_detect_check_order_with_order_id(toolbox):
    assert toolbox.detect_tool("What is the status of order ORD-1001?") == "check_order"


def test_detect_check_shipment_with_order_id(toolbox):
    assert toolbox.detect_tool("Where is my shipment for ORD-1002?") == "check_shipment"


def test_detect_check_shipment_chinese(toolbox):
    assert toolbox.detect_tool("ORD-1002 的物流到哪了") == "check_shipment"


def test_detect_create_ticket_on_human_request(toolbox):
    assert toolbox.detect_tool("I want to talk to a human agent") == "create_ticket"


def test_detect_returns_none_for_plain_question(toolbox):
    assert toolbox.detect_tool("How long does a refund take?") == "create_ticket"


def test_detect_order_without_id_falls_through(toolbox):
    assert toolbox.detect_tool("I want to check my order") is None


def test_check_order_returns_details(toolbox):
    result = toolbox.execute("check_order", "status of ORD-1001", user_id="user-1")
    assert result.success is True
    assert "wireless headset" in result.message
    assert "paid" in result.message


def test_check_order_missing_id(toolbox):
    result = toolbox.execute("check_order", "status of my order", user_id="user-1")
    assert result.success is False
    assert "No order ID" in result.message


def test_check_order_unknown_id(toolbox):
    result = toolbox.execute("check_order", "status of ORD-9999", user_id="user-1")
    assert result.success is False
    assert "not found" in result.message


def test_check_order_rejects_other_users_order(toolbox):
    result = toolbox.execute("check_order", "status of ORD-1001", user_id="user-2")
    assert result.success is False
    assert "does not belong" in result.message


def test_check_order_rejects_guest(toolbox):
    result = toolbox.execute("check_order", "status of ORD-1001", user_id="guest")
    assert result.success is False
    assert "sign in" in result.message


def test_check_shipment_returns_tracking(toolbox):
    result = toolbox.execute("check_shipment", "track ORD-1002", user_id="user-2")
    assert result.success is True
    assert "SF Express" in result.message
    assert "SF123456789HK" in result.message


def test_check_shipment_missing_id(toolbox):
    result = toolbox.execute("check_shipment", "where is my package", user_id="user-2")
    assert result.success is False
    assert "No order ID" in result.message


def test_check_shipment_no_shipment_for_order(toolbox):
    result = toolbox.execute("check_shipment", "track ORD-1001", user_id="user-1")
    assert result.success is False
    assert "No shipment" in result.message


def test_check_shipment_rejects_other_users_order(toolbox):
    result = toolbox.execute("check_shipment", "track ORD-1002", user_id="user-1")
    assert result.success is False
    assert "does not belong" in result.message


def test_create_ticket_returns_ticket_id(toolbox):
    result = toolbox.execute(
        "create_ticket", "please help me", user_id="user-1"
    )
    assert result.success is True
    assert "Ticket #" in result.message


def test_execute_unknown_tool(toolbox):
    result = toolbox.execute("nonexistent", "anything", user_id="user-1")
    assert result.success is False
    assert "Unknown tool" in result.message
