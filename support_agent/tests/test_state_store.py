import pytest

from support_agent.state_store import StateStore


@pytest.fixture
def store(tmp_path):
    s = StateStore(db_path=str(tmp_path / "store.db"))
    return s


def test_ensure_user_and_get_user(store):
    store.ensure_user("u-1", display_name="Alice", email="a@x.com", tier="vip")
    user = store.get_user("u-1")
    assert user["display_name"] == "Alice"
    assert user["tier"] == "vip"


def test_get_user_missing_raises(store):
    with pytest.raises(KeyError):
        store.get_user("no-such-user")


def test_update_user(store):
    store.ensure_user("u-2", display_name="Bob")
    updated = store.update_user("u-2", email="bob@x.com", tier="premium")
    assert updated["email"] == "bob@x.com"
    assert updated["tier"] == "premium"


def test_ensure_session_creates_and_retrieves(store):
    store.ensure_user("u-1")
    sess = store.ensure_session("u-1", session_id="sess-abc")
    assert sess["session_id"] == "sess-abc"
    # calling again returns same session
    sess2 = store.ensure_session("u-1", session_id="sess-abc")
    assert sess2["session_id"] == "sess-abc"


def test_ensure_session_wrong_user_raises(store):
    store.ensure_user("u-1")
    store.ensure_user("u-2")
    store.ensure_session("u-1", session_id="sess-x")
    with pytest.raises(ValueError):
        store.ensure_session("u-2", session_id="sess-x")


def test_get_session_missing_raises(store):
    with pytest.raises(KeyError):
        store.get_session("nonexistent-session")


def test_list_sessions_filtered(store):
    store.ensure_user("u-1")
    store.ensure_user("u-2")
    store.ensure_session("u-1", session_id="s1")
    store.ensure_session("u-2", session_id="s2")
    u1_sessions = store.list_sessions(user_id="u-1")
    assert all(s["user_id"] == "u-1" for s in u1_sessions)
    all_sessions = store.list_sessions()
    assert len(all_sessions) >= 2


def test_trace_lifecycle(store):
    store.ensure_user("u-1")
    store.ensure_session("u-1", session_id="sess-t")
    store.create_trace("trace-1", "sess-t", "u-1", "test query")
    store.append_trace_event("trace-1", "node_enter", '{"node":"guardrails"}')
    store.finish_trace(
        "trace-1",
        answer="hello",
        confidence=0.9,
        escalated=False,
        tool_name=None,
        guardrail_reason=None,
        failure_reason=None,
        duration_ms=42,
    )
    trace = store.get_trace("trace-1")
    assert trace["answer"] == "hello"
    assert len(trace["events"]) == 1


def test_get_trace_missing_raises(store):
    with pytest.raises(KeyError):
        store.get_trace("no-trace")


def test_list_traces_filters(store):
    store.ensure_user("u-1")
    store.ensure_session("u-1", session_id="sess-lt")
    store.create_trace("t1", "sess-lt", "u-1", "q1")
    store.create_trace("t2", "sess-lt", "u-1", "q2")
    results = store.list_traces(session_id="sess-lt")
    assert len(results) == 2
    results_u = store.list_traces(user_id="u-1")
    assert len(results_u) >= 2
    results_both = store.list_traces(session_id="sess-lt", user_id="u-1")
    assert len(results_both) == 2


def test_ticket_full_lifecycle(store):
    store.ensure_user("u-1")
    tid = store.create_ticket("u-1", "broken item")
    store.add_ticket_message(tid, "agent", "Looking into it")
    updated = store.update_ticket_status(tid, "closed")
    assert updated["status"] == "closed"
    ticket = store.get_ticket(tid)
    assert len(ticket["messages"]) == 2


def test_update_ticket_missing_raises(store):
    with pytest.raises(KeyError):
        store.update_ticket_status(999999, "closed")


def test_list_tickets_filters(store):
    store.ensure_user("u-1")
    t1 = store.create_ticket("u-1", "issue A")
    store.create_ticket("u-1", "issue B")
    store.update_ticket_status(t1, "closed")
    open_tickets = store.list_tickets(status="open")
    assert all(t["status"] == "open" for t in open_tickets)
    user_tickets = store.list_tickets(user_id="u-1")
    assert len(user_tickets) >= 2
