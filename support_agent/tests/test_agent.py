import pytest

from support_agent.agent import AnswerResult, RetrievedChunk, SupportAgent
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


def test_ingest_and_retrieve_returns_chunks(agent, tmp_doc):
    agent.ingest([tmp_doc])
    results = agent.retrieve("how to return an item")

    assert results
    assert all(isinstance(chunk, RetrievedChunk) for chunk in results)
    assert results[0].score > 0


def test_retrieve_raises_before_ingest(agent):
    with pytest.raises(RuntimeError):
        agent.retrieve("anything")


def test_is_ready_reflects_ingest(agent, tmp_doc):
    assert agent.is_ready() is False
    agent.ingest([tmp_doc])
    assert agent.is_ready() is True


def test_ingest_empty_docs_raises(agent, tmp_path):
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        agent.ingest([str(empty)])


def test_guardrails_block_sensitive_query(agent, tmp_doc):
    agent.ingest([tmp_doc])
    result = agent.answer_with_metadata("how to hack the system", user_id="user-1")

    assert result.escalated is True
    assert result.guardrail_reason == "sensitive_content"


def test_low_confidence_escalates(agent, tmp_doc, monkeypatch):
    agent.ingest([tmp_doc])
    monkeypatch.setattr(
        agent,
        "retrieve",
        lambda query, k=None: [
            RetrievedChunk(content="noise", source="faq.md", chunk_id=0, score=0.01)
        ],
    )
    result = agent.answer_with_metadata("totally unrelated question", user_id="user-1")

    assert result.escalated is True


def test_answer_fallback_without_openai(agent, tmp_doc):
    agent.ingest([tmp_doc])
    assert agent.openai_client is None
    result = agent.answer_with_metadata("how do I return an item", user_id="user-1")

    assert result.answer
    assert result.escalated is False


def test_answer_returns_metadata_type(agent, tmp_doc):
    agent.ingest([tmp_doc])
    result = agent.answer_with_metadata("payment methods", user_id="user-1")

    assert isinstance(result, AnswerResult)
    assert result.session_id
    assert result.trace_id


def test_build_context_format(agent):
    chunks = [
        RetrievedChunk(content="hello", source="faq.md", chunk_id=0, score=0.9),
    ]
    context = agent.build_context(chunks)

    assert "Source:" in context
    assert "Chunk:" in context
    assert "Score:" in context
    assert "Content:" in context


def test_tool_route_check_order(agent, tmp_doc):
    agent.ingest([tmp_doc])
    result = agent.answer_with_metadata(
        "what is the status of order ORD-1001", user_id="user-1"
    )

    assert result.tool_name == "check_order"
    assert "wireless headset" in result.answer
