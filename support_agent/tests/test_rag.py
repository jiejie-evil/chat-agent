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
def chinese_doc(tmp_path):
    path = tmp_path / "zh_faq.md"
    path.write_text(
        "Q: 退款需要多久？\nA: 退款申请审核后3-5个工作日内完成。\n\n"
        "Q: 如何下单？\nA: 登录账号，选好商品加入购物车，填写地址后支付。",
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def english_doc(tmp_path):
    path = tmp_path / "en_faq.md"
    path.write_text(
        "Q: How do I track my order?\nA: Use the tracking number in your shipment email.\n\n"
        "Q: How do I return an item?\nA: Submit a return request from the order details page.",
        encoding="utf-8",
    )
    return str(path)


def test_chinese_query_matches_chinese_content(agent, chinese_doc):
    agent.ingest([chinese_doc])
    results = agent.retrieve("退款需要多久")

    assert results
    top = results[0]
    assert top.score > agent.min_score
    assert "退款" in top.content


def test_english_query_matches_english_content(agent, english_doc):
    agent.ingest([english_doc])
    results = agent.retrieve("how to return an item")

    assert results
    assert results[0].score > agent.min_score
    assert "return" in results[0].content.lower()


def test_ingest_idempotent(agent, chinese_doc):
    agent.ingest([chinese_doc])
    assert agent.is_ready()
    chunk_count_1 = agent._collection.count()

    agent.ingest([chinese_doc])
    assert agent.is_ready()
    chunk_count_2 = agent._collection.count()

    assert chunk_count_1 == chunk_count_2


def test_retrieve_respects_top_k(agent, english_doc):
    agent.ingest([english_doc])
    results = agent.retrieve("order tracking return", k=1)

    assert len(results) <= 1


def test_retrieve_scores_positive(agent, english_doc):
    agent.ingest([english_doc])
    results = agent.retrieve("tracking order")

    for chunk in results:
        assert chunk.score > 0


def test_build_context_format(agent, english_doc):
    agent.ingest([english_doc])
    chunks = [
        RetrievedChunk(content="refunds take 3 days", source="faq.md", chunk_id=0, score=0.85),
        RetrievedChunk(content="use tracking number", source="faq.md", chunk_id=1, score=0.60),
    ]
    context = agent.build_context(chunks)

    assert "Source:" in context
    assert "Chunk:" in context
    assert "Score:" in context
    assert "Content:" in context
    assert "refunds" in context
