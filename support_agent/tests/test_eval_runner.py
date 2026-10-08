import json
from pathlib import Path

from support_agent.eval_runner import evaluate_case, evaluate_retrieval_case


def test_eval_dataset_meets_prd_shape():
    path = Path(__file__).parents[1] / "data" / "eval_dataset.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))
    assert len(dataset["retrieval"]) >= 300
    assert len(dataset["dialogues"]) >= 500
    assert all("expected_answer_contains" in item for item in dataset["dialogues"])
    assert all("expected_route" in item for item in dataset["dialogues"])
    assert any(item["expected_route"] == "create_ticket" for item in dataset["dialogues"])
    assert any(item["expected_guardrail_reason"] == "sensitive_content" for item in dataset["dialogues"])


class FailingAgent:
    def answer_with_metadata(self, *args, **kwargs):
        raise RuntimeError("dialogue failed")

    def retrieve(self, *args, **kwargs):
        raise RuntimeError("retrieval failed")


def test_dialogue_error_is_reported_without_stopping_evaluation():
    result = evaluate_case(FailingAgent(), {"question": "test"})

    assert result["passed"] is False
    assert result["failures"] == [
        "evaluation error: RuntimeError: dialogue failed"
    ]


def test_retrieval_error_is_reported_without_stopping_evaluation():
    result = evaluate_retrieval_case(
        FailingAgent(),
        {"query": "test", "expected_source": "Q: test"},
    )

    assert result == {
        "hit": False,
        "error": "RuntimeError: retrieval failed",
    }
