import json
from pathlib import Path


def test_eval_dataset_meets_prd_shape():
    path = Path(__file__).parents[1] / "data" / "eval_dataset.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))
    assert len(dataset["retrieval"]) >= 300
    assert len(dataset["dialogues"]) >= 500
    assert all("expected_answer_contains" in item for item in dataset["dialogues"])
    assert all("expected_route" in item for item in dataset["dialogues"])
    assert any(item["expected_route"] == "create_ticket" for item in dataset["dialogues"])
    assert any(item["expected_guardrail_reason"] == "sensitive_content" for item in dataset["dialogues"])
