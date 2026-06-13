import json
import os
from typing import Any, Dict, List

from support_agent.agent import SupportAgent
from support_agent.human_interface import HumanAgentInterface
from support_agent.state_store import StateStore


def _load_doc_paths(project_root: str) -> List[str]:
    data_dir = os.path.join(project_root, "data")
    return [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".md")
    ]


def _load_eval_cases(project_root: str) -> List[Dict[str, Any]]:
    dataset_path = os.path.join(project_root, "data", "eval_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_agent(project_root: str) -> SupportAgent:
    db_path = os.path.join(project_root, "runtime", "eval.db")
    state_store = StateStore(db_path=db_path)
    agent = SupportAgent(
        state_store=state_store,
        human_interface=HumanAgentInterface(state_store),
    )
    agent.ingest(_load_doc_paths(project_root))
    return agent


def evaluate_case(agent: SupportAgent, case: Dict[str, Any]) -> Dict[str, Any]:
    result = agent.answer_with_metadata(
        case["question"],
        user_id=case["user_id"],
        session_id=f"eval-{case['name']}",
    )
    expected = case["expected"]
    answer_text = result.answer.lower()
    passed = True
    failures: List[str] = []

    for snippet in expected.get("answer_contains", []):
        if snippet.lower() not in answer_text:
            passed = False
            failures.append(f"answer missing snippet: {snippet}")

    if result.escalated != expected["escalated"]:
        passed = False
        failures.append("unexpected escalated flag")

    if result.tool_name != expected["tool_name"]:
        passed = False
        failures.append("unexpected tool_name")

    if result.guardrail_reason != expected["guardrail_reason"]:
        passed = False
        failures.append("unexpected guardrail_reason")

    return {
        "name": case["name"],
        "passed": passed,
        "trace_id": result.trace_id,
        "tool_name": result.tool_name,
        "guardrail_reason": result.guardrail_reason,
        "failures": failures,
    }


def main() -> int:
    project_root = os.path.dirname(__file__)
    agent = build_agent(project_root)
    cases = _load_eval_cases(project_root)
    results = [evaluate_case(agent, case) for case in cases]
    passed = sum(1 for item in results if item["passed"])

    print(json.dumps({"passed": passed, "total": len(results), "results": results}, indent=2))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
