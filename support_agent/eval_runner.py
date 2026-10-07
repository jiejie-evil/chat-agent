import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from support_agent.agent import SupportAgent
from support_agent.human_interface import HumanAgentInterface
from support_agent.state_store import StateStore


def _load_doc_paths(project_root: str) -> list[str]:
    data_dir = os.path.join(project_root, "data")
    return [os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.endswith(".md")]


def _load_eval_cases(project_root: str) -> Any:
    with open(os.path.join(project_root, "data", "eval_dataset.json"), encoding="utf-8") as file:
        return json.load(file)


def build_agent(project_root: str) -> SupportAgent:
    state_store = StateStore(db_path=os.path.join(project_root, "runtime", "eval.db"))
    agent = SupportAgent(state_store=state_store, human_interface=HumanAgentInterface(state_store))
    agent.ingest(_load_doc_paths(project_root))
    return agent


def evaluate_case(agent: SupportAgent, case: dict[str, Any]) -> dict[str, Any]:
    result = agent.answer_with_metadata(case.get("question", case.get("query", "")), user_id=case.get("user_id", "eval-user"), session_id=case.get("session_id"))
    expected = case.get("expected", {})
    expected_route = expected.get("expected_route", case.get("expected_route"))
    expected_escalated = expected.get("expected_escalated", expected.get("escalated", case.get("expected_escalated")))
    expected_guardrail = expected.get("expected_guardrail_reason", expected.get("guardrail_reason", case.get("expected_guardrail_reason")))
    expected_answer = expected.get("expected_answer_contains", expected.get("answer_contains", case.get("expected_answer_contains", [])))
    actual_route = result.tool_name or ("escalate" if result.escalated else "retrieve")
    failures=[]
    text=result.answer.lower()
    failures.extend(f"answer missing snippet: {snippet}" for snippet in expected_answer if snippet.lower() not in text)
    if expected_route is not None and actual_route != expected_route:
        failures.append("unexpected route")
    if expected_escalated is not None and result.escalated != expected_escalated:
        failures.append("unexpected escalated flag")
    if expected_guardrail is not None and result.guardrail_reason != expected_guardrail:
        failures.append("unexpected guardrail_reason")
    expected_tool = expected.get("tool_name")
    if expected_tool is not None and result.tool_name != expected_tool:
        failures.append("unexpected tool_name")
    return {"name": case.get("name", "case"), "passed": not failures, "trace_id": result.trace_id, "tool_name": result.tool_name, "guardrail_reason": result.guardrail_reason, "failures": failures}


def main() -> int:
    project_root = os.path.dirname(__file__)
    cases = _load_eval_cases(project_root)
    if isinstance(cases, list):
        agent = build_agent(project_root)
        results = [evaluate_case(agent, case) for case in cases]
        passed = sum(item["passed"] for item in results)
        print(json.dumps({"passed": passed, "total": len(results), "results": results}, ensure_ascii=False, indent=2))
        return 0 if passed == len(results) else 1
    data_dir = Path(project_root) / "data"
    with tempfile.TemporaryDirectory(prefix="support-agent-eval-") as temp_dir:
        store = StateStore(db_path=str(Path(temp_dir) / "eval.db"))
        agent = SupportAgent(state_store=store, human_interface=HumanAgentInterface(store))
        agent.ingest([str(data_dir / "rag_knowledge_base.md")])
        retrieval = cases.get("retrieval", [])
        dialogues = cases.get("dialogues", [])
        hits = sum(
            any(
                item["expected_source"] in result.content
                for result in agent.retrieve(item["query"], k=3)
            )
            for item in retrieval
        )
        passed = 0
        latencies = []
        false_escalations = 0
        fallback_latencies = []
        for item in dialogues:
            started = time.perf_counter()
            result = evaluate_case(agent, item)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            passed += int(result["passed"])
            if (
                item.get("expected_route") == "retrieve"
                and result["guardrail_reason"] == "low_confidence"
            ):
                false_escalations += 1
            if agent.openai_client is None:
                fallback_latencies.append(elapsed)
        dialogue_pass_rate = passed / len(dialogues) if dialogues else 0.0
        retrieval_recall = hits / len(retrieval) if retrieval else 0.0
        average_latency = sum(latencies) / len(latencies) if latencies else 0.0
        fallback_latency = (
            sum(fallback_latencies) / len(fallback_latencies)
            if fallback_latencies
            else 0.0
        )
        false_escalation_rate = false_escalations / len(dialogues) if dialogues else 0.0
        sorted_latencies = sorted(latencies)
        p95_index = min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95)) if sorted_latencies else 0
        p95_latency = sorted_latencies[p95_index] if sorted_latencies else 0.0
        targets = {
            "dialogue_pass_rate": 0.87,
            "retrieval_recall": 0.91,
            "average_latency_seconds": 2.5,
            "fallback_latency_seconds": 0.5,
            "false_escalation_rate": 0.08,
        }
        passed_targets = {
            "dialogue_pass_rate": dialogue_pass_rate >= targets["dialogue_pass_rate"],
            "retrieval_recall": retrieval_recall >= targets["retrieval_recall"],
            "average_latency_seconds": average_latency <= targets["average_latency_seconds"],
            "fallback_latency_seconds": fallback_latency <= targets["fallback_latency_seconds"],
            "false_escalation_rate": false_escalation_rate <= targets["false_escalation_rate"],
        }
        summary = {
            "dialogue_count": len(dialogues),
            "dialogue_pass_rate": dialogue_pass_rate,
            "retrieval_count": len(retrieval),
            "retrieval_recall": retrieval_recall,
            "average_latency_seconds": average_latency,
            "fallback_latency_seconds": fallback_latency,
            "p95_latency_seconds": p95_latency,
            "false_escalation_rate": false_escalation_rate,
            "targets": targets,
            "passed": all(passed_targets.values()),
            "target_results": passed_targets,
        }
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
