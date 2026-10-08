import json
import os
import shutil
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


def build_agent(project_root: str, runtime_dir: str | None = None) -> SupportAgent:
    """Build an evaluation agent with isolated SQLite and Chroma storage.

    Evaluation must never share the application's persistent Chroma directory:
    ingest deletes/recreates the ``knowledge`` collection, so sharing it can
    invalidate another agent while it is querying.
    """
    runtime_path = runtime_dir or tempfile.mkdtemp(prefix="support-agent-eval-")
    os.makedirs(runtime_path, exist_ok=True)
    state_store = StateStore(db_path=os.path.join(runtime_path, "eval.db"))
    agent = SupportAgent(
        state_store=state_store,
        human_interface=HumanAgentInterface(state_store),
        chroma_persist_dir=os.path.join(runtime_path, "chroma"),
    )
    agent.ingest(_load_doc_paths(project_root))
    return agent


def evaluate_case(agent: SupportAgent, case: dict[str, Any]) -> dict[str, Any]:
    question = case.get("question", case.get("query", ""))
    try:
        result = agent.answer_with_metadata(
            question,
            user_id=case.get("user_id", "eval-user"),
            session_id=case.get("session_id"),
        )
    except Exception as exc:
        return {
            "name": case.get("name", "case"),
            "passed": False,
            "trace_id": None,
            "tool_name": None,
            "guardrail_reason": None,
            "failures": [f"evaluation error: {type(exc).__name__}: {exc}"],
        }
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


def evaluate_retrieval_case(
    agent: SupportAgent,
    case: dict[str, Any],
) -> dict[str, Any]:
    try:
        results = agent.retrieve(case["query"], k=3)
        hit = any(case["expected_source"] in result.content for result in results)
        return {"hit": hit, "error": None}
    except Exception as exc:
        return {
            "hit": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    project_root = os.path.dirname(__file__)
    cases = _load_eval_cases(project_root)
    if isinstance(cases, list):
        runtime_dir = tempfile.mkdtemp(prefix="support-agent-eval-")
        try:
            agent = build_agent(project_root, runtime_dir=runtime_dir)
            results = [evaluate_case(agent, case) for case in cases]
            passed = sum(item["passed"] for item in results)
            print(json.dumps({"passed": passed, "total": len(results), "results": results}, ensure_ascii=False, indent=2))
            return 0 if passed == len(results) else 1
        finally:
            shutil.rmtree(runtime_dir, ignore_errors=True)
    data_dir = Path(project_root) / "data"
    temp_dir = tempfile.mkdtemp(prefix="support-agent-eval-")
    try:
        store = StateStore(db_path=str(Path(temp_dir) / "eval.db"))
        agent = SupportAgent(
            state_store=store,
            human_interface=HumanAgentInterface(store),
            chroma_persist_dir=str(Path(temp_dir) / "chroma"),
        )
        agent.ingest([str(data_dir / "rag_knowledge_base.md")])
        retrieval = cases.get("retrieval", [])
        dialogues = cases.get("dialogues", [])
        retrieval_results = [evaluate_retrieval_case(agent, item) for item in retrieval]
        hits = sum(item["hit"] for item in retrieval_results)
        retrieval_errors = sum(item["error"] is not None for item in retrieval_results)
        passed = 0
        latencies = []
        false_escalations = 0
        fallback_latencies = []
        dialogue_errors = 0
        for item in dialogues:
            started = time.perf_counter()
            result = evaluate_case(agent, item)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            passed += int(result["passed"])
            dialogue_errors += int(
                any(failure.startswith("evaluation error:") for failure in result["failures"])
            )
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
            "retrieval_evaluated_count": len(retrieval_results),
            "retrieval_error_count": retrieval_errors,
            "retrieval_recall": retrieval_recall,
            "dialogue_evaluated_count": len(dialogues),
            "dialogue_error_count": dialogue_errors,
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
    finally:
        # Chroma may keep SQLite handles briefly on Windows. Cleanup is best
        # effort; evaluation correctness must not depend on deleting the temp
        # directory successfully.
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
