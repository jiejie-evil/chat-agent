import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from support_agent.resilience import log_event


@dataclass
class TraceContext:
    trace_id: str
    started_at: float


class Observability:
    """Small local observability helper for traces, events, and eval outputs."""

    def __init__(self, state_store):
        self.state_store = state_store

    def start_trace(
        self,
        user_id: str,
        session_id: str,
        query: str,
    ) -> TraceContext:
        trace_id = f"trace-{uuid.uuid4().hex[:12]}"
        started_at = time.time()
        self.state_store.create_trace(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            query=query,
        )
        self.log_event(
            trace_id,
            "trace_started",
            {
                "query": query,
                "session_id": session_id,
                "user_id": user_id,
            },
        )
        return TraceContext(trace_id=trace_id, started_at=started_at)

    def log_event(
        self,
        trace_id: str,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        event_payload = payload or {}
        self.state_store.append_trace_event(
            trace_id=trace_id,
            event_type=event_type,
            payload=json.dumps(event_payload, ensure_ascii=True),
        )
        log_event(
            {
                "trace_id": trace_id,
                "event_type": event_type,
                **event_payload,
            }
        )

    def finish_trace(
        self,
        trace: TraceContext,
        answer: str,
        confidence: float,
        escalated: bool,
        tool_name: Optional[str],
        guardrail_reason: Optional[str],
        failure_reason: Optional[str] = None,
    ) -> None:
        duration_ms = int((time.time() - trace.started_at) * 1000)
        self.log_event(
            trace.trace_id,
            "trace_finished",
            {
                "confidence": confidence,
                "escalated": escalated,
                "tool_name": tool_name,
                "guardrail_reason": guardrail_reason,
                "failure_reason": failure_reason,
                "duration_ms": duration_ms,
            },
        )
        self.state_store.finish_trace(
            trace_id=trace.trace_id,
            answer=answer,
            confidence=confidence,
            escalated=escalated,
            tool_name=tool_name,
            guardrail_reason=guardrail_reason,
            failure_reason=failure_reason,
            duration_ms=duration_ms,
        )
