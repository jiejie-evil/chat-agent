from dataclasses import dataclass
from typing import Optional

_BLOCKED_PATTERNS = [
    "hack", "exploit", "illegal", "password reset",
    "sql injection", "xss", "malware", "phishing",
]

@dataclass
class Decision:
    action: str  # "allow" | "block" | "escalate"
    reason: Optional[str] = None
    message: Optional[str] = None


class Guardrails:
    def inspect_query(self, query: str) -> Decision:
        lowered = query.lower()
        for pattern in _BLOCKED_PATTERNS:
            if pattern in lowered:
                return Decision(
                    action="block",
                    reason="sensitive_content",
                    message="This request cannot be processed.",
                )
        return Decision(action="allow")

    def evaluate_tool_result(self, success: bool, message: str) -> Decision:
        if not success:
            return Decision(action="escalate", reason="tool_failure")
        return Decision(action="allow")

    def evaluate_retrieval_confidence(
        self, confidence: float, min_score: float
    ) -> Decision:
        if confidence < min_score:
            return Decision(
                action="escalate",
                reason="low_confidence",
                message=(
                    "I don't have enough information to answer this question accurately. "
                    "A human agent will follow up with you shortly."
                ),
            )
        return Decision(action="allow")
