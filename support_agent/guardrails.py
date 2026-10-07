import re
from dataclasses import dataclass
from typing import Optional

_BLOCKED_PATTERNS = [
    "hack", "exploit", "illegal", "password reset",
    "sql injection", "xss", "malware", "phishing",
    "prompt injection", "jailbreak", "system prompt",
]
_PAYMENT_DATA_PATTERNS = (
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    re.compile(r"\b(?:cvv|cvc|security code)\s*[:=]?\s*\d{3,4}\b", re.I),
)

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
        if any(pattern.search(query) for pattern in _PAYMENT_DATA_PATTERNS):
            return Decision(
                action="block",
                reason="sensitive_content",
                message="Please do not share payment card details. Use a secure channel instead.",
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
