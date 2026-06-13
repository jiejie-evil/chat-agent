from dataclasses import dataclass
from typing import Optional


SENSITIVE_TERMS = [
    "cvv",
    "cvc",
    "credit card number",
    "bank account password",
    "social security number",
    "身份证号",
    "银行卡密码",
    "信用卡卡号",
    "验证码",
]


@dataclass
class GuardrailDecision:
    action: str
    reason: Optional[str] = None
    message: Optional[str] = None


class Guardrails:
    """Policy checks for support interactions."""

    def inspect_query(self, query: str) -> GuardrailDecision:
        lowered = query.lower()
        for term in SENSITIVE_TERMS:
            if term.lower() in lowered:
                return GuardrailDecision(
                    action="block",
                    reason="sensitive_content",
                    message=(
                        "I cannot process or collect highly sensitive credentials or payment secrets. "
                        "Please remove that information and contact a human agent through a secure channel if needed."
                    ),
                )
        return GuardrailDecision(action="allow")

    def evaluate_tool_result(self, success: bool, message: str) -> GuardrailDecision:
        if success:
            return GuardrailDecision(action="allow")

        lowered = message.lower()
        if "does not belong to user" in lowered or "please sign in" in lowered:
            return GuardrailDecision(
                action="escalate",
                reason="unauthorized_access",
                message=message,
            )
        return GuardrailDecision(action="allow")

    def evaluate_retrieval_confidence(self, confidence: float, minimum_score: float) -> GuardrailDecision:
        if confidence < minimum_score:
            return GuardrailDecision(
                action="escalate",
                reason="low_confidence",
                message=(
                    "I could not verify enough grounded knowledge for this request. "
                    "Please hand the conversation to a human agent."
                ),
            )
        return GuardrailDecision(action="allow")
