import re
from dataclasses import dataclass
from typing import Dict, Optional

from support_agent.human_interface import HumanAgentInterface
from support_agent.state_store import StateStore


SHIPPING_TERMS = [
    "\u7269\u6d41",
    "\u5feb\u9012",
    "\u914d\u9001",
    "shipping",
    "tracking",
]

ORDER_TERMS = [
    "\u8ba2\u5355",
    "order",
    "\u72b6\u6001",
    "\u91d1\u989d",
    "\u8d2d\u4e70",
]

TICKET_TERMS = [
    "\u5de5\u5355",
    "ticket",
    "\u4eba\u5de5",
    "\u9000\u6b3e\u7533\u8bf7",
    "\u6295\u8bc9",
]


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    message: str
    payload: Dict[str, str]


class SupportToolbox:
    def __init__(self, human_interface: HumanAgentInterface, state_store: StateStore):
        self.human_interface = human_interface
        self.state_store = state_store

    def detect_tool(self, query: str) -> Optional[str]:
        lowered = query.lower()
        if self._extract_order_id(query):
            if any(token in lowered for token in SHIPPING_TERMS):
                return "lookup_shipping"
            if any(token in lowered for token in ORDER_TERMS):
                return "lookup_order"

        if any(token in lowered for token in TICKET_TERMS):
            return "create_ticket"
        return None

    def execute(
        self,
        tool_name: str,
        query: str,
        user_id: str = "guest",
        session_id: Optional[str] = None,
    ) -> ToolResult:
        if tool_name == "lookup_order":
            return self.lookup_order(query, user_id)
        if tool_name == "lookup_shipping":
            return self.lookup_shipping(query, user_id)
        if tool_name == "create_ticket":
            return self.create_ticket(query, user_id, session_id=session_id)
        return ToolResult(
            tool_name=tool_name,
            success=False,
            message="Unsupported tool request.",
            payload={},
        )

    def lookup_order(self, query: str, user_id: str) -> ToolResult:
        order_id = self._extract_order_id(query)
        if not order_id:
            return ToolResult(
                "lookup_order",
                False,
                "Please provide an order ID like ORD-1001.",
                {},
            )

        order = self.state_store.get_order(order_id)
        if not order:
            return ToolResult(
                "lookup_order",
                False,
                f"Order {order_id} was not found.",
                {"order_id": order_id},
            )

        if user_id == "guest":
            return ToolResult(
                "lookup_order",
                False,
                "Please sign in before querying a specific order.",
                {"order_id": order_id},
            )

        if order["user_id"] != user_id:
            return ToolResult(
                "lookup_order",
                False,
                f"Order {order_id} does not belong to user {user_id}.",
                {"order_id": order_id},
            )

        return ToolResult(
            "lookup_order",
            True,
            f"Order {order_id} is currently {order['status']}. "
            f"Items: {order['items']}. Amount: {order['amount']}.",
            {"order_id": order_id, **order},
        )

    def lookup_shipping(self, query: str, user_id: str) -> ToolResult:
        order_result = self.lookup_order(query, user_id)
        if not order_result.success:
            return ToolResult(
                "lookup_shipping",
                False,
                order_result.message,
                order_result.payload,
            )

        order_id = order_result.payload["order_id"]
        shipment = self.state_store.get_shipment(order_id)
        if not shipment:
            return ToolResult(
                "lookup_shipping",
                False,
                f"No shipment has been created for {order_id} yet.",
                {"order_id": order_id},
            )

        return ToolResult(
            "lookup_shipping",
            True,
            f"Order {order_id} is shipped with {shipment['carrier']}. "
            f"Tracking number: {shipment['tracking_no']}. "
            f"Status: {shipment['shipping_status']}. ETA: {shipment['eta']}.",
            {"order_id": order_id, **shipment},
        )

    def create_ticket(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> ToolResult:
        ticket_id = self.human_interface.create_ticket(
            user_id=user_id,
            initial_message=query,
            session_id=session_id,
        )
        return ToolResult(
            "create_ticket",
            True,
            f"Ticket {ticket_id} has been created and handed to human support.",
            {
                "ticket_id": str(ticket_id),
                "user_id": user_id,
                "session_id": session_id or "",
                "status": "open",
            },
        )

    def _extract_order_id(self, query: str) -> Optional[str]:
        match = re.search(r"ORD-\d+", query.upper())
        return match.group(0) if match else None
