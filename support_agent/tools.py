import re
from dataclasses import dataclass
from typing import Dict, Optional

from support_agent.human_interface import HumanAgentInterface


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
    def __init__(self, human_interface: HumanAgentInterface):
        self.human_interface = human_interface
        self.orders = {
            "ORD-1001": {
                "user_id": "user-1",
                "status": "paid",
                "amount": "HKD 299",
                "items": "wireless headset",
            },
            "ORD-1002": {
                "user_id": "user-2",
                "status": "shipped",
                "amount": "HKD 88",
                "items": "phone case",
            },
        }
        self.shipments = {
            "ORD-1002": {
                "carrier": "SF Express",
                "tracking_no": "SF123456789HK",
                "shipping_status": "in_transit",
                "eta": "2026-06-15",
            },
        }

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

    def execute(self, tool_name: str, query: str, user_id: str = "guest") -> ToolResult:
        if tool_name == "lookup_order":
            return self.lookup_order(query, user_id)
        if tool_name == "lookup_shipping":
            return self.lookup_shipping(query, user_id)
        if tool_name == "create_ticket":
            return self.create_ticket(query, user_id)
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

        order = self.orders.get(order_id)
        if not order:
            return ToolResult(
                "lookup_order",
                False,
                f"Order {order_id} was not found.",
                {"order_id": order_id},
            )

        if user_id != "guest" and order["user_id"] != user_id:
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
        shipment = self.shipments.get(order_id)
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

    def create_ticket(self, query: str, user_id: str) -> ToolResult:
        ticket_id = self.human_interface.create_ticket(user_id=user_id, initial_message=query)
        return ToolResult(
            "create_ticket",
            True,
            f"Ticket {ticket_id} has been created and handed to human support.",
            {"ticket_id": str(ticket_id), "user_id": user_id, "status": "open"},
        )

    def _extract_order_id(self, query: str) -> Optional[str]:
        match = re.search(r"ORD-\d+", query.upper())
        return match.group(0) if match else None
