import re
from dataclasses import dataclass
from typing import Optional

from support_agent.state_store import StateStore


@dataclass
class ToolResult:
    success: bool
    message: str


_ASCII_ORDER = {"order", "purchase", "buy", "bought", "paid"}
_ASCII_SHIPMENT = {"ship", "track", "delivery", "tracking", "shipment"}
_ASCII_TICKET = {"help", "agent", "human", "complaint"}
_CJK_ORDER = ("订单",)
_CJK_SHIPMENT = ("物流", "快递", "运输")
_CJK_TICKET = ("人工", "客服", "投诉")
_ORDER_ID_RE = re.compile(r"(ORD-\d+)", re.IGNORECASE)


class SupportToolbox:
    def __init__(self, human_interface, state_store: StateStore):
        self.human_interface = human_interface
        self.state_store = state_store

    def detect_tool(self, query: str) -> Optional[str]:
        lowered = query.lower()
        words = set(re.findall(r"[a-z0-9]+", lowered))
        has_order_id = _ORDER_ID_RE.search(query) is not None
        is_shipment = bool(words & _ASCII_SHIPMENT) or any(p in query for p in _CJK_SHIPMENT)
        is_order = bool(words & _ASCII_ORDER) or any(p in query for p in _CJK_ORDER)
        is_ticket = bool(words & _ASCII_TICKET) or any(p in query for p in _CJK_TICKET)
        if is_shipment and has_order_id:
            return "check_shipment"
        if is_order and has_order_id:
            return "check_order"
        if is_ticket:
            return "create_ticket"
        return None

    def execute(
        self,
        tool_name: str,
        query: str,
        user_id: str = "guest",
        session_id: Optional[str] = None,
    ) -> ToolResult:
        if tool_name == "check_order":
            return self._check_order(query)
        if tool_name == "check_shipment":
            return self._check_shipment(query)
        if tool_name == "create_ticket":
            return self._create_ticket(user_id, query, session_id)
        return ToolResult(success=False, message=f"Unknown tool: {tool_name}")

    def _check_order(self, query: str) -> ToolResult:
        match = _ORDER_ID_RE.search(query)
        if not match:
            return ToolResult(success=False, message="No order ID found in your message.")
        order_id = match.group(1).upper()
        order = self.state_store.get_order(order_id)
        if not order:
            return ToolResult(success=False, message=f"Order {order_id} not found.")
        return ToolResult(
            success=True,
            message=(
                f"Order {order['order_id']}: status={order['status']}, "
                f"amount={order['amount']}, items={order['items']}."
            ),
        )

    def _check_shipment(self, query: str) -> ToolResult:
        match = _ORDER_ID_RE.search(query)
        if not match:
            return ToolResult(success=False, message="No order ID found in your message.")
        order_id = match.group(1).upper()
        shipment = self.state_store.get_shipment(order_id)
        if not shipment:
            return ToolResult(success=False, message=f"No shipment found for order {order_id}.")
        return ToolResult(
            success=True,
            message=(
                f"Shipment for {order_id}: carrier={shipment['carrier']}, "
                f"tracking={shipment['tracking_no']}, "
                f"status={shipment['shipping_status']}, ETA={shipment['eta']}."
            ),
        )

    def _create_ticket(
        self, user_id: str, message: str, session_id: Optional[str]
    ) -> ToolResult:
        ticket_id = self.human_interface.create_ticket(
            user_id=user_id,
            initial_message=message,
            session_id=session_id,
        )
        return ToolResult(
            success=True,
            message=f"Ticket #{ticket_id} created. A human agent will follow up shortly.",
        )
