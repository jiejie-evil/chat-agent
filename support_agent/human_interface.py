from typing import Optional

from support_agent.state_store import StateStore


class HumanAgentInterface:
    def __init__(self, state_store: StateStore):
        self.state_store = state_store

    def create_ticket(
        self,
        user_id: str,
        initial_message: str,
        session_id: Optional[str] = None,
    ) -> int:
        return self.state_store.create_ticket(
            user_id=user_id,
            initial_message=initial_message,
            session_id=session_id,
        )

    def add_agent_response(self, ticket_id: int, message: str) -> None:
        self.state_store.add_ticket_message(ticket_id, "agent", message)

    def list_open(self):
        return {
            ticket["ticket_id"]: ticket
            for ticket in self.state_store.list_tickets(status="open")
        }

    def close_ticket(self, ticket_id: int) -> None:
        self.state_store.update_ticket_status(ticket_id, "closed")
