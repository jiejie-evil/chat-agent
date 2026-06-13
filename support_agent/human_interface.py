from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Ticket:
    user_id: str
    messages: List[Tuple[str, str]] = field(default_factory=list)
    status: str = "open"


class HumanAgentInterface:
    def __init__(self):
        self.tickets: Dict[int, Ticket] = {}

    def create_ticket(self, user_id: str, initial_message: str) -> int:
        ticket_id = len(self.tickets) + 1
        self.tickets[ticket_id] = Ticket(
            user_id=user_id,
            messages=[("user", initial_message)],
        )
        return ticket_id

    def add_agent_response(self, ticket_id: int, message: str) -> None:
        self.tickets[ticket_id].messages.append(("agent", message))

    def list_open(self) -> Dict[int, dict]:
        return {
            ticket_id: {
                "user_id": ticket.user_id,
                "messages": ticket.messages,
                "status": ticket.status,
            }
            for ticket_id, ticket in self.tickets.items()
            if ticket.status == "open"
        }

    def close_ticket(self, ticket_id: int) -> None:
        self.tickets[ticket_id].status = "closed"
