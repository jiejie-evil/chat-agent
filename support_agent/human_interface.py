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
