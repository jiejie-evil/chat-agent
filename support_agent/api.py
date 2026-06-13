import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from support_agent.agent import SupportAgent
from support_agent.human_interface import HumanAgentInterface


def _load_doc_paths() -> List[str]:
    project_root = os.path.dirname(__file__)
    data_dir = os.path.join(project_root, "data")
    return [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".md")
    ]


human_interface = HumanAgentInterface()
agent = SupportAgent(human_interface=human_interface)


def _ensure_agent_ready() -> None:
    if not agent.is_ready():
        agent.ingest(_load_doc_paths())


@asynccontextmanager
async def lifespan(_: FastAPI):
    _ensure_agent_ready()
    yield


app = FastAPI(title="Support Agent API", version="0.2.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    user_id: str = "guest"


class AskResponse(BaseModel):
    answer: str
    escalated: bool = False
    sources: List[str] = []
    confidence: float = 0.0
    tool_name: Optional[str] = None


class TicketRequest(BaseModel):
    user_id: str
    message: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/answer", response_model=AskResponse)
def answer_question(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    _ensure_agent_ready()
    result = agent.answer_with_metadata(question, user_id=payload.user_id)
    return AskResponse(
        answer=result.answer,
        escalated=result.escalated,
        sources=result.sources,
        confidence=result.confidence,
        tool_name=result.tool_name,
    )


@app.post("/tickets")
def create_ticket(payload: TicketRequest) -> dict:
    ticket_id = human_interface.create_ticket(
        user_id=payload.user_id,
        initial_message=payload.message,
    )
    return {"ticket_id": ticket_id, "status": "open"}


@app.get("/tickets")
def list_tickets() -> dict:
    return human_interface.list_open()
