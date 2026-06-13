import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from support_agent.agent import SupportAgent
from support_agent.human_interface import HumanAgentInterface
from support_agent.state_store import StateStore


def _load_doc_paths() -> List[str]:
    project_root = os.path.dirname(__file__)
    data_dir = os.path.join(project_root, "data")
    return [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".md")
    ]


state_store = StateStore()
human_interface = HumanAgentInterface(state_store)
agent = SupportAgent(state_store=state_store, human_interface=human_interface)


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
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    escalated: bool = False
    sources: List[str] = []
    confidence: float = 0.0
    tool_name: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    guardrail_reason: Optional[str] = None


class TicketRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None


class TicketStatusRequest(BaseModel):
    status: str


class UserUpsertRequest(BaseModel):
    display_name: Optional[str] = None
    email: Optional[str] = None
    tier: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/answer", response_model=AskResponse)
def answer_question(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    _ensure_agent_ready()
    result = agent.answer_with_metadata(
        question,
        user_id=payload.user_id,
        session_id=payload.session_id,
    )
    return AskResponse(
        answer=result.answer,
        escalated=result.escalated,
        sources=result.sources,
        confidence=result.confidence,
        tool_name=result.tool_name,
        session_id=result.session_id,
        trace_id=result.trace_id,
        guardrail_reason=result.guardrail_reason,
    )


@app.post("/tickets")
def create_ticket(payload: TicketRequest) -> dict:
    ticket_id = human_interface.create_ticket(
        user_id=payload.user_id,
        initial_message=payload.message,
        session_id=payload.session_id,
    )
    return {"ticket_id": ticket_id, "status": "open"}


@app.get("/tickets")
def list_tickets(status: Optional[str] = None, user_id: Optional[str] = None) -> dict:
    tickets = state_store.list_tickets(status=status, user_id=user_id)
    return {"tickets": tickets}


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: int) -> dict:
    try:
        return state_store.get_ticket(ticket_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/tickets/{ticket_id}")
def update_ticket(ticket_id: int, payload: TicketStatusRequest) -> dict:
    try:
        return state_store.update_ticket_status(ticket_id, payload.status)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/sessions")
def list_sessions(user_id: Optional[str] = None) -> dict:
    return {"sessions": state_store.list_sessions(user_id=user_id)}


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    try:
        return state_store.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/traces")
def list_traces(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 20,
) -> dict:
    return {"traces": state_store.list_traces(session_id=session_id, user_id=user_id, limit=limit)}


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict:
    try:
        return state_store.get_trace(trace_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/users/{user_id}")
def get_user(user_id: str) -> dict:
    try:
        return state_store.get_user(user_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/users/{user_id}")
def upsert_user(user_id: str, payload: UserUpsertRequest) -> dict:
    return state_store.update_user(
        user_id=user_id,
        display_name=payload.display_name,
        email=payload.email,
        tier=payload.tier,
    )
