import os
from contextlib import asynccontextmanager
import time
from typing import List, Optional

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from support_agent.agent import SupportAgent
from support_agent.auth import AuthContext, authenticate_api_key
from support_agent.config import get_settings
from support_agent.errors import (
    AuthenticationInvalidError,
    AuthenticationRequiredError,
    RateLimitedError,
    ServiceError,
)
from support_agent.human_interface import HumanAgentInterface
from support_agent.rate_limit import InMemoryRateLimiter
from support_agent.resilience import log_event
from support_agent.state_store import StateStore


def _load_doc_paths() -> List[str]:
    project_root = os.path.dirname(__file__)
    data_dir = os.path.join(project_root, "data")
    return [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".md")
    ]


settings = get_settings()
state_store = StateStore(db_path=settings.database_path)
human_interface = HumanAgentInterface(state_store)
agent = SupportAgent(state_store=state_store, human_interface=human_interface)
answer_rate_limiter = InMemoryRateLimiter(
    max_requests=settings.support_agent_rate_limit_requests,
    window_seconds=settings.support_agent_rate_limit_window_seconds,
)


def _ensure_agent_ready() -> None:
    if not agent.is_ready():
        agent.ingest(_load_doc_paths())


@asynccontextmanager
async def lifespan(_: FastAPI):
    _ensure_agent_ready()
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    started_at = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        log_event(
            {
                "trace_id": getattr(request.state, "trace_id", None),
                "session_id": getattr(request.state, "session_id", None),
                "user_id": getattr(request.state, "user_id", None),
                "path": request.url.path,
                "status_code": response.status_code if response else None,
                "duration_ms": duration_ms,
                "rate_limit_key_type": getattr(request.state, "rate_limit_key_type", None),
                "auth_subject": getattr(request.state, "auth_subject", None),
                "event_type": "request_completed",
            }
        )


@app.exception_handler(ServiceError)
async def service_error_handler(_: Request, exc: ServiceError) -> JSONResponse:
    headers = {}
    retry_after = (exc.details or {}).get("retry_after_seconds")
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return JSONResponse(status_code=exc.status_code, content=exc.to_response(), headers=headers)


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


def _validate_user_id(user_id: str) -> None:
    normalized = user_id.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="user_id cannot be empty")
    if not normalized.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(
            status_code=400,
            detail="user_id may contain only letters, numbers, hyphens, and underscores",
        )


def _authenticate_request(x_api_key: Optional[str]) -> AuthContext:
    try:
        auth = authenticate_api_key(x_api_key, settings)
        log_event(
            {
                "trace_id": None,
                "event_type": "auth_checked",
                "auth_subject": auth.subject,
                "status_code": 200,
            }
        )
        return auth
    except (AuthenticationRequiredError, AuthenticationInvalidError) as exc:
        log_event(
            {
                "trace_id": None,
                "event_type": "auth_checked",
                "auth_subject": None,
                "status_code": exc.status_code,
                "error_code": exc.error_code,
            }
        )
        raise


def _enforce_answer_rate_limit(auth: AuthContext, user_id: str) -> tuple[str, str]:
    bucket_key = f"user:{user_id}"
    bucket_type = "user_id"
    if user_id == "guest":
        bucket_key = f"apikey:{auth.key_fingerprint}"
        bucket_type = "api_key"

    decision = answer_rate_limiter.check(bucket_key, bucket_type=bucket_type)
    log_event(
        {
            "trace_id": None,
            "event_type": "rate_limit_checked",
            "auth_subject": auth.subject,
            "rate_limit_key_type": bucket_type,
            "allowed": decision.allowed,
        }
    )
    if not decision.allowed:
        log_event(
            {
                "trace_id": None,
                "event_type": "rate_limit_rejected",
                "auth_subject": auth.subject,
                "rate_limit_key_type": bucket_type,
                "retry_after_seconds": decision.retry_after_seconds,
                "status_code": 429,
            }
        )
        raise RateLimitedError(decision.retry_after_seconds)
    return bucket_key, bucket_type


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": settings.app_env}


@app.get("/ready")
def ready() -> dict:
    try:
        _ensure_agent_ready()
        return {
            "status": "ready",
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "knowledge_loaded": agent.is_ready(),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"agent is not ready: {exc}") from exc


@app.post("/answer", response_model=AskResponse)
def answer_question(
    payload: AskRequest,
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> AskResponse:
    auth = _authenticate_request(x_api_key)
    request.state.auth_subject = auth.subject
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")
    _validate_user_id(payload.user_id)
    request.state.user_id = payload.user_id

    bucket_key, bucket_type = _enforce_answer_rate_limit(auth, payload.user_id)
    request.state.rate_limit_key_type = bucket_type
    request.state.rate_limit_key = bucket_key

    _ensure_agent_ready()
    result = agent.answer_with_metadata(
        question,
        user_id=payload.user_id,
        session_id=payload.session_id,
    )
    request.state.trace_id = result.trace_id
    request.state.session_id = result.session_id
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


@app.post("/answer/stream")
async def stream_answer_question(
    payload: AskRequest,
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> StreamingResponse:
    auth = _authenticate_request(x_api_key)
    request.state.auth_subject = auth.subject
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")
    _validate_user_id(payload.user_id)
    request.state.user_id = payload.user_id

    bucket_key, bucket_type = _enforce_answer_rate_limit(auth, payload.user_id)
    request.state.rate_limit_key_type = bucket_type
    request.state.rate_limit_key = bucket_key

    _ensure_agent_ready()

    _STREAM_DONE = object()

    async def event_stream():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            gen = agent.stream_answer(
                question,
                user_id=payload.user_id,
                session_id=payload.session_id,
            )
            try:
                while True:
                    token = await loop.run_in_executor(
                        pool, next, gen, _STREAM_DONE
                    )
                    if token is _STREAM_DONE:
                        break
                    yield f"data: {token}\n\n"
            finally:
                gen.close()
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/tickets")
def create_ticket(
    payload: TicketRequest,
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    auth = _authenticate_request(x_api_key)
    _validate_user_id(payload.user_id)
    request.state.auth_subject = auth.subject
    request.state.user_id = payload.user_id
    ticket_id = human_interface.create_ticket(
        user_id=payload.user_id,
        initial_message=payload.message,
        session_id=payload.session_id,
    )
    return {"ticket_id": ticket_id, "status": "open"}


@app.get("/tickets")
def list_tickets(
    request: Request,
    status: Optional[str] = None,
    user_id: Optional[str] = None,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    auth = _authenticate_request(x_api_key)
    request.state.auth_subject = auth.subject
    if user_id is not None:
        _validate_user_id(user_id)
        request.state.user_id = user_id
    tickets = state_store.list_tickets(status=status, user_id=user_id)
    return {"tickets": tickets}


@app.get("/tickets/{ticket_id}")
def get_ticket(
    ticket_id: int,
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    auth = _authenticate_request(x_api_key)
    request.state.auth_subject = auth.subject
    try:
        return state_store.get_ticket(ticket_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/tickets/{ticket_id}")
def update_ticket(
    ticket_id: int,
    payload: TicketStatusRequest,
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    auth = _authenticate_request(x_api_key)
    request.state.auth_subject = auth.subject
    try:
        return state_store.update_ticket_status(ticket_id, payload.status)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/sessions")
def list_sessions(
    request: Request,
    user_id: Optional[str] = None,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    auth = _authenticate_request(x_api_key)
    request.state.auth_subject = auth.subject
    if user_id is not None:
        _validate_user_id(user_id)
        request.state.user_id = user_id
    return {"sessions": state_store.list_sessions(user_id=user_id)}


@app.get("/sessions/{session_id}")
def get_session(
    session_id: str,
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    auth = _authenticate_request(x_api_key)
    request.state.auth_subject = auth.subject
    try:
        return state_store.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/traces")
def list_traces(
    request: Request,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 20,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    auth = _authenticate_request(x_api_key)
    request.state.auth_subject = auth.subject
    if user_id is not None:
        _validate_user_id(user_id)
        request.state.user_id = user_id
    return {"traces": state_store.list_traces(session_id=session_id, user_id=user_id, limit=limit)}


@app.get("/traces/{trace_id}")
def get_trace(
    trace_id: str,
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    auth = _authenticate_request(x_api_key)
    request.state.auth_subject = auth.subject
    try:
        return state_store.get_trace(trace_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/users/{user_id}")
def get_user(
    user_id: str,
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    auth = _authenticate_request(x_api_key)
    request.state.auth_subject = auth.subject
    _validate_user_id(user_id)
    request.state.user_id = user_id
    try:
        return state_store.get_user(user_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/users/{user_id}")
def upsert_user(
    user_id: str,
    payload: UserUpsertRequest,
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    auth = _authenticate_request(x_api_key)
    request.state.auth_subject = auth.subject
    _validate_user_id(user_id)
    request.state.user_id = user_id
    return state_store.update_user(
        user_id=user_id,
        display_name=payload.display_name,
        email=payload.email,
        tier=payload.tier,
    )
