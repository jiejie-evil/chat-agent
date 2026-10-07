# LangGraph Support Agent

An enterprise-style customer support agent built with `LangGraph`, `FastAPI`, and a `RAG retrieval` pipeline.

It supports:

- RAG-based grounded answering
- order and shipping tool calls
- ticket creation and escalation
- persistent sessions, users, tickets, and traces
- guardrails for sensitive content, unauthorized access, and low-confidence fallback
- local deployment with `Docker` and `docker-compose`

## What this project demonstrates

This project is positioned as an `AI application engineering` sample rather than a pure model demo.

Core capabilities:

1. `RAG retrieval` over support knowledge documents
2. `tool calling` for order lookup, shipping lookup, and ticket creation
3. `state management` for session history, identity, and ticket lifecycle
4. `guardrails` for sensitive content, access control, and escalation
5. `observability` with per-request traces and stored execution events
6. `offline eval` for retrieval, tools, and guardrail behavior
7. `deployment baseline` with config management, readiness checks, and containerization

## Architecture

The request flow is:

1. load RAG knowledge documents
2. route tool-like queries before retrieval
3. run guardrails before answering
4. retrieve relevant chunks when needed
5. assess confidence
6. answer or escalate
7. persist session and trace state in SQLite

Key files:

- [agent.py](/D:/code/langchain-support-agent/support_agent/agent.py)
- [api.py](/D:/code/langchain-support-agent/support_agent/api.py)
- [state_store.py](/D:/code/langchain-support-agent/support_agent/state_store.py)
- [guardrails.py](/D:/code/langchain-support-agent/support_agent/guardrails.py)
- [observability.py](/D:/code/langchain-support-agent/support_agent/observability.py)
- [config.py](/D:/code/langchain-support-agent/support_agent/config.py)
- [serve.py](/D:/code/langchain-support-agent/support_agent/serve.py)

## Data sources

RAG knowledge and support seed data live here:

- [rag_knowledge_base.md](/D:/code/langchain-support-agent/support_agent/data/rag_knowledge_base.md)
- [eval_dataset.json](/D:/code/langchain-support-agent/support_agent/data/eval_dataset.json)

## Local run

```powershell
Set-Location -Path 'D:\code\langchain-support-agent'
.\.venv\Scripts\Activate.ps1
python -m pip install -r support_agent\requirements.txt
python -m pytest support_agent\tests -q -p no:cacheprovider
python -m support_agent.eval_runner
python -m support_agent.serve
```

Health and readiness:

- `GET /health`
- `GET /ready`

## Environment variables

See [.env.example](/D:/code/langchain-support-agent/.env.example).

Important values:

- `APP_NAME`
- `APP_VERSION`
- `APP_ENV`
- `APP_HOST`
- `APP_PORT`
- `SUPPORT_AGENT_DB_PATH`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

## Docker deployment

Build and run with Docker:

```powershell
docker build -t support-agent .
docker run --rm -p 8000:8000 --env-file .env support-agent
```

Run with Compose:

```powershell
docker compose up --build
```

Deployment files:

- [Dockerfile](/D:/code/langchain-support-agent/Dockerfile)
- [docker-compose.yml](/D:/code/langchain-support-agent/docker-compose.yml)
- [.dockerignore](/D:/code/langchain-support-agent/.dockerignore)

## API endpoints

- `POST /answer`
- `POST /tickets`
- `GET /tickets`
- `GET /tickets/{ticket_id}`
- `PATCH /tickets/{ticket_id}`
- `GET /sessions`
- `GET /sessions/{session_id}`
- `GET /traces`
- `GET /traces/{trace_id}`
- `GET /users/{user_id}`
- `PUT /users/{user_id}`
- `GET /health`
- `GET /ready`

Example request:

```json
POST /answer
{
  "question": "Please check order ORD-1001 status",
  "user_id": "user-1",
  "session_id": "session-001"
}
```

Example response:

```json
{
  "answer": "Order ORD-1001 is currently paid. Items: wireless headset. Amount: HKD 299.",
  "escalated": false,
  "sources": ["check_order"],
  "confidence": 1.0,
  "tool_name": "check_order",
  "session_id": "session-001",
  "trace_id": "trace-xxxx",
  "guardrail_reason": null
}
```

## Eval and verification

Run tests:

```powershell
python -m pytest support_agent\tests -q -p no:cacheprovider
```

Run offline eval:

```powershell
python -m support_agent.eval_runner
```

The evaluator reports dialogue pass rate, retrieval recall, average/p95 latency,
fallback latency, and false-escalation rate. It exits non-zero when the PRD
targets are not met (87% dialogue pass rate, 91% retrieval recall, 2.5s average
latency, 500ms fallback latency, and 8% false escalation rate).

## Resume-ready highlights

This project is suitable for resume positioning such as:

- Built a `LangGraph`-based customer support agent with `RAG`, tool calling, state persistence, guardrails, and traceable execution.
- Developed a `FastAPI` service with persistent sessions, ticket lifecycle management, and queryable observability data in SQLite.
- Added offline evaluation, readiness checks, and Docker-based deployment for a more production-like AI application workflow.
