# LangGraph Support Agent

This project uses `LangGraph` as the orchestration layer and is being built in staged milestones.

## Current milestone

Step 5 is now complete:

1. model
2. prompt
3. RAG
4. tools for orders, shipping, and ticket creation
5. persistent state for sessions, identity, and ticket flow
6. guardrails for sensitive content, unauthorized access, and low-confidence escalation
7. observability and offline evaluation

The current baseline can answer support questions, call operational tools, persist support state across restarts, enforce core safety policies, and expose traces and eval results for review.

## Current architecture

The current workflow is:

1. load support documents
2. split documents into retrieval-ready chunks
3. route tool-style requests before retrieval
4. retrieve the most relevant chunks
5. assess retrieval confidence
6. generate a grounded answer or escalate to a human agent
7. persist session history and support state in SQLite
8. record structured traces and events for each request

The graph is implemented in [support_agent/agent.py](/D:/code/langchain-support-agent/support_agent/agent.py).

## Why this is more stable now

- Retrieval uses character n-gram TF-IDF, which works much better for Chinese support queries than default word tokenization.
- Q/A style source files are split into smaller semantic blocks instead of one oversized chunk.
- The API now returns `sources`, `confidence`, and `escalated`, so the RAG layer is observable and reviewable.
- Low-confidence retrieval is explicitly routed to human escalation.
- Sessions, users, tickets, orders, and shipment data are now persisted in SQLite.
- Ticket flow and conversation history are queryable and recoverable through the API.
- Sensitive payment secrets and identity secrets are blocked before retrieval or tool execution.
- Order and shipping lookups now enforce sign-in and ownership checks instead of allowing anonymous access.
- Each request now produces a trace with step-level events, outcomes, and timing.
- An offline eval dataset and runner now verify RAG, tools, and guardrails together.

## Project structure

```text
langchain-support-agent/
├─ support_agent/
│  ├─ agent.py
│  ├─ api.py
│  ├─ app.py
│  ├─ human_interface.py
│  ├─ ingest.py
│  ├─ requirements.txt
│  ├─ data/
│  │  └─ customer_faq.md
│  └─ tests/
│     └─ test_e2e.py
├─ .env.example
├─ .gitignore
└─ README.md
```

## Run locally

```powershell
Set-Location -Path 'D:\code\langchain-support-agent'
.\.venv\Scripts\Activate.ps1
python -m pip install -r support_agent\requirements.txt
python -m pytest support_agent\tests -q -p no:cacheprovider
python -m support_agent.eval_runner
python -m support_agent.app --demo
python -m uvicorn support_agent.api:app --reload
```

## How to use the agent

1. Start the API with `python -m uvicorn support_agent.api:app --reload`
2. Create or update a user with `PUT /users/{user_id}` when you want an identified customer
3. Ask questions through `POST /answer` with `question`, `user_id`, and an optional `session_id`
4. Use the returned `session_id` and `trace_id` to inspect history and decision traces
5. Create and manage manual follow-up through the ticket endpoints when escalation is needed

Example answer request:

```json
POST /answer
{
  "question": "请帮我查询订单 ORD-1001 的状态",
  "user_id": "user-1",
  "session_id": "session-001"
}
```

Example response:

```json
{
  "answer": "Order ORD-1001 is currently paid. Items: wireless headset. Amount: HKD 299.",
  "escalated": false,
  "sources": ["lookup_order"],
  "confidence": 1.0,
  "tool_name": "lookup_order",
  "session_id": "session-001",
  "trace_id": "trace-xxxx",
  "guardrail_reason": null
}
```

## State API

- `POST /answer` supports `user_id` and optional `session_id`, and returns the active `session_id`
- `GET /sessions/{session_id}` returns persisted conversation history
- `GET /users/{user_id}` and `PUT /users/{user_id}` query and update user identity
- `GET /tickets`, `GET /tickets/{ticket_id}`, and `PATCH /tickets/{ticket_id}` query and update ticket flow

## Observability

- `POST /answer` now returns `trace_id` and `guardrail_reason`
- `GET /traces` lists recent traces by `session_id` or `user_id`
- `GET /traces/{trace_id}` returns the full event timeline for a request
- Trace data is stored in SQLite alongside sessions and tickets

## Guardrails

- Sensitive content is blocked before the request reaches retrieval or tools
- Unauthorized order access is escalated with a clear policy reason
- Low-confidence retrieval is escalated with a structured `guardrail_reason`
- `POST /answer` now returns `guardrail_reason` for review and observability

## Offline eval

- Eval cases live in [support_agent/data/eval_dataset.json](/D:/code/langchain-support-agent/support_agent/data/eval_dataset.json)
- Run `python -m support_agent.eval_runner`
- The runner checks expected snippets, escalation behavior, tool routing, and guardrail reasons

## Step-by-step roadmap

1. model + prompt + RAG
2. tools for orders, shipping, and ticket creation
3. state for sessions, identity, and ticket flow
4. guardrails and human escalation policy
5. observability and offline evaluation
6. deployment hardening
