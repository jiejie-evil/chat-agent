# LangGraph Support Agent

This project uses `LangGraph` as the orchestration layer and is being built in staged milestones.

## Current milestone

Step 1 is the active baseline:

1. model
2. prompt
3. RAG

The goal of this step is to answer customer support questions stably before adding tools, state, and guardrails.

## Step 1 architecture

The current workflow is:

1. load support documents
2. split documents into retrieval-ready chunks
3. retrieve the most relevant chunks
4. assess retrieval confidence
5. generate a grounded answer or escalate to a human agent

The graph is implemented in [support_agent/agent.py](/D:/code/langchain-support-agent/support_agent/agent.py).

## Why this is more stable now

- Retrieval uses character n-gram TF-IDF, which works much better for Chinese support queries than default word tokenization.
- Q/A style source files are split into smaller semantic blocks instead of one oversized chunk.
- The API now returns `sources`, `confidence`, and `escalated`, so the RAG layer is observable and reviewable.
- Low-confidence retrieval is explicitly routed to human escalation.

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
python -m support_agent.app --demo
python -m uvicorn support_agent.api:app --reload
```

## Step-by-step roadmap

1. model + prompt + RAG
2. tools for orders, shipping, and ticket creation
3. state for sessions, identity, and ticket flow
4. guardrails and human escalation policy
5. observability and offline evaluation
6. deployment hardening
