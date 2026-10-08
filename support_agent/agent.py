import os
import re
from dataclasses import dataclass
from typing import List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from support_agent.config import get_settings
from support_agent.errors import ServiceError, UpstreamUnavailableError
from support_agent.guardrails import Guardrails
from support_agent.human_interface import HumanAgentInterface
from support_agent.observability import Observability
from support_agent.resilience import run_with_retries, run_with_timeout
from support_agent.state_store import StateStore
from support_agent.tools import SupportToolbox, ToolResult


SYSTEM_PROMPT = (
    "You are an enterprise customer support assistant. "
    "Answer only from the retrieved knowledge. "
    "If the retrieved knowledge is insufficient, explicitly say the issue "
    "should be escalated to a human agent. "
    "Be concise, accurate, and operationally useful."
)


@dataclass
class RetrievedChunk:
    content: str
    source: str
    chunk_id: int
    score: float


@dataclass
class AnswerResult:
    answer: str
    sources: List[str]
    confidence: float
    escalated: bool
    tool_name: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    guardrail_reason: Optional[str] = None
    failure_reason: Optional[str] = None


class AgentState(TypedDict, total=False):
    query: str
    user_id: str
    session_id: str
    trace_id: str
    guardrail_reason: Optional[str]
    retrieved: List[RetrievedChunk]
    context: str
    confidence: float
    tool_name: Optional[str]
    tool_result: Optional[ToolResult]
    answer: str
    sources: List[str]
    escalated: bool


class SupportAgent:
    """A LangGraph-based RAG and tool baseline for customer support."""

    def __init__(
        self,
        chunk_size: int = 220,
        chunk_overlap: int = 40,
        top_k: int = 3,
        min_score: float = 0.12,
        state_store: Optional[StateStore] = None,
        human_interface: Optional[HumanAgentInterface] = None,
        chroma_persist_dir: Optional[str] = None,
    ):
        self.settings = get_settings()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.min_score = min_score
        self.chroma_persist_dir = chroma_persist_dir or self.settings.chroma_persist_dir
        self.chunks: List[RetrievedChunk] = []
        self._embedder = None
        self._collection = None
        self.openai_client: Optional[OpenAI] = None
        self.state_store = state_store or StateStore()
        self.human_interface = human_interface or HumanAgentInterface(self.state_store)
        self.toolbox = SupportToolbox(self.human_interface, self.state_store)
        self.guardrails = Guardrails()
        self.observability = Observability(self.state_store)
        self.graph = self._build_graph()

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)

    def _ensure_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.settings.embedding_model)
        return self._embedder

    def ingest(self, doc_paths: List[str]) -> None:
        import chromadb

        embedder = self._ensure_embedder()
        client = chromadb.PersistentClient(path=self.chroma_persist_dir)
        try:
            client.delete_collection("knowledge")
        except Exception:
            pass
        self._collection = client.create_collection(
            "knowledge", metadata={"hnsw:space": "cosine"}
        )

        chunk_texts: List[str] = []
        ids: List[str] = []
        metadatas: List[dict] = []
        self.chunks = []

        for path in doc_paths:
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()

            for chunk_id, chunk in enumerate(self._split_text(text)):
                cleaned = self._normalize_text(chunk)
                if not cleaned:
                    continue
                source = os.path.basename(path)
                uid = f"{source}::{chunk_id}"
                chunk_texts.append(cleaned)
                ids.append(uid)
                metadatas.append({"source": source, "chunk_id": chunk_id})
                self.chunks.append(
                    RetrievedChunk(
                        content=cleaned,
                        source=source,
                        chunk_id=chunk_id,
                        score=0.0,
                    )
                )

        if not chunk_texts:
            raise ValueError("No valid documents were ingested.")

        embeddings = embedder.encode(chunk_texts).tolist()
        self._collection.add(
            documents=chunk_texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        if self._collection is None or self._embedder is None:
            raise RuntimeError("Please ingest documents before querying the agent.")

        top_k = k or self.top_k
        query_embedding = self._embedder.encode([query]).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self._collection.count()),
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        chunks: List[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            score = 1.0 - float(distance)
            if score <= 0:
                continue
            chunks.append(
                RetrievedChunk(
                    content=document,
                    source=metadata["source"],
                    chunk_id=int(metadata["chunk_id"]),
                    score=score,
                )
            )
        if chunks and not any(self._has_grounding_overlap(query, chunk.content) for chunk in chunks):
            return []
        return chunks

    def _has_grounding_overlap(self, query: str, content: str) -> bool:
        query_terms = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", query.lower()))
        content_terms = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", content.lower()))
        shared = query_terms & content_terms
        latin_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        if latin_terms and not (latin_terms & content_terms):
            return False
        return len(shared) >= 1

    def answer(
        self,
        query: str,
        user_id: str = "guest",
        session_id: Optional[str] = None,
    ) -> str:
        return self.answer_with_metadata(query, user_id=user_id, session_id=session_id).answer

    def answer_with_metadata(
        self,
        query: str,
        user_id: str = "guest",
        session_id: Optional[str] = None,
    ) -> AnswerResult:
        session = self.state_store.ensure_session(user_id=user_id, session_id=session_id)
        self.state_store.append_session_message(session["session_id"], "user", query)
        trace = self.observability.start_trace(
            user_id=user_id,
            session_id=session["session_id"],
            query=query,
        )
        try:
            result = run_with_timeout(
                lambda: self.graph.invoke(
                    {
                        "query": query,
                        "user_id": user_id,
                        "session_id": session["session_id"],
                        "trace_id": trace.trace_id,
                    }
                ),
                timeout_seconds=self.settings.support_agent_request_timeout_seconds,
            )
        except ServiceError as exc:
            if exc.error_code == "request_timeout":
                self.observability.log_event(
                    trace.trace_id,
                    "request_timed_out",
                    {"timeout_seconds": self.settings.support_agent_request_timeout_seconds},
                )
            self.observability.finish_trace(
                trace=trace,
                answer="",
                confidence=0.0,
                escalated=True,
                tool_name=None,
                guardrail_reason=None,
                failure_reason=exc.error_code,
            )
            raise
        except Exception:
            self.observability.finish_trace(
                trace=trace,
                answer="",
                confidence=0.0,
                escalated=True,
                tool_name=None,
                guardrail_reason=None,
                failure_reason="unexpected_error",
            )
            raise

        self.state_store.append_session_message(session["session_id"], "assistant", result["answer"])
        self.observability.finish_trace(
            trace=trace,
            answer=result["answer"],
            confidence=result["confidence"],
            escalated=result["escalated"],
            tool_name=result.get("tool_name"),
            guardrail_reason=result.get("guardrail_reason"),
            failure_reason=None,
        )
        return AnswerResult(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            escalated=result["escalated"],
            tool_name=result.get("tool_name"),
            session_id=session["session_id"],
            trace_id=trace.trace_id,
            guardrail_reason=result.get("guardrail_reason"),
            failure_reason=None,
        )

    def stream_answer(
        self,
        query: str,
        user_id: str = "guest",
        session_id: Optional[str] = None,
    ):
        """Yield answer tokens incrementally. Bypasses the graph for SSE compatibility."""
        session = self.state_store.ensure_session(user_id=user_id, session_id=session_id)
        self.state_store.append_session_message(session["session_id"], "user", query)
        trace = self.observability.start_trace(
            user_id=user_id,
            session_id=session["session_id"],
            query=query,
        )

        collected: List[str] = []

        def emit(text: str):
            collected.append(text)
            return text

        escalated = False
        tool_name: Optional[str] = None
        guardrail_reason: Optional[str] = None
        confidence = 1.0

        try:
            decision = self.guardrails.inspect_query(query)
            if decision.action == "block":
                guardrail_reason = decision.reason
                escalated = True
                yield emit(decision.message or "Request blocked.")
                return

            tool_name = self.toolbox.detect_tool(query)
            if tool_name:
                tool_result = self.toolbox.execute(
                    tool_name, query, user_id=user_id, session_id=session["session_id"]
                )
                escalated = tool_name == "create_ticket" or not tool_result.success
                confidence = 1.0 if tool_result.success else 0.0
                yield emit(tool_result.message)
                return

            retrieved = self.retrieve(query)
            confidence = retrieved[0].score if retrieved else 0.0
            assessment = self.guardrails.evaluate_retrieval_confidence(confidence, self.min_score)
            if assessment.action == "escalate" or not retrieved:
                escalated = True
                guardrail_reason = assessment.reason
                yield emit(
                    assessment.message
                    or (
                        "I could not find enough grounded knowledge for this question. "
                        "Please hand the conversation to a human agent."
                    )
                )
                return

            context = self.build_context(retrieved)
            if self.openai_client:
                for token in self._stream_with_openai(query, context, trace.trace_id):
                    yield emit(token)
            else:
                yield emit(self._generate_fallback_answer(retrieved))
        finally:
            answer = "".join(collected)
            self.state_store.append_session_message(session["session_id"], "assistant", answer)
            self.observability.finish_trace(
                trace=trace,
                answer=answer,
                confidence=confidence,
                escalated=escalated,
                tool_name=tool_name,
                guardrail_reason=guardrail_reason,
                failure_reason=None,
            )

    def _stream_with_openai(self, query: str, context: str, trace_id: str):
        with self.openai_client.responses.stream(
            model=self.settings.openai_model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Retrieved knowledge:\n{context}\n\n"
                        f"Customer question:\n{query}"
                    ),
                },
            ],
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta

    def is_ready(self) -> bool:
        return self._collection is not None and self._embedder is not None

    def build_context(self, retrieved: List[RetrievedChunk]) -> str:
        return "\n\n".join(
            (
                f"Source: {item.source}\n"
                f"Chunk: {item.chunk_id}\n"
                f"Score: {item.score:.4f}\n"
                f"Content: {item.content}"
            )
            for item in retrieved
        )

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("guardrails", self._guardrails_node)
        graph.add_node("route", self._route_node)
        graph.add_node("tool", self._tool_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("assess", self._assess_node)
        graph.add_node("generate", self._generate_node)
        graph.add_edge(START, "guardrails")
        graph.add_conditional_edges(
            "guardrails",
            self._after_guardrails,
            {"blocked": END, "route": "route"},
        )
        graph.add_conditional_edges(
            "route",
            self._after_route,
            {"tool": "tool", "retrieve": "retrieve"},
        )
        graph.add_edge("tool", END)
        graph.add_edge("retrieve", "assess")
        graph.add_edge("assess", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def _guardrails_node(self, state: AgentState) -> AgentState:
        decision = self.guardrails.inspect_query(state["query"])
        self.observability.log_event(
            state["trace_id"],
            "guardrails_checked",
            {
                "action": decision.action,
                "reason": decision.reason,
            },
        )
        if decision.action == "block":
            return {
                "answer": decision.message or "Request blocked.",
                "sources": ["guardrails"],
                "confidence": 1.0,
                "escalated": True,
                "tool_name": None,
                "guardrail_reason": decision.reason,
            }
        return {"guardrail_reason": None}

    def _after_guardrails(self, state: AgentState) -> str:
        return "blocked" if state.get("guardrail_reason") == "sensitive_content" else "route"

    def _route_node(self, state: AgentState) -> AgentState:
        tool_name = self.toolbox.detect_tool(state["query"])
        self.observability.log_event(
            state["trace_id"],
            "route_selected",
            {"tool_name": tool_name or "", "path": "tool" if tool_name else "retrieve"},
        )
        return {"tool_name": tool_name}

    def _after_route(self, state: AgentState) -> str:
        return "tool" if state.get("tool_name") else "retrieve"

    def _tool_node(self, state: AgentState) -> AgentState:
        tool_name = state.get("tool_name")
        user_id = state.get("user_id", "guest")
        session_id = state.get("session_id")
        if not tool_name:
            return {
                "answer": "No tool selected.",
                "sources": [],
                "confidence": 1.0,
                "escalated": False,
                "tool_name": None,
            }

        tool_result = self.toolbox.execute(
            tool_name,
            state["query"],
            user_id=user_id,
            session_id=session_id,
        )
        escalated = tool_name == "create_ticket"
        guardrail_reason = None
        if not tool_result.success:
            decision = self.guardrails.evaluate_tool_result(
                success=tool_result.success,
                message=tool_result.message,
            )
            if decision.reason:
                guardrail_reason = decision.reason
                escalated = decision.action == "escalate"
            tool_message = tool_result.message.lower()
            if "not authorized" in tool_message or "sign in" in tool_message or "does not belong" in tool_message:
                guardrail_reason = "unauthorized_access"
                escalated = True
        self.observability.log_event(
            state["trace_id"],
            "tool_executed",
            {
                "tool_name": tool_name,
                "success": tool_result.success,
                "guardrail_reason": guardrail_reason,
            },
        )
        return {
            "tool_result": tool_result,
            "answer": tool_result.message,
            "sources": [tool_name],
            "confidence": 1.0 if tool_result.success else 0.0,
            "escalated": escalated,
            "tool_name": tool_name,
            "guardrail_reason": guardrail_reason,
        }

    def _retrieve_node(self, state: AgentState) -> AgentState:
        query = state["query"]
        retrieved = self.retrieve(query)
        context = self.build_context(retrieved) if retrieved else ""
        sources = [f"{item.source}#{item.chunk_id}" for item in retrieved]
        confidence = retrieved[0].score if retrieved else 0.0
        self.observability.log_event(
            state["trace_id"],
            "retrieval_completed",
            {
                "retrieved_count": len(retrieved),
                "confidence": confidence,
                "sources": sources,
            },
        )
        return {
            "retrieved": retrieved,
            "context": context,
            "confidence": confidence,
            "sources": sources,
        }

    def _assess_node(self, state: AgentState) -> AgentState:
        confidence = state.get("confidence", 0.0)
        decision = self.guardrails.evaluate_retrieval_confidence(confidence, self.min_score)
        escalated = decision.action == "escalate"
        self.observability.log_event(
            state["trace_id"],
            "retrieval_assessed",
            {
                "confidence": confidence,
                "threshold": self.min_score,
                "escalated": escalated,
                "reason": decision.reason,
            },
        )
        return {"escalated": escalated, "guardrail_reason": decision.reason}

    def _generate_node(self, state: AgentState) -> AgentState:
        query = state["query"]
        retrieved = state.get("retrieved", [])
        confidence = state.get("confidence", 0.0)
        sources = state.get("sources", [])
        escalated = state.get("escalated", False)
        guardrail_reason = state.get("guardrail_reason")

        if escalated or not retrieved:
            self.observability.log_event(
                state["trace_id"],
                "answer_escalated",
                {
                    "reason": guardrail_reason or "no_retrieval",
                    "confidence": confidence,
                },
            )
            return {
                "answer": (
                    "I could not find enough grounded knowledge for this question. "
                    "Please hand the conversation to a human agent."
                    if guardrail_reason != "low_confidence"
                    else self.guardrails.evaluate_retrieval_confidence(
                        confidence,
                        self.min_score,
                    ).message
                ),
                "sources": sources,
                "confidence": confidence,
                "escalated": True,
                "tool_name": None,
                "guardrail_reason": guardrail_reason,
            }

        context = state["context"]
        if self.openai_client:
            answer = self._generate_with_openai(query, context, state["trace_id"])
        else:
            answer = self._generate_fallback_answer(retrieved)
        self.observability.log_event(
            state["trace_id"],
            "answer_generated",
            {
                "escalated": False,
                "used_openai": self.openai_client is not None,
                "source_count": len(sources),
            },
        )

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "escalated": False,
            "tool_name": None,
            "guardrail_reason": None,
        }

    def _generate_with_openai(self, query: str, context: str, trace_id: str) -> str:
        def invoke_once() -> str:
            def do_call() -> str:
                response = self.openai_client.responses.create(
                    model=self.settings.openai_model,
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Retrieved knowledge:\n{context}\n\n"
                                f"Customer question:\n{query}"
                            ),
                        },
                    ],
                )
                return response.output_text.strip()

            return run_with_timeout(
                do_call,
                timeout_seconds=self.settings.support_agent_openai_timeout_seconds,
            )

        try:
            return run_with_retries(
                invoke_once,
                max_retries=self.settings.support_agent_openai_max_retries,
                on_retry=lambda attempt, delay: self.observability.log_event(
                    trace_id,
                    "openai_retry_scheduled",
                    {"attempt": attempt, "delay_seconds": round(delay, 3)},
                ),
                on_exhausted=lambda: self.observability.log_event(
                    trace_id,
                    "openai_retry_exhausted",
                    {"max_retries": self.settings.support_agent_openai_max_retries},
                ),
            )
        except UpstreamUnavailableError:
            self.observability.log_event(
                trace_id,
                "upstream_generation_failed",
                {"failure_reason": "generation_failed"},
            )
            raise

    def _generate_fallback_answer(self, retrieved: List[RetrievedChunk]) -> str:
        top_items = retrieved[:2]
        snippets = "\n\n".join(
            f"{item.content}\nReference: {item.source}#{item.chunk_id}"
            for item in top_items
        )
        return f"Answer grounded by retrieved knowledge:\n{snippets}"

    def _split_text(self, text: str) -> List[str]:
        normalized = self._normalize_text(text)
        qa_blocks = self._extract_qa_blocks(normalized)
        if qa_blocks:
            return qa_blocks

        paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]
        if not paragraphs:
            return [normalized] if normalized else []

        chunks: List[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = paragraph if not current else f"{current}\n\n{paragraph}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)

            if len(paragraph) <= self.chunk_size:
                current = paragraph
                continue

            parts = self._split_long_paragraph(paragraph)
            chunks.extend(parts[:-1])
            current = parts[-1]

        if current:
            chunks.append(current)

        return self._apply_overlap(chunks)

    def _extract_qa_blocks(self, text: str) -> List[str]:
        blocks = re.findall(r"(Q:\s.*?\nA:\s.*?)(?=\nQ:\s|\Z)", text, flags=re.S)
        return [block.strip() for block in blocks if block.strip()]

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        parts: List[str] = []
        start = 0
        while start < len(paragraph):
            end = start + self.chunk_size
            parts.append(paragraph[start:end])
            if end >= len(paragraph):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return parts

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        merged: List[str] = [chunks[0]]
        for index in range(1, len(chunks)):
            previous_tail = chunks[index - 1][-self.chunk_overlap :]
            merged.append(f"{previous_tail}\n{chunks[index]}")
        return merged

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", text).strip()
