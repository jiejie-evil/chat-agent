import os
import re
from dataclasses import dataclass
from typing import List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from support_agent.human_interface import HumanAgentInterface
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


class AgentState(TypedDict, total=False):
    query: str
    user_id: str
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
        human_interface: Optional[HumanAgentInterface] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.min_score = min_score
        self.chunks: List[RetrievedChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.chunk_matrix = None
        self.openai_client: Optional[OpenAI] = None
        self.human_interface = human_interface or HumanAgentInterface()
        self.toolbox = SupportToolbox(self.human_interface)
        self.graph = self._build_graph()

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)

    def ingest(self, doc_paths: List[str]) -> None:
        chunk_texts: List[str] = []
        self.chunks = []

        for path in doc_paths:
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()

            for chunk_id, chunk in enumerate(self._split_text(text)):
                cleaned = self._normalize_text(chunk)
                if not cleaned:
                    continue
                self.chunks.append(
                    RetrievedChunk(
                        content=cleaned,
                        source=os.path.basename(path),
                        chunk_id=chunk_id,
                        score=0.0,
                    )
                )
                chunk_texts.append(cleaned)

        if not chunk_texts:
            raise ValueError("No valid documents were ingested.")

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            lowercase=False,
        )
        self.chunk_matrix = self.vectorizer.fit_transform(chunk_texts)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        if not self.vectorizer or self.chunk_matrix is None:
            raise RuntimeError("Please ingest documents before querying the agent.")

        top_k = k or self.top_k
        normalized_query = self._normalize_text(query)
        query_vector = self.vectorizer.transform([normalized_query])
        similarities = cosine_similarity(query_vector, self.chunk_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]

        results: List[RetrievedChunk] = []
        for index in top_indices:
            score = float(similarities[index])
            if score <= 0:
                continue
            chunk = self.chunks[index]
            results.append(
                RetrievedChunk(
                    content=chunk.content,
                    source=chunk.source,
                    chunk_id=chunk.chunk_id,
                    score=score,
                )
            )
        return results

    def answer(self, query: str, user_id: str = "guest") -> str:
        return self.answer_with_metadata(query, user_id=user_id).answer

    def answer_with_metadata(self, query: str, user_id: str = "guest") -> AnswerResult:
        result = self.graph.invoke({"query": query, "user_id": user_id})
        return AnswerResult(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            escalated=result["escalated"],
            tool_name=result.get("tool_name"),
        )

    def is_ready(self) -> bool:
        return self.vectorizer is not None and self.chunk_matrix is not None

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
        graph.add_node("route", self._route_node)
        graph.add_node("tool", self._tool_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("assess", self._assess_node)
        graph.add_node("generate", self._generate_node)
        graph.add_edge(START, "route")
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

    def _route_node(self, state: AgentState) -> AgentState:
        tool_name = self.toolbox.detect_tool(state["query"])
        return {"tool_name": tool_name}

    def _after_route(self, state: AgentState) -> str:
        return "tool" if state.get("tool_name") else "retrieve"

    def _tool_node(self, state: AgentState) -> AgentState:
        tool_name = state.get("tool_name")
        user_id = state.get("user_id", "guest")
        if not tool_name:
            return {
                "answer": "No tool selected.",
                "sources": [],
                "confidence": 1.0,
                "escalated": False,
                "tool_name": None,
            }

        tool_result = self.toolbox.execute(tool_name, state["query"], user_id=user_id)
        escalated = tool_name == "create_ticket"
        return {
            "tool_result": tool_result,
            "answer": tool_result.message,
            "sources": [tool_name],
            "confidence": 1.0 if tool_result.success else 0.0,
            "escalated": escalated,
            "tool_name": tool_name,
        }

    def _retrieve_node(self, state: AgentState) -> AgentState:
        query = state["query"]
        retrieved = self.retrieve(query)
        context = self.build_context(retrieved) if retrieved else ""
        sources = [f"{item.source}#{item.chunk_id}" for item in retrieved]
        confidence = retrieved[0].score if retrieved else 0.0
        return {
            "retrieved": retrieved,
            "context": context,
            "confidence": confidence,
            "sources": sources,
        }

    def _assess_node(self, state: AgentState) -> AgentState:
        confidence = state.get("confidence", 0.0)
        escalated = confidence < self.min_score
        return {"escalated": escalated}

    def _generate_node(self, state: AgentState) -> AgentState:
        query = state["query"]
        retrieved = state.get("retrieved", [])
        confidence = state.get("confidence", 0.0)
        sources = state.get("sources", [])
        escalated = state.get("escalated", False)

        if escalated or not retrieved:
            return {
                "answer": (
                    "I could not find enough grounded knowledge for this question. "
                    "Please hand the conversation to a human agent."
                ),
                "sources": sources,
                "confidence": confidence,
                "escalated": True,
                "tool_name": None,
            }

        context = state["context"]
        if self.openai_client:
            answer = self._generate_with_openai(query, context)
        else:
            answer = self._generate_fallback_answer(retrieved)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "escalated": False,
            "tool_name": None,
        }

    def _generate_with_openai(self, query: str, context: str) -> str:
        response = self.openai_client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
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
