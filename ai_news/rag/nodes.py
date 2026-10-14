"""Node implementations for the corrective-RAG graph."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ai_news.fetchers.common import parse_iso
from ai_news.rag import prompts
from ai_news.rag.state import Candidate, GraphState
from ai_news.schema import NewsItem
from ai_news.search import passes_filters
from ai_news.store import CorpusStore

log = logging.getLogger(__name__)

RRF_K = 60
MIN_RELEVANT = 3
RERANK_TOP = 8


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in an LLM reply (tolerates ```json fences and chatter)."""
    text = re.sub(r"```(?:json)?", "", text)
    pairs = sorted((("{", "}"), ("[", "]")), key=lambda p: (text.find(p[0]) == -1, text.find(p[0])))
    for open_, close in pairs:
        start = text.find(open_)
        end = text.rfind(close)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON in: {text[:200]!r}")


def to_candidate(it: NewsItem, score: float = 0.0) -> Candidate:
    return Candidate(id=it.id, title=it.title, url=it.url, source=it.source, source_type=it.source_type,
                     published=it.published, summary=it.summary, upvotes=it.upvotes, score=score)


def recency_boost(published: str, half_life_days: float = 30.0) -> float:
    age = (datetime.now(timezone.utc) - parse_iso(published)).total_seconds() / 86400
    return 0.5 ** (max(age, 0) / half_life_days)


def format_docs(cands: list[Candidate], summary_chars: int) -> str:
    return "\n\n".join(
        f"[{n}] {c['title']} ({c['source']}, {c['published'][:10]})\n{c['summary'][:summary_chars]}"
        for n, c in enumerate(cands, 1))


class RagNodes:
    def __init__(self, llm: BaseChatModel, store: CorpusStore, reranker=None,
                 live_search: Callable[[str], list[NewsItem]] | None = None):
        self.llm = llm
        self.store = store
        self.reranker = reranker
        self.live_search_fn = live_search

    def _ask(self, prompt: str) -> str:
        return self.llm.invoke([HumanMessage(content=prompt)]).content

    def _filters(self, state: GraphState) -> tuple[list[str] | None, str | None, int | None]:
        f = state.get("filters") or {}
        source_type = f.get("source_type") or state.get("source_type")
        return f.get("sources") or None, None if source_type in (None, "any", "all") else source_type, \
            f.get("days") or state.get("days")

    # ---- nodes ----
    def analyze_query(self, state: GraphState) -> dict:
        q = state["question"]
        hist = state.get("history") or []
        history = ("Earlier questions in this conversation: " + " | ".join(map(str, hist[-2:]))) if hist else ""
        try:
            data = extract_json(self._ask(prompts.ANALYZE.format(question=q, history=history)))
            queries = [s for s in data.get("search_queries", []) if isinstance(s, str) and s.strip()][:3] or [q]
            days = data.get("date_window_days")
            days = int(days) if isinstance(days, (int, float)) and days > 0 else None
            stype = data.get("source_type") if data.get("source_type") in ("paper", "blog") else None
            intent = str(data.get("intent", q))
        except Exception as e:
            log.warning("analyze_query fallback: %s", e)
            queries, days, stype, intent = [q], None, None, q
        if q not in queries:
            queries.append(q)  # always also search the raw question
        return {"intent": intent, "queries": queries, "days": days, "source_type": stype, "retries": 0,
                "live_searched": False,
                "trace": [f"**Analyze** → intent: _{intent}_; queries: {queries}; "
                          f"window: {days or 'any'} days; type: {stype or 'any'}"]}

    def retrieve(self, state: GraphState) -> dict:
        sources, stype, days = self._filters(state)
        fused: dict[str, float] = {}
        items: dict[str, NewsItem] = {}
        for q in state["queries"]:
            for ranking in (self.store.bm25_scores(q)[:50], self.store.vector_search(q, k=50)):
                for rank, (it, _) in enumerate(ranking):
                    if not passes_filters(it, sources, stype, days):
                        continue
                    items[it.id] = it
                    fused[it.id] = fused.get(it.id, 0.0) + 1.0 / (RRF_K + rank + 1)
        # A date window from the LLM can be too strict; fall back to no window rather than nothing.
        if not fused and days and not (state.get("filters") or {}).get("days"):
            return self.retrieve({**state, "days": None})
        top = sorted(fused, key=fused.get, reverse=True)[:30]
        cands = [to_candidate(items[i], fused[i]) for i in top]
        return {"candidates": cands,
                "trace": [f"**Retrieve** (BM25 + FAISS, RRF) → {len(cands)} candidates"]}

    def rerank(self, state: GraphState) -> dict:
        cands = state.get("candidates", [])
        if self.reranker and cands:
            scores = self.reranker.score(state["question"], [f"{c['title']}. {c['summary'][:1000]}" for c in cands])
            for c, s in zip(cands, scores):
                rel = 1 / (1 + math.exp(-s))
                c["score"] = rel + 0.15 * recency_boost(c["published"]) + 0.03 * math.log1p(c.get("upvotes", 0))
            cands = sorted(cands, key=lambda c: c["score"], reverse=True)
        top = cands[:RERANK_TOP]
        return {"candidates": top, "trace": [f"**Rerank** (cross-encoder + recency/upvotes) → top {len(top)}"]}

    def grade_documents(self, state: GraphState) -> dict:
        cands = state.get("candidates", [])
        if not cands:
            return {"relevant": [], "trace": ["**Grade** → no candidates"]}
        try:
            verdicts = extract_json(self._ask(prompts.GRADE.format(question=state["question"],
                                                                   docs=format_docs(cands, 400))))
            if isinstance(verdicts, dict):
                verdicts = verdicts.get("results") or [verdicts]
            relevant = []
            for v in verdicts:
                n = int(v.get("doc", 0))
                if v.get("relevant") in (True, "true", "yes") and 1 <= n <= len(cands):
                    relevant.append({**cands[n - 1], "reason": str(v.get("reason", ""))})
        except Exception as e:
            log.warning("grade fallback: %s", e)
            relevant = [{**c, "reason": ""} for c in cands[:5]]
        # keep any previously graded relevant docs (after a rewrite or live search)
        seen = {c["id"] for c in relevant}
        relevant += [c for c in state.get("relevant", []) if c["id"] not in seen]
        return {"relevant": relevant, "trace": [f"**Grade** (LLM judge) → {len(relevant)}/{len(cands)} relevant"]}

    def route_after_grade(self, state: GraphState) -> str:
        if len(state.get("relevant", [])) >= MIN_RELEVANT:
            return "generate"
        if state.get("retries", 0) < 1:
            return "rewrite_query"
        if self.live_search_fn and not state.get("live_searched"):
            return "live_search"
        return "generate"

    def rewrite_query(self, state: GraphState) -> dict:
        try:
            data = extract_json(self._ask(prompts.REWRITE.format(question=state["question"],
                                                                 queries=state.get("queries"))))
            queries = [s for s in data.get("search_queries", []) if isinstance(s, str) and s.strip()][:3]
        except Exception as e:
            log.warning("rewrite fallback: %s", e)
            queries = []
        queries = queries or [state["question"]]
        return {"queries": queries, "retries": state.get("retries", 0) + 1,
                "trace": [f"**Rewrite** (too few relevant) → {queries}"]}

    def live_search(self, state: GraphState) -> dict:
        q = (state.get("queries") or [state["question"]])[0]
        try:
            found = self.live_search_fn(q) if self.live_search_fn else []
        except Exception as e:
            log.warning("live search failed: %s", e)
            found = []
        if found:
            self.store.add(found)  # grows the corpus; persisted on the next refresh
        known = {c["id"] for c in state.get("candidates", [])}
        cands = state.get("candidates", []) + [to_candidate(it) for it in found if it.id not in known]
        return {"candidates": cands, "live_searched": True,
                "trace": [f"**Live arXiv search** for '{q}' → {len(found)} papers"]}

    def generate(self, state: GraphState) -> dict:
        refs = state.get("relevant") or []
        partial = False
        if not refs:
            refs, partial = state.get("candidates", [])[:5], True
        if not refs:
            return {"answer": "I couldn't find anything in the corpus or on arXiv for this request. "
                              "Try broader terms or remove filters.",
                    "references": [], "trace": ["**Generate** → nothing to answer from"]}
        refs = sorted(refs, key=lambda c: c.get("score", 0), reverse=True)[:8]
        answer = self._ask(prompts.GENERATE.format(question=state["question"], docs=format_docs(refs, 700)))
        if partial:
            answer = "_No result passed the relevance check; these are the closest matches._\n\n" + answer
        return {"answer": answer, "references": refs, "trace": [f"**Generate** → answer from {len(refs)} sources"]}
