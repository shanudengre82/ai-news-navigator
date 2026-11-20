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
from ai_news.guardrails import MAX_QUERY_CHARS, check_input, check_output, sanitize_document
from ai_news.rag import prompts
from ai_news.rag.state import Candidate, GraphState
from ai_news.schema import NewsItem
from ai_news.search import passes_filters
from ai_news.store import CorpusStore

log = logging.getLogger(__name__)

RRF_K = 60
MIN_RELEVANT = 3
POOL_PER_RANKING = 200  # BM25 and FAISS hits per query that go into the fusion
MAX_POOL = 300          # compute cap for the cross-encoder, not the result count
MIN_RELEVANCE = 0.02    # reranker probability to count as a match; bge-reranker scores on-topic papers
                        # as low as 0.02, so this favours recall (tuned on the real corpus, Oct 2026)
MIN_KEEP = 5            # keep at least this many matches so the LLM has something to judge
GRADE_TOP = 10          # the LLM reviews and summarizes this many new matches per grading pass
MIN_RESULTS = 5         # always list at least this many references, flagging low matches as weak
RANKED_KEEP = 50        # scored items remembered for padding


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
        f"[{n}] {sanitize_document(c['title'])} ({c['source']}, {c['published'][:10]})\n"
        f"{sanitize_document(c['summary'][:summary_chars])}"
        for n, c in enumerate(cands, 1))


class RagNodes:
    def __init__(self, llm: BaseChatModel, store: CorpusStore, reranker=None,
                 live_search: Callable[[str], list[NewsItem]] | None = None,
                 guardrails_enabled: bool = True, pii_mode: str = "redact",
                 max_query_chars: int = MAX_QUERY_CHARS):
        self.llm = llm
        self.store = store
        self.reranker = reranker
        self.live_search_fn = live_search
        self.guardrails_enabled = guardrails_enabled
        self.pii_mode = pii_mode
        self.max_query_chars = max_query_chars

    def _ask(self, prompt: str) -> str:
        return self.llm.invoke([HumanMessage(content=prompt)]).content

    def _filters(self, state: GraphState) -> tuple[list[str] | None, str | None, int | None]:
        f = state.get("filters") or {}
        source_type = f.get("source_type") or state.get("source_type")
        return f.get("sources") or None, None if source_type in (None, "any", "all") else source_type, \
            f.get("days") or state.get("days")

    # ---- nodes ----
    def input_guardrail(self, state: GraphState) -> dict:
        res = check_input(
            state.get("question", ""),
            state.get("history") or [],
            enabled=self.guardrails_enabled,
            pii_mode=self.pii_mode,
            max_chars=self.max_query_chars,
        )
        pii_list = [f.to_dict() for f in res.pii.findings]
        if not res.allowed:
            return {
                "question": res.sanitized_text,
                "history": res.sanitized_history,
                "guardrail_blocked": True,
                "guardrail_reason": res.blocked_reason,
                "guardrail_flags": res.flags,
                "pii_findings": pii_list,
                "answer": res.blocked_reason or "Request blocked by input guardrails.",
                "references": [],
                "summarized": 0,
                "trace": [f"**Input guardrail** → blocked ({', '.join(res.flags) or 'policy'})"],
            }
        note = f"redacted PII: {', '.join(res.pii.entity_types)}" if res.pii.had_pii else "ok"
        return {
            "question": res.sanitized_text,
            "history": res.sanitized_history,
            "guardrail_blocked": False,
            "guardrail_reason": None,
            "guardrail_flags": res.flags,
            "pii_findings": pii_list,
            "trace": [f"**Input guardrail** → {note}"],
        }

    def route_after_input_guardrail(self, state: GraphState) -> str:
        return "end" if state.get("guardrail_blocked") else "analyze_query"
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
            for ranking in (self.store.bm25_scores(q)[:POOL_PER_RANKING],
                            self.store.vector_search(q, k=POOL_PER_RANKING)):
                for rank, (it, _) in enumerate(ranking):
                    if not passes_filters(it, sources, stype, days):
                        continue
                    items[it.id] = it
                    fused[it.id] = fused.get(it.id, 0.0) + 1.0 / (RRF_K + rank + 1)
        # A date window from the LLM can be too strict; fall back to no window rather than nothing.
        if not fused and days and not (state.get("filters") or {}).get("days"):
            return self.retrieve({**state, "days": None})
        top = sorted(fused, key=fused.get, reverse=True)[:MAX_POOL]
        cands = [to_candidate(items[i], fused[i]) for i in top]
        return {"candidates": cands,
                "trace": [f"**Retrieve** (BM25 + FAISS, RRF) → {len(cands)} candidates"]}

    def _stored_vectors(self, cands: list[Candidate]) -> dict[str, Any]:
        """Fetch already-indexed embeddings from CorpusStore without re-embedding."""
        from ai_news.store import faiss_id
        if not self.store or not len(self.store.vectors):
            return {}
        with self.store._lock:
            present = [c["id"] for c in cands if faiss_id(c["id"]) in self.store._by_faiss_id]
            if not present:
                return {}
            mat = self.store.vectors.get(present)
        return {cid: mat[i] for i, cid in enumerate(present)}

    def rerank(self, state: GraphState) -> dict:
        """Score the pool with the reranker (two-stage cross-encoder + MMR diversity when available,
        or single-stage `.score()` fallback), keep every relevant item, and merge with earlier passes."""
        cands = state.get("candidates", [])
        kept = cands
        stage_note = f"cross-encoder, relevance ≥ {MIN_RELEVANCE}, × recency/upvotes"
        preserve_mmr_order = False
        if self.reranker and cands:
            if hasattr(self.reranker, "rerank"):
                outcome = self.reranker.rerank(
                    state["question"],
                    cands,
                    intent=state.get("intent"),
                    vectors=self._stored_vectors(cands),
                    recency_fn=recency_boost,
                )
                ordered = outcome.candidates
                kept = [c for c in ordered if c.get("relevance", 0.0) >= MIN_RELEVANCE]
                if len(kept) < MIN_KEEP:
                    kept = ordered[:MIN_KEEP]
                preserve_mmr_order = outcome.mmr_applied
                parts = [f"stage1: {outcome.stage1_count}"]
                if outcome.stage2_count:
                    parts.append(f"stage2: {outcome.stage2_count}")
                if outcome.mmr_applied:
                    parts.append("MMR diversity")
                stage_note = f"two-stage cross-encoder ({', '.join(parts)}), relevance ≥ {MIN_RELEVANCE}"
            else:
                scores = self.reranker.score(state["question"], [f"{c['title']}. {c['summary'][:1000]}" for c in cands])
                rels = [min(max(s, 0.0), 1.0) for s in scores]  # Reranker.score returns probabilities
                for c, rel in zip(cands, rels):
                    c["relevance"] = rel
                    # multiplicative, so recency and upvotes only reorder items of similar relevance
                    c["score"] = rel * (1 + 0.25 * recency_boost(c["published"]) + 0.05 * math.log1p(c.get("upvotes", 0)))
                by_rel = sorted(zip(rels, cands), key=lambda p: p[0], reverse=True)
                kept = [c for r, c in by_rel if r >= MIN_RELEVANCE]
                if len(kept) < MIN_KEEP:
                    kept = [c for _, c in by_rel[:MIN_KEEP]]
        ranked = {c["id"]: c for c in state.get("ranked", [])}
        ranked.update({c["id"]: c for c in cands})
        ranked = sorted(ranked.values(), key=lambda c: (c.get("relevance", 0), c.get("score", 0)),
                        reverse=True)[:RANKED_KEEP]
        verdicts = state.get("verdicts") or {}
        prev_matches = state.get("matches", [])
        if preserve_mmr_order and not prev_matches:
            matches = [c for c in kept if verdicts.get(c["id"], {}).get("relevant", True)]
        else:
            merged = {c["id"]: c for c in prev_matches}
            merged.update({c["id"]: {**merged.get(c["id"], {}), **c} for c in kept})
            matches = sorted((c for c in merged.values() if verdicts.get(c["id"], {}).get("relevant", True)),
                             key=lambda c: c.get("score", 0), reverse=True)
        return {"matches": matches, "ranked": ranked,
                "pool_size": len(cands), "rerank_kept": len(kept),
                "trace": [f"**Rerank** ({stage_note}) → "
                          f"{len(kept)} of {len(cands)} kept, {len(matches)} matches in total"]}

    def grade_documents(self, state: GraphState) -> dict:
        verdicts = dict(state.get("verdicts") or {})
        matches = state.get("matches", [])
        todo = [c for c in matches if c["id"] not in verdicts][:GRADE_TOP]
        if todo:
            try:
                out = extract_json(self._ask(prompts.GRADE.format(question=state["question"],
                                                                  docs=format_docs(todo, 400))))
                if isinstance(out, dict):
                    out = out.get("results") or [out]
                for v in out:
                    n = int(v.get("doc", 0))
                    if 1 <= n <= len(todo):
                        digest = str(v.get("digest") or v.get("summary") or v.get("reason") or "").strip()
                        verdicts[todo[n - 1]["id"]] = {"relevant": v.get("relevant") in (True, "true", "yes"),
                                                       "digest": digest}
            except Exception as e:
                log.warning("grade fallback: %s", e)
                verdicts.update({c["id"]: {"relevant": True, "digest": ""} for c in todo[:5]})
        matches = [c for c in matches if verdicts.get(c["id"], {}).get("relevant", True)]
        relevant = [{**c, "digest": verdicts[c["id"]]["digest"]} for c in matches if c["id"] in verdicts]
        n_ok = sum(1 for c in todo if verdicts.get(c["id"], {}).get("relevant"))
        return {"verdicts": verdicts, "matches": matches, "relevant": relevant,
                "trace": [f"**Grade** (LLM judge + digests) → {n_ok}/{len(todo)} new relevant, "
                          f"{len(relevant)} relevant in total"]}

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
        known = {c["id"] for c in state.get("matches", [])}
        cands = [to_candidate(it) for it in found if it.id not in known]
        return {"candidates": cands, "live_searched": True,
                "trace": [f"**Live arXiv search** for '{q}' → {len(found)} papers"]}

    def generate(self, state: GraphState) -> dict:
        """Summary from the best reviewed matches; references are every match, reviewed ones first, topped up
        to MIN_RESULTS with the closest low matches (flagged weak). Citation numbers [n] are positions in
        that reference list."""
        verdicts = state.get("verdicts") or {}
        relevant = sorted(state.get("relevant") or [], key=lambda c: c.get("score", 0), reverse=True)
        seen = {c["id"] for c in relevant}
        rest = [{**c, "weak": c.get("relevance", 1.0) < MIN_RELEVANCE}
                for c in state.get("matches", []) if c["id"] not in seen]
        refs = relevant + rest
        n_pad = 0
        if len(refs) < MIN_RESULTS:
            listed = {c["id"] for c in refs}
            spare = [c for c in (state.get("ranked") or state.get("candidates", [])) if c["id"] not in listed]
            spare.sort(key=lambda c: verdicts.get(c["id"], {}).get("relevant") is False)  # not rejected first
            pad = [{**c, "weak": True, "rejected": verdicts.get(c["id"], {}).get("relevant") is False}
                   for c in spare[:MIN_RESULTS - len(refs)]]
            n_pad = len(pad)
            refs += pad
        refs = [{**c, "summary": c["summary"][:700]} for c in refs]
        if not refs:
            return {"answer": "I couldn't find anything in the corpus or on arXiv for this request. "
                              "Try broader terms.",
                    "references": [], "trace": ["**Generate** → nothing to answer from"]}
        used = refs[:min(len(relevant), GRADE_TOP)] or refs[:MIN_KEEP]
        answer = self._ask(prompts.GENERATE.format(question=state["question"], docs=format_docs(used, 700)))
        if not relevant:
            answer = "_No result passed the relevance check; these are the closest matches._\n\n" + answer
        padded = f" ({len(refs) - n_pad} + {n_pad} low-match to reach {MIN_RESULTS})" if n_pad else ""
        return {"answer": answer, "references": refs, "summarized": len(used),
                "trace": [f"**Generate** → summary from {len(used)} sources, {len(refs)} references{padded}"]}

    def output_guardrail(self, state: GraphState) -> dict:
        refs = state.get("references") or []
        summarized = int(state.get("summarized") or 0)
        max_cite = summarized if summarized > 0 else len(refs)
        allowed_urls = [r["url"] for r in refs if r.get("url")] if refs else None
        res = check_output(
            state.get("answer") or "",
            max_citation=max_cite if refs else None,
            allowed_urls=allowed_urls,
            enabled=self.guardrails_enabled,
        )
        meta = {
            "valid_citations": res.valid_citations,
            "invalid_citations": res.invalid_citations,
            "hallucinated_urls": res.hallucinated_urls,
            "pii_redacted": res.pii.had_pii,
            "pii_findings": [f.to_dict() for f in res.pii.findings],
            "flags": res.flags,
        }
        note = ", ".join(res.flags) if res.flags else "ok"
        return {
            "answer": res.text,
            "output_guardrail": meta,
            "trace": [f"**Output guardrail** → {note}"],
        }
