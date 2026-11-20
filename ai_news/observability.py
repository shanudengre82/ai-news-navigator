"""Fork-safe Langfuse observability and automated RAG quality evaluation.

Tracks:
- Root trace per Ask / Post-rewrite request (always PII-redacted)
- Automatic LangChain/LangGraph node & LLM generation callbacks
- Structured snapshots of retrieval, reranking, grading, and guardrail objects
- Automated retrieval-quality and answer-quality scores posted to Langfuse

ZeroGPU compatibility:
- Background ingestion threads created in a parent process do not survive `os.fork()`.
  We register an `os.register_at_fork(after_in_child=...)` hook to reset the cached client,
  and always `flush()` in a `finally:` block before the GPU worker exits.
- When Langfuse credentials are not configured, every operation is a zero-cost no-op.
"""

from __future__ import annotations

import logging
import os
import re
import statistics
from typing import Any

from ai_news.config import Settings, settings as default_settings
from ai_news.guardrails import redact_pii
from ai_news.text import tokenize

log = logging.getLogger(__name__)

_CLIENT: Any = None
_START_WITH_RE = re.compile(r"\*\*Start with\*\*\s*\[(\d+)\]", re.I)
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _reset_client_after_fork() -> None:
    """Clear any inherited Langfuse client in a ZeroGPU forked child process."""
    global _CLIENT
    _CLIENT = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_client_after_fork)


def is_enabled(cfg: Settings | None = None) -> bool:
    cfg = cfg or default_settings
    return bool(cfg.langfuse_enabled and cfg.langfuse_public_key and cfg.langfuse_secret_key)


def get_client(cfg: Settings | None = None) -> Any | None:
    """Lazily initialize and return a Langfuse client, or None if disabled/unavailable."""
    global _CLIENT
    cfg = cfg or default_settings
    if not is_enabled(cfg):
        return None
    if _CLIENT is not None:
        return _CLIENT
    try:
        from langfuse import Langfuse
        _CLIENT = Langfuse(
            public_key=cfg.langfuse_public_key,
            secret_key=cfg.langfuse_secret_key,
            host=cfg.langfuse_host,
        )
        return _CLIENT
    except Exception as e:
        log.warning("Langfuse unavailable: %s", e)
        return None


def _make_langchain_handler(cfg: Settings) -> Any | None:
    """Build a Langfuse LangChain CallbackHandler (supports v3 and v2 imports)."""
    try:
        from langfuse.langchain import CallbackHandler
        try:
            return CallbackHandler(public_key=cfg.langfuse_public_key)
        except TypeError:
            return CallbackHandler()
    except Exception:
        try:
            from langfuse.callback import CallbackHandler
            return CallbackHandler(
                public_key=cfg.langfuse_public_key,
                secret_key=cfg.langfuse_secret_key,
                host=cfg.langfuse_host,
            )
        except Exception as e:
            log.debug("Langfuse LangChain CallbackHandler unavailable: %s", e)
            return None


# ------------------------------------------------------------------ Automated evaluation metrics
def _lexical_grounding(answer: str, used_refs: list[dict]) -> float:
    """Fast faithfulness proxy: fraction of answer content tokens supported by cited/used sources."""
    ans_tokens = set(tokenize(answer))
    if not ans_tokens or not used_refs:
        return 0.0
    src_tokens: set[str] = set()
    for r in used_refs:
        src_tokens.update(tokenize(f"{r.get('title', '')} {r.get('digest', '')} {r.get('summary', '')}"))
    if not src_tokens:
        return 0.0
    return round(len(ans_tokens & src_tokens) / len(ans_tokens), 4)


def compute_rag_scores(state: dict) -> dict[str, tuple[float, str]]:
    """Compute retrieval-quality, answer-quality, and guardrail scores from final GraphState.

    Returns `{score_name: (numeric_value, short_comment)}`.
    """
    scores: dict[str, tuple[float, str]] = {}

    blocked = bool(state.get("guardrail_blocked"))
    pii_findings = state.get("pii_findings") or []
    scores["guardrail_passed"] = (0.0 if blocked else 1.0, state.get("guardrail_reason") or "passed")
    scores["pii_detected"] = (
        1.0 if pii_findings else 0.0,
        ", ".join(f"{f['entity_type']}x{f['count']}" for f in pii_findings) if pii_findings else "none",
    )

    if blocked:
        return scores

    # --- Retrieval quality ---
    matches = state.get("matches") or []
    refs = state.get("references") or []
    verdicts = state.get("verdicts") or {}
    pool_size = int(state.get("pool_size") or len(state.get("candidates") or []))
    rerank_kept = int(state.get("rerank_kept") or len(matches))

    rels = [float(c["relevance"]) for c in matches if c.get("relevance") is not None]
    if rels:
        scores["retrieval_top1_relevance"] = (round(max(rels), 4), "max cross-encoder probability")
        top5 = sorted(rels, reverse=True)[:5]
        scores["retrieval_mean_top5_relevance"] = (round(sum(top5) / len(top5), 4), f"mean of top {len(top5)}")

    if pool_size > 0:
        keep_rate = min(1.0, rerank_kept / pool_size)
        scores["retrieval_rerank_keep_rate"] = (round(keep_rate, 4), f"{rerank_kept}/{pool_size} kept")

    if verdicts:
        n_rel = sum(1 for v in verdicts.values() if v.get("relevant"))
        scores["retrieval_grade_pass_rate"] = (round(n_rel / len(verdicts), 4), f"{n_rel}/{len(verdicts)} relevant")

    if refs:
        n_strong = sum(1 for r in refs if not r.get("weak"))
        scores["retrieval_strong_ref_ratio"] = (round(n_strong / len(refs), 4), f"{n_strong}/{len(refs)} strong")

    retry_count = int(state.get("retries") or 0) + (1 if state.get("live_searched") else 0)
    scores["retrieval_retry_count"] = (float(retry_count), "rewrites + live searches (0 is best)")

    # --- Answer quality ---
    answer = str(state.get("answer") or "")
    summarized = int(state.get("summarized") or 0)
    used_refs = refs[:summarized] if summarized > 0 else []

    og = state.get("output_guardrail") or {}
    valid_cites = og.get("valid_citations")
    invalid_cites = og.get("invalid_citations")
    if valid_cites is None:
        found = [int(m.group(1)) for m in _CITATION_RE.finditer(answer)]
        valid_cites = [n for n in found if 1 <= n <= len(used_refs)] if used_refs else found
        invalid_cites = [n for n in found if used_refs and not (1 <= n <= len(used_refs))]

    total_cites = len(valid_cites) + len(invalid_cites or [])
    if total_cites > 0:
        scores["answer_citation_validity"] = (
            round(len(valid_cites) / total_cites, 4),
            f"{len(valid_cites)}/{total_cites} valid inline citations",
        )
    elif used_refs:
        scores["answer_citation_validity"] = (0.0, "no inline citations found")

    if used_refs:
        unique_cited = {n for n in valid_cites if 1 <= n <= len(used_refs)}
        scores["answer_citation_coverage"] = (
            round(len(unique_cited) / len(used_refs), 4),
            f"{len(unique_cited)}/{len(used_refs)} used sources cited",
        )
        scores["answer_lexical_grounding"] = (
            _lexical_grounding(answer, used_refs),
            "answer content-token overlap with used sources",
        )

    sw = _START_WITH_RE.search(answer)
    has_sw = bool(sw and (not used_refs or 1 <= int(sw.group(1)) <= len(used_refs)))
    scores["answer_has_start_with"] = (1.0 if has_sw else 0.0, "includes valid **Start with** [n]")

    return scores


# ------------------------------------------------------------------ Structured object snapshots
def _compact_cand(c: dict) -> dict:
    return {
        "id": c.get("id"),
        "title": str(c.get("title", ""))[:120],
        "source": c.get("source"),
        "relevance": round(float(c["relevance"]), 4) if c.get("relevance") is not None else None,
        "score": round(float(c["score"]), 4) if c.get("score") is not None else None,
    }


def summarize_node_object(node: str, delta: dict) -> dict:
    """Produce a compact, PII-free structured snapshot of the objects flowing through a RAG node."""
    if node == "input_guardrail":
        return {
            "blocked": bool(delta.get("guardrail_blocked")),
            "reason": delta.get("guardrail_reason"),
            "flags": delta.get("guardrail_flags") or [],
            "pii_findings": delta.get("pii_findings") or [],
        }
    if node == "analyze_query":
        return {
            "intent": delta.get("intent"),
            "queries": delta.get("queries") or [],
            "days": delta.get("days"),
            "source_type": delta.get("source_type"),
        }
    if node == "retrieve":
        cands = delta.get("candidates") or []
        return {
            "candidate_count": len(cands),
            "top_candidates": [_compact_cand(c) for c in cands[:10]],
        }
    if node == "rerank":
        matches = delta.get("matches") or []
        rels = [float(c["relevance"]) for c in matches if c.get("relevance") is not None]
        stats = {
            "min": round(min(rels), 4),
            "median": round(statistics.median(rels), 4),
            "mean": round(sum(rels) / len(rels), 4),
            "max": round(max(rels), 4),
        } if rels else {}
        return {
            "pool_size": delta.get("pool_size"),
            "kept_count": delta.get("rerank_kept"),
            "total_matches": len(matches),
            "relevance_stats": stats,
            "top_matches": [_compact_cand(c) for c in matches[:10]],
        }
    if node == "grade_documents":
        verdicts = delta.get("verdicts") or {}
        rel = delta.get("relevant") or []
        return {
            "total_reviewed": len(verdicts),
            "total_relevant": len(rel),
            "pass_rate": round(len(rel) / len(verdicts), 4) if verdicts else 0.0,
        }
    if node == "rewrite_query":
        return {"queries": delta.get("queries") or [], "retries": delta.get("retries")}
    if node == "live_search":
        cands = delta.get("candidates") or []
        return {"found_new": len(cands), "titles": [str(c.get("title", ""))[:100] for c in cands[:10]]}
    if node == "generate":
        refs = delta.get("references") or []
        return {
            "answer_chars": len(str(delta.get("answer") or "")),
            "references_count": len(refs),
            "summarized_count": delta.get("summarized", 0),
        }
    if node == "output_guardrail":
        return dict(delta.get("output_guardrail") or {})
    return {"keys": sorted(delta.keys())}


# ------------------------------------------------------------------ RagTracer context manager
class RagTracer:
    """Context manager that traces one RAG run in Langfuse and posts retrieval/answer scores."""

    def __init__(
        self,
        question: str,
        *,
        corpus_size: int = 0,
        filters: dict | None = None,
        history_len: int = 0,
        cfg: Settings | None = None,
        client: Any | None = None,
    ):
        self.cfg = cfg or default_settings
        # Always redact PII before anything touches Langfuse
        self.redacted_question = redact_pii(question).text
        self.corpus_size = corpus_size
        self.filters = filters or {}
        self.history_len = history_len
        self._client = client if client is not None else get_client(self.cfg)
        self._root: Any = None
        self._trace_id: str | None = None
        self._lc_handler: Any = None

    @property
    def active(self) -> bool:
        return self._client is not None

    def __enter__(self) -> "RagTracer":
        if not self._client:
            return self
        meta = {
            "corpus_size": self.corpus_size,
            "history_turns": self.history_len,
            "filters": self.filters,
            "llm_model_id": self.cfg.llm_model_id,
            "embed_model_id": self.cfg.embed_model_id,
            "rerank_model_id": self.cfg.rerank_model_id,
            "pii_mode": self.cfg.guardrails_pii_mode,
        }
        inp = {"question": self.redacted_question, "filters": self.filters}
        try:
            if hasattr(self._client, "start_span"):
                self._root = self._client.start_span(name="rag_assistant", input=inp, metadata=meta)
                self._trace_id = getattr(self._root, "trace_id", None)
            elif hasattr(self._client, "trace"):
                self._root = self._client.trace(name="rag_assistant", input=inp, metadata=meta, tags=["rag"])
                self._trace_id = getattr(self._root, "id", None)
            self._lc_handler = _make_langchain_handler(self.cfg)
        except Exception as e:
            log.warning("Langfuse trace start failed: %s", e)
            self._root = None
        return self

    def langchain_callbacks(self) -> list[Any]:
        return [self._lc_handler] if self._lc_handler is not None else []

    def record_node(self, node: str, delta: dict) -> None:
        if not self._root:
            return
        payload = summarize_node_object(node, delta)
        try:
            if hasattr(self._root, "start_span"):
                sp = self._root.start_span(name=f"node:{node}", output=payload)
                sp.end()
            elif hasattr(self._root, "span"):
                sp = self._root.span(name=f"node:{node}", output=payload)
                if hasattr(sp, "end"):
                    sp.end()
        except Exception as e:
            log.debug("Langfuse span failed for %s: %s", node, e)

    def finish(self, final_state: dict, error: str | None = None) -> dict[str, tuple[float, str]]:
        scores = {} if error else compute_rag_scores(final_state)
        if not self._root:
            return scores
        try:
            out = (
                {"error": error}
                if error
                else {
                    "answer": redact_pii(str(final_state.get("answer") or "")).text,
                    "references_count": len(final_state.get("references") or []),
                    "summarized": final_state.get("summarized", 0),
                    "guardrail_blocked": bool(final_state.get("guardrail_blocked")),
                }
            )
            if hasattr(self._root, "update"):
                self._root.update(output=out)
            if hasattr(self._root, "end"):
                self._root.end()
            for name, (val, comment) in scores.items():
                self._post_score(name, val, comment)
        except Exception as e:
            log.warning("Langfuse finish/score failed: %s", e)
        return scores

    def _post_score(self, name: str, value: float, comment: str) -> None:
        if self._client and hasattr(self._client, "create_score") and self._trace_id:
            self._client.create_score(trace_id=self._trace_id, name=name, value=value, comment=comment)
        elif self._root and hasattr(self._root, "score"):
            self._root.score(name=name, value=value, comment=comment)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._client:
            try:
                if exc is not None and self._root is not None:
                    self.finish({}, error=f"{exc_type.__name__}: {exc}")
                self._client.flush()
            except Exception as e:
                log.debug("Langfuse flush failed: %s", e)


def trace_polish_run(
    draft: str,
    result_text: str,
    note: str,
    *,
    blocked: bool = False,
    pii_detected: bool = False,
    cfg: Settings | None = None,
    client: Any | None = None,
) -> None:
    """Record and flush a single Langfuse trace for the Post tab's AI rewrite."""
    c = client if client is not None else get_client(cfg)
    if not c:
        return
    try:
        inp = {"draft_chars": len(draft), "draft_preview": redact_pii(draft[:400]).text}
        out = {"result_chars": len(result_text), "note": note, "blocked": blocked}
        trace_id = None
        root = None
        if hasattr(c, "start_span"):
            root = c.start_span(name="polish_post", input=inp, output=out)
            trace_id = getattr(root, "trace_id", None)
            root.end()
        elif hasattr(c, "trace"):
            root = c.trace(name="polish_post", input=inp, output=out, tags=["post"])
            trace_id = getattr(root, "id", None)

        kept_links = 0.0 if ("dropped some links" in note or "empty" in note or blocked) else 1.0
        for name, val in (
            ("polish_guardrail_passed", 0.0 if blocked else 1.0),
            ("polish_pii_detected", 1.0 if pii_detected else 0.0),
            ("polish_links_preserved", kept_links),
        ):
            if hasattr(c, "create_score") and trace_id:
                c.create_score(trace_id=trace_id, name=name, value=val)
            elif root and hasattr(root, "score"):
                root.score(name=name, value=val)
        c.flush()
    except Exception as e:
        log.debug("Langfuse polish trace failed: %s", e)
