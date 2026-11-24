"""Provider-agnostic observability and automated RAG quality evaluation.

Design:
- The rest of the codebase (`assistant.py`, `app.py`, `rag/`) depends ONLY on `RagTracer`,
  `trace_polish_run`, and `compute_rag_scores` — never on Langfuse directly.
- When `OBSERVABILITY_BACKEND=none` or Langfuse credentials (`LANGFUSE_PUBLIC_KEY`,
  `LANGFUSE_SECRET_KEY`) are missing in `auto` mode, all Langfuse imports, tracing, span recording,
  scoring, and flushing are completely skipped with zero overhead.
- Custom backends can be plugged in via `register_backend(name, factory)` without touching any
  application code.
- Langfuse (optional backend) is fork-safe for ZeroGPU via `os.register_at_fork`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from ai_news.config import Settings, settings as default_settings
from ai_news.guardrails import redact_pii
from ai_news.text import tokenize

log = logging.getLogger(__name__)

_START_WITH_RE = re.compile(r"\*\*Start with\*\*\s*\[(\d+)\]", re.I)
_CITATION_RE = re.compile(r"\[(\d+)\]")


# ------------------------------------------------------------------ Provider-agnostic protocols
class TraceHandle(Protocol):
    def record_span(self, name: str, output: dict) -> None: ...
    def record_score(self, name: str, value: float, comment: str = "") -> None: ...
    def finish(self, output: dict) -> None: ...


class ObservabilityBackend(Protocol):
    def start_trace(self, name: str, *, input: dict, metadata: dict | None = None,
                    tags: list[str] | None = None) -> TraceHandle | None: ...
    def langchain_callbacks(self) -> list[Any]: ...
    def flush(self) -> None: ...


# ------------------------------------------------------------------ Optional Langfuse backend
_LANGFUSE_CLIENT: Any = None


def _reset_client_after_fork() -> None:
    """Clear any inherited Langfuse client in a ZeroGPU forked child process."""
    global _LANGFUSE_CLIENT
    _LANGFUSE_CLIENT = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_client_after_fork)


class _LangfuseTraceHandle:
    def __init__(self, client: Any, root: Any, trace_id: str | None):
        self._client = client
        self._root = root
        self._trace_id = trace_id

    def record_span(self, name: str, output: dict) -> None:
        try:
            if hasattr(self._root, "start_span"):
                sp = self._root.start_span(name=name, output=output)
                sp.end()
            elif hasattr(self._root, "span"):
                sp = self._root.span(name=name, output=output)
                if hasattr(sp, "end"):
                    sp.end()
        except Exception as e:
            log.debug("Langfuse span failed for %s: %s", name, e)

    def record_score(self, name: str, value: float, comment: str = "") -> None:
        try:
            if self._client and hasattr(self._client, "create_score") and self._trace_id:
                self._client.create_score(trace_id=self._trace_id, name=name, value=value, comment=comment)
            elif hasattr(self._root, "score"):
                self._root.score(name=name, value=value, comment=comment)
        except Exception as e:
            log.debug("Langfuse score failed for %s: %s", name, e)

    def finish(self, output: dict) -> None:
        try:
            if hasattr(self._root, "update"):
                self._root.update(output=output)
            if hasattr(self._root, "end"):
                self._root.end()
        except Exception as e:
            log.debug("Langfuse trace finish failed: %s", e)


class LangfuseBackend:
    """Optional Langfuse adapter (supports SDK v3 and v2). Never imported unless configured."""

    def __init__(self, cfg: Settings, client: Any | None = None):
        self.cfg = cfg
        self._client = client if client is not None else self._init_client(cfg)

    @staticmethod
    def _init_client(cfg: Settings) -> Any | None:
        global _LANGFUSE_CLIENT
        if _LANGFUSE_CLIENT is not None:
            return _LANGFUSE_CLIENT
        try:
            from langfuse import Langfuse
            _LANGFUSE_CLIENT = Langfuse(
                public_key=cfg.langfuse_public_key,
                secret_key=cfg.langfuse_secret_key,
                host=cfg.langfuse_host,
            )
            return _LANGFUSE_CLIENT
        except Exception as e:
            log.warning("Langfuse backend requested but unavailable (%s); skipping observability", e)
            return None

    @property
    def ready(self) -> bool:
        return self._client is not None

    def start_trace(self, name: str, *, input: dict, metadata: dict | None = None,
                    tags: list[str] | None = None) -> TraceHandle | None:
        if not self._client:
            return None
        try:
            if hasattr(self._client, "start_span"):
                root = self._client.start_span(name=name, input=input, metadata=metadata)
                return _LangfuseTraceHandle(self._client, root, getattr(root, "trace_id", None))
            if hasattr(self._client, "trace"):
                root = self._client.trace(name=name, input=input, metadata=metadata, tags=tags or [])
                return _LangfuseTraceHandle(self._client, root, getattr(root, "id", None))
        except Exception as e:
            log.warning("Langfuse trace start failed: %s", e)
        return None

    def langchain_callbacks(self) -> list[Any]:
        if not self._client:
            return []
        try:
            from langfuse.langchain import CallbackHandler
            try:
                return [CallbackHandler(public_key=self.cfg.langfuse_public_key)]
            except TypeError:
                return [CallbackHandler()]
        except Exception:
            try:
                from langfuse.callback import CallbackHandler
                return [CallbackHandler(
                    public_key=self.cfg.langfuse_public_key,
                    secret_key=self.cfg.langfuse_secret_key,
                    host=self.cfg.langfuse_host,
                )]
            except Exception as e:
                log.debug("Langfuse LangChain CallbackHandler unavailable: %s", e)
                return []

    def flush(self) -> None:
        if self._client:
            try:
                self._client.flush()
            except Exception as e:
                log.debug("Langfuse flush failed: %s", e)


# ------------------------------------------------------------------ Built-in Local JSONL & Composite backends
_LOCAL_TRACE_LOCK = threading.Lock()


class _LocalTraceHandle:
    def __init__(
        self,
        path: Path,
        name: str,
        *,
        inp: dict,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ):
        self._path = path
        self._name = name
        self._trace_id = uuid.uuid4().hex[:12]
        self._started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self._t0 = time.perf_counter()
        self._input = dict(inp or {})
        self._metadata = dict(metadata or {})
        self._tags = list(tags or [])
        self._spans: dict[str, dict] = {}
        self._scores: dict[str, list] = {}
        self._output: dict | None = None
        self._written = False

    def record_span(self, name: str, output: dict) -> None:
        self._spans[name] = output

    def record_score(self, name: str, value: float, comment: str = "") -> None:
        self._scores[name] = [value, comment]

    def finish(self, output: dict) -> None:
        self._output = dict(output or {})

    def _prompt_redacted(self) -> str:
        if "question" in self._input:
            return str(self._input.get("question") or "")
        if "draft_preview" in self._input:
            return str(self._input.get("draft_preview") or "")
        return ""

    def flush(self) -> None:
        if self._written:
            return
        self._written = True
        latency_ms = round((time.perf_counter() - self._t0) * 1000.0, 1)
        out = self._output or {}
        record = {
            "trace_id": self._trace_id,
            "timestamp": self._started_at,
            "entrypoint": self._name,
            "latency_ms": latency_ms,
            "prompt_redacted": self._prompt_redacted(),
            "input": self._input,
            "metadata": self._metadata,
            "tags": self._tags,
            "spans": self._spans,
            "scores": self._scores,
            "guardrail_report": out.get("guardrail_report"),
            "output": {k: v for k, v in out.items() if k != "guardrail_report"},
            "error": out.get("error"),
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            with _LOCAL_TRACE_LOCK:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as e:
            log.warning("Local JSONL trace write failed (%s): %s", self._path, e)


class LocalJsonlBackend:
    """Always-available local JSONL trace backend (zero external dependencies, PII-redacted)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._pending: list[_LocalTraceHandle] = []

    def start_trace(self, name: str, *, input: dict, metadata: dict | None = None,
                    tags: list[str] | None = None) -> TraceHandle | None:
        handle = _LocalTraceHandle(self.path, name, inp=input, metadata=metadata, tags=tags)
        self._pending.append(handle)
        return handle

    def langchain_callbacks(self) -> list[Any]:
        return []

    def flush(self) -> None:
        while self._pending:
            self._pending.pop(0).flush()


class _CompositeTraceHandle:
    def __init__(self, handles: list[TraceHandle]):
        self._handles = handles

    def record_span(self, name: str, output: dict) -> None:
        for h in self._handles:
            h.record_span(name, output)

    def record_score(self, name: str, value: float, comment: str = "") -> None:
        for h in self._handles:
            h.record_score(name, value, comment)

    def finish(self, output: dict) -> None:
        for h in self._handles:
            h.finish(output)


class CompositeBackend:
    """Fan-out observability backend (e.g., local JSONL + Langfuse simultaneously)."""

    def __init__(self, backends: list[ObservabilityBackend]):
        self._backends = list(backends)

    def start_trace(self, name: str, *, input: dict, metadata: dict | None = None,
                    tags: list[str] | None = None) -> TraceHandle | None:
        handles = [
            h for b in self._backends
            if (h := b.start_trace(name, input=input, metadata=metadata, tags=tags)) is not None
        ]
        return _CompositeTraceHandle(handles) if handles else None

    def langchain_callbacks(self) -> list[Any]:
        cbs: list[Any] = []
        for b in self._backends:
            cbs.extend(b.langchain_callbacks())
        return cbs

    def flush(self) -> None:
        for b in self._backends:
            b.flush()


def read_recent_traces(
    path: Path | None = None,
    *,
    limit: int = 100,
    filter_mode: str = "all",
) -> list[dict]:
    """Read recent per-prompt traces from the local JSONL log, newest first."""
    target = Path(path) if path is not None else default_settings.trace_log_path
    if not target.exists():
        return []
    entries: list[dict] = []
    try:
        with target.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        log.warning("Failed to read trace log %s: %s", target, e)
        return []

    entries.reverse()
    mode = (filter_mode or "all").strip().lower()
    if mode != "all":
        filtered: list[dict] = []
        for e in entries:
            rep = e.get("guardrail_report") or {}
            in_flags = rep.get("input_flags") or []
            out_flags = rep.get("output_flags") or []
            in_pii = rep.get("input_pii") or []
            out_pii = rep.get("output_pii") or []
            blocked = rep.get("allowed") is False or bool((e.get("output") or {}).get("guardrail_blocked")) or bool((e.get("output") or {}).get("blocked"))
            if mode == "flagged" and (blocked or in_flags or out_flags or (rep.get("docs_sanitized") or 0) > 0):
                filtered.append(e)
            elif mode == "pii" and (in_pii or out_pii or (e.get("scores", {}).get("pii_detected", [0])[0] > 0) or (e.get("scores", {}).get("polish_pii_detected", [0])[0] > 0)):
                filtered.append(e)
            elif mode == "polish" and e.get("entrypoint") == "polish_post":
                filtered.append(e)
        entries = filtered
    return entries[: max(1, int(limit))]


# ------------------------------------------------------------------ Backend registry & selection
BackendFactory = Callable[[Settings], ObservabilityBackend | None]
_BACKENDS: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
    """Register a custom observability backend factory under `name`."""
    _BACKENDS[name.strip().lower()] = factory


def unregister_backend(name: str) -> None:
    _BACKENDS.pop(name.strip().lower(), None)


def _langfuse_configured(cfg: Settings) -> bool:
    return bool(cfg.langfuse_enabled and cfg.langfuse_public_key and cfg.langfuse_secret_key)


def is_enabled(cfg: Settings | None = None) -> bool:
    """Return True when at least one observability backend (local JSONL, Langfuse, or custom) is active."""
    cfg = cfg or default_settings
    mode = (cfg.observability_backend or "auto").strip().lower()
    if mode in {"none", "off", "false", "0", ""}:
        return False
    if mode in _BACKENDS:
        return True
    if mode == "local":
        return bool(cfg.trace_local_enabled)
    if mode == "langfuse":
        return _langfuse_configured(cfg)
    if mode == "auto":
        return bool(cfg.trace_local_enabled) or _langfuse_configured(cfg)
    return False


def get_backend(cfg: Settings | None = None) -> ObservabilityBackend | None:
    """Return the active ObservabilityBackend (single or Composite), or None to skip tracing."""
    cfg = cfg or default_settings
    mode = (cfg.observability_backend or "auto").strip().lower()
    if mode in {"none", "off", "false", "0", ""}:
        return None
    if mode in _BACKENDS:
        return _BACKENDS[mode](cfg)
    if mode == "local":
        return LocalJsonlBackend(cfg.trace_log_path) if cfg.trace_local_enabled else None
    if mode == "langfuse":
        if not _langfuse_configured(cfg):
            return None
        backend = LangfuseBackend(cfg)
        return backend if backend.ready else None
    if mode == "auto":
        backends: list[ObservabilityBackend] = []
        if cfg.trace_local_enabled:
            backends.append(LocalJsonlBackend(cfg.trace_log_path))
        if _langfuse_configured(cfg):
            lf = LangfuseBackend(cfg)
            if lf.ready:
                backends.append(lf)
        if not backends:
            return None
        return backends[0] if len(backends) == 1 else CompositeBackend(backends)
    log.warning("Unknown OBSERVABILITY_BACKEND=%r; skipping observability", mode)
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

    Provider-agnostic: returns `{score_name: (numeric_value, short_comment)}`.
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


def build_guardrail_report(state: dict, *, cached: bool = False) -> dict:
    """Build a compact, PII-free summary of all guardrail & PII checks for one RAG prompt."""
    blocked = bool(state.get("guardrail_blocked"))
    og = state.get("output_guardrail") or {}
    return {
        "allowed": not blocked,
        "blocked_reason": state.get("guardrail_reason"),
        "input_flags": list(state.get("guardrail_flags") or []),
        "input_pii": list(state.get("pii_findings") or []),
        "docs_sanitized": int(state.get("docs_sanitized") or 0),
        "output_flags": list(og.get("flags") or []),
        "output_pii": list(og.get("pii_findings") or []),
        "valid_citations": list(og.get("valid_citations") or []),
        "invalid_citations": list(og.get("invalid_citations") or []),
        "hallucinated_urls": list(og.get("hallucinated_urls") or []),
        "grounding_score": og.get("grounding_score"),
        "checks": list(state.get("guardrail_checks") or []),
        "cached": cached,
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
            "docs_sanitized": int(delta.get("docs_sanitized") or 0),
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
            "docs_sanitized": int(delta.get("docs_sanitized") or 0),
        }
    if node == "output_guardrail":
        return dict(delta.get("output_guardrail") or {})
    return {"keys": sorted(delta.keys())}


# ------------------------------------------------------------------ Provider-agnostic RagTracer
class RagTracer:
    """Context manager that traces one RAG run via the active ObservabilityBackend.

    When no backend is configured (e.g. Langfuse env vars are missing), every method is a
    zero-overhead no-op.
    """

    def __init__(
        self,
        question: str,
        *,
        corpus_size: int = 0,
        filters: dict | None = None,
        history_len: int = 0,
        cfg: Settings | None = None,
        backend: ObservabilityBackend | None = None,
        client: Any | None = None,
    ):
        self.cfg = cfg or default_settings
        self._question = question
        self.corpus_size = corpus_size
        self.filters = filters or {}
        self.history_len = history_len
        if backend is not None:
            self._backend: ObservabilityBackend | None = backend
        elif client is not None:
            self._backend = LangfuseBackend(self.cfg, client=client)
        else:
            self._backend = get_backend(self.cfg)
        self._handle: TraceHandle | None = None
        self._callbacks: list[Any] = []

    @property
    def active(self) -> bool:
        return self._backend is not None

    def __enter__(self) -> "RagTracer":
        if self._backend is None:
            return self
        meta = {
            "corpus_size": self.corpus_size,
            "history_turns": self.history_len,
            "filters": self.filters,
            "llm_model_id": self.cfg.llm_model_id,
            "embed_model_id": self.cfg.embed_model_id,
            "rerank_model_id": self.cfg.rerank_model_id,
            "rerank_stage2_model_id": self.cfg.rerank_stage2_model_id,
            "pii_mode": self.cfg.guardrails_pii_mode,
        }
        inp = {"question": redact_pii(self._question).text, "filters": self.filters}
        self._handle = self._backend.start_trace("rag_assistant", input=inp, metadata=meta, tags=["rag"])
        if self._handle is not None:
            self._callbacks = self._backend.langchain_callbacks()
        return self

    def langchain_callbacks(self) -> list[Any]:
        return list(self._callbacks)

    def record_node(self, node: str, delta: dict) -> None:
        if self._handle is None:
            return
        self._handle.record_span(f"node:{node}", summarize_node_object(node, delta))

    def finish(self, final_state: dict, error: str | None = None) -> dict[str, tuple[float, str]]:
        if self._handle is None:
            return {}
        scores = {} if error else compute_rag_scores(final_state)
        out = (
            {"error": error}
            if error
            else {
                "answer": redact_pii(str(final_state.get("answer") or "")).text,
                "references_count": len(final_state.get("references") or []),
                "summarized": final_state.get("summarized", 0),
                "guardrail_blocked": bool(final_state.get("guardrail_blocked")),
                "guardrail_report": build_guardrail_report(final_state),
            }
        )
        self._handle.finish(out)
        for name, (val, comment) in scores.items():
            self._handle.record_score(name, val, comment)
        return scores

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._backend is None:
            return
        if exc is not None and self._handle is not None:
            self.finish({}, error=f"{exc_type.__name__}: {exc}")
        self._backend.flush()


def trace_cached_ask(
    question: str,
    cached_event: dict,
    *,
    filters: dict | None = None,
    cfg: Settings | None = None,
    backend: ObservabilityBackend | None = None,
) -> None:
    """Record a lightweight per-prompt trace when an Ask query is served from QueryCache."""
    if backend is None:
        backend = get_backend(cfg)
    if backend is None:
        return
    redacted_q = redact_pii(question)
    base_rep = dict(cached_event.get("guardrail_report") or {})
    report = {
        "allowed": bool(base_rep.get("allowed", True)),
        "blocked_reason": base_rep.get("blocked_reason"),
        "input_flags": list(base_rep.get("input_flags") or []),
        "input_pii": list(base_rep.get("input_pii") or []),
        "docs_sanitized": int(base_rep.get("docs_sanitized") or 0),
        "output_flags": list(base_rep.get("output_flags") or []),
        "output_pii": list(base_rep.get("output_pii") or []),
        "valid_citations": list(base_rep.get("valid_citations") or []),
        "invalid_citations": list(base_rep.get("invalid_citations") or []),
        "hallucinated_urls": list(base_rep.get("hallucinated_urls") or []),
        "grounding_score": base_rep.get("grounding_score"),
        "checks": list(base_rep.get("checks") or []),
        "cached": True,
    }
    handle = backend.start_trace(
        "rag_assistant_cached",
        input={"question": redacted_q.text, "filters": filters or {}},
        metadata={"cache_hit": True},
        tags=["rag", "cache_hit"],
    )
    if handle is None:
        return
    handle.finish({
        "answer": redact_pii(str(cached_event.get("answer") or "")).text,
        "references_count": len(cached_event.get("references") or []),
        "summarized": cached_event.get("summarized", 0),
        "guardrail_blocked": not report["allowed"],
        "guardrail_report": report,
    })
    handle.record_score("guardrail_passed", 1.0 if report["allowed"] else 0.0, "cached")
    handle.record_score("pii_detected", 1.0 if redacted_q.had_pii else 0.0, "none" if not redacted_q.had_pii else "redacted")
    handle.record_score("cache_hit", 1.0, "served from QueryCache")
    backend.flush()


def trace_polish_run(
    draft: str,
    result_text: str,
    note: str,
    *,
    blocked: bool = False,
    pii_detected: bool = False,
    input_flags: list[str] | None = None,
    input_pii_findings: list[dict] | None = None,
    output_flags: list[str] | None = None,
    output_pii_findings: list[dict] | None = None,
    checks: list[dict] | None = None,
    cfg: Settings | None = None,
    backend: ObservabilityBackend | None = None,
    client: Any | None = None,
) -> None:
    """Record and flush a single trace for the Post tab's AI rewrite; no-op when unconfigured."""
    if backend is None:
        backend = LangfuseBackend(cfg or default_settings, client=client) if client is not None else get_backend(cfg)
    if backend is None:
        return
    report = {
        "allowed": not blocked,
        "blocked_reason": note if blocked else None,
        "input_flags": list(input_flags or []),
        "input_pii": list(input_pii_findings or []),
        "docs_sanitized": 0,
        "output_flags": list(output_flags or []),
        "output_pii": list(output_pii_findings or []),
        "valid_citations": [],
        "invalid_citations": [],
        "hallucinated_urls": [],
        "checks": list(checks or []),
        "cached": False,
    }
    inp = {"draft_chars": len(draft), "draft_preview": redact_pii(draft[:400]).text}
    out = {"result_chars": len(result_text), "note": note, "blocked": blocked, "guardrail_report": report}
    handle = backend.start_trace("polish_post", input=inp, tags=["post"])
    if handle is None:
        return
    handle.record_span("node:input_guardrail", {
        "blocked": blocked,
        "flags": report["input_flags"],
        "pii_findings": report["input_pii"],
    })
    if not blocked:
        handle.record_span("node:output_guardrail", {
            "flags": report["output_flags"],
            "pii_findings": report["output_pii"],
        })
    handle.finish(out)
    kept_links = 0.0 if ("dropped some links" in note or "empty" in note or blocked) else 1.0
    for name, val in (
        ("polish_guardrail_passed", 0.0 if blocked else 1.0),
        ("polish_pii_detected", 1.0 if pii_detected else 0.0),
        ("polish_links_preserved", kept_links),
    ):
        handle.record_score(name, val)
    backend.flush()
