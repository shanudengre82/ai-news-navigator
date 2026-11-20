"""AI News & Research Navigator — Hugging Face Space entry point (Gradio + ZeroGPU)."""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # tokenizers' thread pool is not fork-safe

if os.getenv("SPACE_ID"):  # on Spaces, keep caches and data in writable locations
    os.environ.setdefault("HF_HOME", "/tmp/hf-cache")
    os.environ.setdefault("DATA_DIR", "/tmp/ai-news-data")

import spaces  # noqa: E402  — must be imported before torch

import logging  # noqa: E402

from ai_news.config import settings  # noqa: E402
from ai_news.fetchers import arxiv, load_sources  # noqa: E402
from ai_news.ingest import refresh, sync_from_hub  # noqa: E402
from ai_news.llm import TwoStageReranker, load_chat_model, load_embeddings  # noqa: E402
from ai_news.rag.graph import build_graph  # noqa: E402
from ai_news.rag.nodes import RagNodes  # noqa: E402
from ai_news.store import CorpusStore  # noqa: E402
from ai_news import scheduler  # noqa: E402
from ai_news.assistant import message_text, render, run_events  # noqa: E402
from ai_news.guardrails import MAX_DRAFT_CHARS, check_input, check_output  # noqa: E402
from ai_news.observability import trace_polish_run  # noqa: E402
from ai_news.post import keep_links  # noqa: E402
from ai_news.rag.prompts import POLISH  # noqa: E402
from ai_news.ui import CSS, FIT_JS, FONTS_HEAD, REF_PAGE, STEPS_EMPTY, THEME_JS, build_ui, theme  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

# ---- corpus (CPU) ----
store = CorpusStore(settings.data_dir, load_embeddings(settings.embed_model_id),
                    dataset_repo=settings.dataset_repo, hf_token=settings.hf_token)
store.pull()
store.load()

# ---- models: loaded once at import; ZeroGPU streams them to the GPU on the first @spaces.GPU call ----
llm = load_chat_model(settings.llm_backend, settings.llm_model_id, settings.llm_max_new_tokens, settings.hf_token)
reranker = TwoStageReranker(
    stage1_model_id=settings.rerank_model_id,
    stage2_model_id=settings.rerank_stage2_model_id,
    stage1_top_k=settings.rerank_stage1_top_k,
    stage2_weight=settings.rerank_stage2_weight,
    mmr_lambda=settings.rerank_mmr_lambda,
)
graph = build_graph(RagNodes(
    llm, store, reranker, live_search=arxiv.search,
    guardrails_enabled=settings.guardrails_enabled,
    pii_mode=settings.guardrails_pii_mode,
    max_query_chars=settings.guardrails_max_query_chars,
))

sources_cfg = load_sources(settings.sources_file)
SOURCE_NAMES = (["arXiv"] if sources_cfg.get("arxiv", {}).get("enabled") else []) + \
               (["HF Daily Papers"] if sources_cfg.get("hf_daily_papers", {}).get("enabled") else []) + \
               [f["name"] for f in sources_cfg.get("rss", [])]


def run_refresh() -> str:
    if settings.refresh_mode == "pull":
        return sync_from_hub(store)
    return refresh(store, settings.sources_file, settings.lookback_days,
                   push=settings.push_to_hub and bool(settings.hf_token))


@spaces.GPU(duration=75)  # ZeroGPU reserves ~1.5x this from the visitor's daily quota
def _assistant_gpu(question: str, history_texts: list[str], filters: dict):
    """Runs in the ZeroGPU worker: yields plain dicts (picklable) and reports errors as events."""
    yield from run_events(graph, store, question, history_texts, filters)


def _guarded(events):
    """Turn ZeroGPU-level failures (quota, worker crash) into an error event instead of a bare toast."""
    try:
        yield from events
    except Exception as e:
        log.exception("ZeroGPU call failed")
        yield {"error": f"{type(e).__name__}: {e}"[:500]}


def research_assistant(message: str, history: list, sources: list[str], source_type: str, window: str,
                       sort: str = "relevance"):
    """Ask the AI research assistant. It searches the corpus (hybrid BM25 + vectors), keeps every item a
    cross-encoder finds relevant, has an LLM review and summarize the best ones, rewrites the query or searches
    arXiv live if needed, and returns a cited summary followed by every matching reference.

    Args:
        message: the research question
        history: (internal, from gr.State — not part of the public API) previous chat messages
        sources: only list references from these sources (empty = all); does not narrow the search
        source_type: "any", "paper" or "blog"; only narrows the reference list
        window: only list references from this window, e.g. "Any time", "Last week", "Last month"
        sort: order of the reference list: "relevance", "date", "oldest" or "upvotes"
    """
    history = history if isinstance(history, list) else []
    if not message or not message.strip():
        yield ("Type a question first, for example: *recent work on speculative decoding*.", "", STEPS_EMPTY,
               "", history, {}, REF_PAGE)
        return
    history_texts = [message_text(m.get("content")) for m in (history or [])
                     if isinstance(m, dict) and m.get("role") == "user"]
    # Filters narrow the list afterwards (see ui.refine), not the search: it finds everything that matches.
    view = {"sort": sort or "relevance", "source_type": source_type or "any", "window": window or "Any time",
            "sources": sources or None}
    yield from render(_guarded(_assistant_gpu(message, history_texts, {})), history, message, view)


@spaces.GPU(duration=60)
def _polish_gpu(draft: str) -> str:
    """Runs in the ZeroGPU worker: one LLM call that rewrites the weekly post."""
    from langchain_core.messages import HumanMessage
    return str(llm.invoke([HumanMessage(content=POLISH.format(draft=draft))]).content)


def polish_post(draft: str) -> tuple[str, str]:
    """Rewrite a draft post with the LLM; returns (text, note). Applies input/output guardrails and PII
    redaction, keeps the draft if the rewrite fails or drops a link, and traces the run in Langfuse."""
    if not draft or not draft.strip():
        return draft, "Build a draft first."
    ig = check_input(
        draft,
        enabled=settings.guardrails_enabled,
        pii_mode=settings.guardrails_pii_mode,
        max_chars=MAX_DRAFT_CHARS,
    )
    if not ig.allowed:
        note = ig.blocked_reason or "The draft was blocked by safety/privacy guardrails."
        trace_polish_run(draft, draft, note, blocked=True, pii_detected=ig.pii.had_pii)
        return draft, note
    try:
        raw = _polish_gpu(ig.sanitized_text)
        og = check_output(raw, enabled=settings.guardrails_enabled)
        out, note = keep_links(ig.sanitized_text, og.text)
        if ig.pii.had_pii and not note:
            note = f"Rewritten with AI (redacted PII: {', '.join(ig.pii.entity_types)}). Check it before posting."
        trace_polish_run(ig.sanitized_text, out, note, blocked=False,
                         pii_detected=ig.pii.had_pii or og.pii.had_pii)
        return out, note
    except Exception as e:
        log.exception("ZeroGPU rewrite failed")
        note = f"The rewrite did not run ({type(e).__name__}). The draft is unchanged; try again later."
        trace_polish_run(ig.sanitized_text, draft, note, blocked=False, pii_detected=ig.pii.had_pii)
        return draft, note


demo = build_ui(store, SOURCE_NAMES, research_assistant, run_refresh, pull_mode=settings.refresh_mode == "pull",
                polish_fn=polish_post)

if settings.refresh_hours > 0:
    scheduler.start(run_refresh, settings.refresh_hours,
                    run_first_after=60 if settings.refresh_on_start or not store.items else None)
log.info("refresh mode: %s, every %s h", settings.refresh_mode, settings.refresh_hours)

if __name__ == "__main__":
    # THEME_JS first: applies the saved colour theme before the page paints. FIT_JS: lets the page scroll
    # inside the Spaces iframe.
    head = f"<script>{THEME_JS}</script>{FONTS_HEAD}<script>{FIT_JS}</script>"
    demo.queue(default_concurrency_limit=2).launch(theme=theme(), css=CSS, head=head, mcp_server=True,
                                                   show_error=True)
