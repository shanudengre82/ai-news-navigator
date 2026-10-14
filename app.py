"""AI News & Research Navigator — Hugging Face Space entry point (Gradio + ZeroGPU)."""

import os

if os.getenv("SPACE_ID"):  # on Spaces, keep caches and data in writable locations
    os.environ.setdefault("HF_HOME", "/tmp/hf-cache")
    os.environ.setdefault("DATA_DIR", "/tmp/ai-news-data")

import spaces  # noqa: E402  — must be imported before torch

import logging  # noqa: E402

from ai_news.config import settings  # noqa: E402
from ai_news.fetchers import arxiv, load_sources  # noqa: E402
from ai_news.ingest import refresh, sync_from_hub  # noqa: E402
from ai_news.llm import Reranker, load_chat_model, load_embeddings  # noqa: E402
from ai_news.rag.graph import build_graph  # noqa: E402
from ai_news.rag.nodes import RagNodes  # noqa: E402
from ai_news.search import DATE_WINDOWS  # noqa: E402
from ai_news.store import CorpusStore  # noqa: E402
from ai_news import scheduler  # noqa: E402
from ai_news.ui import CSS, build_ui, format_references  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

# ---- corpus (CPU) ----
store = CorpusStore(settings.data_dir, load_embeddings(settings.embed_model_id),
                    dataset_repo=settings.dataset_repo, hf_token=settings.hf_token)
store.pull()
store.load()

# ---- models: loaded once at import; ZeroGPU streams them to the GPU on the first @spaces.GPU call ----
llm = load_chat_model(settings.llm_backend, settings.llm_model_id, settings.llm_max_new_tokens, settings.hf_token)
reranker = Reranker(settings.rerank_model_id)
graph = build_graph(RagNodes(llm, store, reranker, live_search=arxiv.search))

sources_cfg = load_sources(settings.sources_file)
SOURCE_NAMES = (["arXiv"] if sources_cfg.get("arxiv", {}).get("enabled") else []) + \
               (["HF Daily Papers"] if sources_cfg.get("hf_daily_papers", {}).get("enabled") else []) + \
               [f["name"] for f in sources_cfg.get("rss", [])]


def run_refresh() -> str:
    if settings.refresh_mode == "pull":
        return sync_from_hub(store)
    return refresh(store, settings.sources_file, settings.lookback_days,
                   push=settings.push_to_hub and bool(settings.hf_token))


@spaces.GPU(duration=120)
def research_assistant(message: str, history: list, sources: list[str], source_type: str, window: str):
    """Ask the AI research assistant. It searches the corpus (hybrid BM25 + vectors), reranks, has an LLM
    judge relevance, rewrites the query or searches arXiv live if needed, and answers with cited references.

    Args:
        message: the research question
        history: previous chat messages ([] for a new conversation)
        sources: restrict to these sources (empty = all)
        source_type: "any", "paper" or "blog"
        window: date window, e.g. "Any time", "Last week", "Last month"
    """
    if not message or not message.strip():
        yield history, "_Ask a question to get ranked references._", "", ""
        return
    store.reload_if_stale()
    history = list(history or []) + [{"role": "user", "content": message},
                                     {"role": "assistant", "content": "⏳ Analyzing your question…"}]
    yield history, "_Searching…_", "", ""
    state = {"question": message,
             "history": [m["content"] for m in history[:-2] if m.get("role") == "user"],
             "filters": {"sources": sources or None, "source_type": source_type,
                         "days": DATE_WINDOWS.get(window)}}
    trace: list[str] = []
    final: dict = {}
    for update in graph.stream(state, stream_mode="updates"):
        for node, delta in update.items():
            trace += delta.get("trace", [])
            final.update(delta)
            history[-1]["content"] = f"⏳ {trace[-1].split('→')[0].replace('**', '').strip()} done, working…"
            yield history, "_Searching…_", "\n\n".join(f"{i}. {t}" for i, t in enumerate(trace, 1)), ""
    history[-1]["content"] = final.get("answer", "Something went wrong — please try again.")
    yield history, format_references(final.get("references", [])), \
        "\n\n".join(f"{i}. {t}" for i, t in enumerate(trace, 1)), ""


demo = build_ui(store, SOURCE_NAMES, research_assistant, run_refresh, pull_mode=settings.refresh_mode == "pull")

if settings.refresh_hours > 0:
    scheduler.start(run_refresh, settings.refresh_hours,
                    run_first_after=60 if settings.refresh_on_start or not store.items else None)
log.info("refresh mode: %s, every %s h", settings.refresh_mode, settings.refresh_hours)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(css=CSS, mcp_server=True)
