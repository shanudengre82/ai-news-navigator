"""Research-assistant plumbing shared by app.py and the tests.

`run_events` runs inside the ZeroGPU worker (a forked process), so it only yields plain, picklable
dicts and never raises: ZeroGPU reports a worker exception to the UI as just its class name, so we
catch it here and send the real message back as an event instead.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator

from ai_news.store import CorpusStore
from ai_news.ui import format_references

log = logging.getLogger(__name__)

WORKING = "⏳ Analyzing your question…"


def message_text(content: Any) -> str:
    """Plain text of a Gradio chat message's content (Gradio 6 may send a list of content blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    if isinstance(content, (list, tuple)):
        return " ".join(t for t in (message_text(c) for c in content) if t)
    return str(content)


def run_events(graph, store: CorpusStore, question: str, history: list[str], filters: dict) -> Iterator[dict]:
    trace: list[str] = []
    try:
        store.reload_if_stale()
        state = {"question": question, "history": history, "filters": filters}
        final: dict = {}
        for update in graph.stream(state, stream_mode="updates"):
            for delta in update.values():
                trace += delta.get("trace", [])
                final.update(delta)
                yield {"trace": list(trace)}
        yield {"answer": final.get("answer") or "No answer was produced — please try again.",
               "references": final.get("references", []), "trace": list(trace)}
    except Exception as e:  # report instead of raising across the fork
        log.exception("assistant failed")
        yield {"error": f"{type(e).__name__}: {e}"[:500], "trace": list(trace)}


def _trace_md(trace: list[str]) -> str:
    return "\n\n".join(f"{i}. {t}" for i, t in enumerate(trace, 1))


def error_message(err: str) -> str:
    if "quota" in err.lower():
        return ("⚠️ The free ZeroGPU quota for this visitor is used up.\n\n"
                f"`{err}`\n\nLog in to huggingface.co (free accounts get more GPU time) or try again later. "
                "Keyword Search and Latest keep working without a GPU.")
    return (f"⚠️ The assistant hit an error: `{err}`\n\n"
            "Please try again in a moment. Keyword Search still works meanwhile.")


def render(events: Iterable[dict], history: list, question: str) -> Iterator[tuple]:
    """Turn assistant events into the UI outputs (chat, references_md, trace_md, cleared_textbox)."""
    base = list(history or []) + [{"role": "user", "content": question}]

    def chat(text: str) -> list:
        return base + [{"role": "assistant", "content": text}]

    yield chat(WORKING), "_Searching…_", "", ""
    for ev in events:
        trace = _trace_md(ev.get("trace", []))
        if "error" in ev:
            yield chat(error_message(ev["error"])), "_No references (the request failed)._", trace, ""
            return
        if "answer" in ev:
            yield chat(ev["answer"]), format_references(ev["references"]), trace, ""
            return
        last = ev["trace"][-1] if ev.get("trace") else ""
        yield chat(f"⏳ {last.split('→')[0].replace('**', '').strip()} done, working…"), "_Searching…_", trace, ""
