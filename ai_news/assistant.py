"""Research-assistant plumbing shared by app.py and the tests.

`run_events` runs inside the ZeroGPU worker (a forked process), so it only yields plain, picklable
dicts and never raises: ZeroGPU reports a worker exception to the UI as just its class name, so we
catch it here and send the real message back as an event instead.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator

from ai_news.store import CorpusStore
from ai_news.ui import answer_md, reading_list_html, sieve_html

log = logging.getLogger(__name__)

WORKING = "Reading your question…"


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


COUNT_FIELD = {"retrieve": "candidates", "rerank": "candidates", "live_search": "candidates",
               "grade_documents": "relevant", "rewrite_query": "queries", "generate": "references"}


def step_of(node: str, delta: dict) -> dict:
    field = COUNT_FIELD.get(node)
    return {"node": node, "count": len(delta.get(field) or []) if field else None}


def run_events(graph, store: CorpusStore, question: str, history: list[str], filters: dict) -> Iterator[dict]:
    """Events: {"total", "steps", "trace"} per graph node, then a final one with "answer" + "references"
    (or "error"). Steps carry real counts from the graph state, for the sieve visualization."""
    trace: list[str] = []
    steps: list[dict] = []
    total = 0
    try:
        store.reload_if_stale()
        total = len(store.items)
        state = {"question": question, "history": history, "filters": filters}
        final: dict = {}
        for update in graph.stream(state, stream_mode="updates"):
            for node, delta in update.items():
                trace += delta.get("trace", [])
                steps.append(step_of(node, delta))
                final.update(delta)
                yield {"total": total, "steps": list(steps), "trace": list(trace)}
        yield {"total": total, "steps": list(steps), "trace": list(trace),
               "answer": final.get("answer") or "No answer was produced. Please ask again.",
               "references": final.get("references", [])}
    except Exception as e:  # report instead of raising across the fork
        log.exception("assistant failed")
        yield {"error": f"{type(e).__name__}: {e}"[:500], "total": total, "steps": list(steps),
               "trace": list(trace)}


def error_message(err: str) -> str:
    if "quota" in err.lower():
        return ("**The free GPU time for this visitor is used up.** Log in to huggingface.co for more GPU "
                "time, or try again later. Search and Latest keep working without a GPU.\n\n"
                f"Details: `{err}`")
    return (f"**The assistant stopped with an error.** Ask again in a moment; Search still works.\n\n"
            f"Details: `{err}`")


def _earlier(history: list) -> list[tuple[str, str]]:
    """(question, answer) pairs from the conversation state."""
    pairs, q = [], None
    for m in history or []:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "user":
            q = message_text(m.get("content"))
        elif m.get("role") == "assistant" and q is not None:
            pairs.append((q, message_text(m.get("content"))))
            q = None
    return pairs


def render(events: Iterable[dict], history: list, question: str) -> Iterator[tuple]:
    """Turn assistant events into the UI outputs: (answer_md, reading_list_html, sieve_html,
    cleared_textbox, conversation_state)."""
    earlier = _earlier(history)
    base = list(history or []) + [{"role": "user", "content": question}]

    def out(body: str, refs_html: str, sieve: str, final: bool = False) -> tuple:
        conv = base + [{"role": "assistant", "content": body}] if final else list(history or [])
        return answer_md(question, body, earlier), refs_html, sieve, "", conv

    yield out(f"_{WORKING}_", "", "")
    for ev in events:
        total, steps, trace = ev.get("total", 0), ev.get("steps", []), ev.get("trace", [])
        if "error" in ev:
            yield out(error_message(ev["error"]), "", sieve_html(total, steps, trace), final=True)
            return
        if "answer" in ev:
            yield out(ev["answer"], reading_list_html(ev["references"]), sieve_html(total, steps, trace), final=True)
            return
        yield out("_Working through the collection…_", "", sieve_html(total, steps, trace, working=True))
