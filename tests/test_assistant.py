import json
import os

from langchain_core.language_models import FakeListChatModel

from ai_news.assistant import message_text, render, run_events
from ai_news.rag.graph import build_graph
from ai_news.rag.nodes import RagNodes


def _graph(store, responses):
    return build_graph(RagNodes(FakeListChatModel(responses=responses), store))


def test_message_text_normalizes_gradio6_content():
    assert message_text("hi") == "hi"
    assert message_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a b"
    assert message_text({"text": "x"}) == "x"
    assert message_text(None) == ""


def test_run_events_streams_trace_then_answer(store):
    analyze = json.dumps({"intent": "x", "search_queries": ["reasoning"], "date_window_days": None,
                          "source_type": "any"})
    grade = json.dumps([{"doc": i, "relevant": True, "reason": "r"} for i in range(1, 9)])
    events = list(run_events(_graph(store, [analyze, grade, "Answer [1]."]), store,
                             "test-time compute", ["an earlier question"], {}))
    assert all("trace" in e for e in events)
    assert events[-1]["answer"] == "Answer [1]." and events[-1]["references"]
    assert len(events) == 6  # five nodes + final


def test_run_events_reports_errors_instead_of_raising(store):
    class Boom:
        def stream(self, *a, **k):
            raise ValueError("boom")

    assert list(run_events(Boom(), store, "q", [], {})) == [{"error": "ValueError: boom", "trace": []}]


def test_render_shows_error_and_references():
    history = [{"role": "user", "content": [{"type": "text", "text": "old"}]},
               {"role": "assistant", "content": "old answer"}]
    outs = list(render([{"error": "RuntimeError: cuda"}], history, "new q"))
    chat = outs[-1][0]
    assert len(chat) == 4 and chat[-2]["content"] == "new q"
    assert "⚠️" in chat[-1]["content"] and "RuntimeError: cuda" in chat[-1]["content"]

    ref = {"title": "T", "url": "https://arxiv.org/abs/1", "source": "arXiv", "published": "2026-10-01",
           "reason": "because"}
    outs = list(render([{"trace": ["**Analyze** → x"]}, {"answer": "A", "references": [ref], "trace": []}], [], "q"))
    assert outs[1][0][-1]["content"].startswith("⏳ Analyze done")
    assert outs[-1][0][-1]["content"] == "A" and "https://arxiv.org/abs/1" in outs[-1][1]


def test_store_lock_is_usable_in_forked_child(store):
    with store._lock:  # parent holds the lock while forking, like a running background refresh
        pid = os.fork()
        if pid == 0:
            ok = store._lock.acquire(timeout=2)
            os._exit(0 if ok else 1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


def test_quota_errors_get_a_friendly_hint():
    from ai_news.assistant import error_message
    msg = error_message("Error: 'You have exceeded your ZeroGPU quota (180s requested vs. 179s left).'")
    assert "Log in" in msg and "quota" in msg
