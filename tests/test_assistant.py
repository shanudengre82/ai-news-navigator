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
    assert all("trace" in e and "steps" in e for e in events)
    assert [st["node"] for st in events[-1]["steps"]] == ["analyze_query", "retrieve", "rerank",
                                                          "grade_documents", "generate"]
    assert events[-1]["total"] == len(store.items)
    assert events[-1]["answer"] == "Answer [1]." and events[-1]["references"]
    assert len(events) == 6  # five nodes + final


def test_run_events_reports_errors_instead_of_raising(store):
    class Boom:
        def stream(self, *a, **k):
            raise ValueError("boom")

    events = list(run_events(Boom(), store, "q", [], {}))
    assert len(events) == 1 and events[0]["error"] == "ValueError: boom" and events[0]["steps"] == []


def test_render_shows_error_and_reading_list():
    history = [{"role": "user", "content": [{"type": "text", "text": "old q"}]},
               {"role": "assistant", "content": "old answer"}]
    outs = list(render([{"error": "RuntimeError: cuda", "total": 10, "steps": []}], history, "new q"))
    assert all(len(o) == 5 for o in outs)
    answer, refs, sieve, box, conv = outs[-1]
    assert "RuntimeError: cuda" in answer and "new q" in answer
    assert "Earlier in this session (1)" in answer and "old answer" in answer
    assert box == "" and conv[-2]["content"] == "new q" and len(conv) == 4

    ref = {"title": "T", "url": "https://arxiv.org/abs/1", "source": "arXiv", "source_type": "paper",
           "published": "2026-10-01T00:00:00+00:00", "reason": "because"}
    steps = [{"node": "analyze_query", "count": None}, {"node": "retrieve", "count": 30}]
    events = [{"total": 2000, "steps": steps[:1], "trace": ["**Analyze** → x"]},
              {"total": 2000, "steps": steps, "trace": ["**Analyze** → x", "**Retrieve** → 30"]},
              {"total": 2000, "steps": steps, "trace": [], "answer": "A [1]", "references": [ref]}]
    outs = list(render(events, [], "q"))
    assert "Reading your question" in outs[1][2]                 # sieve shows the current step
    assert outs[1][4] == []                                      # state unchanged until final answer
    answer, refs, sieve, _, conv = outs[-1]
    assert "A [1]" in answer and "https://arxiv.org/abs/1" in refs and "because" in refs
    assert "2,000" in sieve and ">30<" in sieve
    assert conv[-1] == {"role": "assistant", "content": "A [1]"}


def test_sieve_bars_shrink_with_counts():
    import re

    from ai_news.ui import sieve_html
    html = sieve_html(3000, [{"node": "retrieve", "count": 30}, {"node": "rerank", "count": 8},
                             {"node": "grade_documents", "count": 3}])
    kept = [float(w) for w in re.findall(r'class="kept" style="width:([\d.]+)%', html)]
    assert kept == sorted(kept, reverse=True) and kept[0] == 100.0
    assert html.count('class="was"') == 3                       # each step shows what it filtered out
    assert sieve_html(10, []) == ""


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
