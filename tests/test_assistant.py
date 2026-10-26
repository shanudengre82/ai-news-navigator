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
    grade = json.dumps([{"doc": i, "relevant": True, "digest": "d"} for i in range(1, 9)])
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


def test_render_shows_error_and_references():
    history = [{"role": "user", "content": [{"type": "text", "text": "old q"}]},
               {"role": "assistant", "content": "old answer"}]
    outs = list(render([{"error": "RuntimeError: cuda", "total": 10, "steps": []}], history, "new q"))
    assert all(len(o) == 7 for o in outs)
    answer, refs, sieve, box, conv, results, shown = outs[-1]
    assert "RuntimeError: cuda" in answer and "new q" in answer
    assert "Earlier in this session (1)" in answer and "old answer" in answer
    assert box == "" and conv[-2]["content"] == "new q" and len(conv) == 4 and results["refs"] == []

    ref = {"title": "T", "url": "https://arxiv.org/abs/1", "source": "arXiv", "source_type": "paper",
           "published": "2026-10-01T00:00:00+00:00", "digest": "because", "summary": "abstract"}
    other = {**ref, "title": "U", "url": "https://arxiv.org/abs/2", "digest": "", "summary": "the abstract of U"}
    steps = [{"node": "analyze_query", "count": None}, {"node": "retrieve", "count": 30}]
    events = [{"total": 2000, "steps": steps[:1], "trace": ["**Analyze** → x"]},
              {"total": 2000, "steps": steps, "trace": ["**Analyze** → x", "**Retrieve** → 30"]},
              {"total": 2000, "steps": steps, "trace": [], "answer": "A [1]", "references": [ref, other],
               "summarized": 1}]
    outs = list(render(events, [], "q"))
    assert "Reading your question" in outs[1][2]                 # steps panel shows the current step
    assert outs[1][4] == []                                      # state unchanged until final answer
    answer, refs, sieve, _, conv, results, shown = outs[-1]
    assert "A [1]" in answer and "Summary of the 1 most relevant of 2 references" in answer
    assert "https://arxiv.org/abs/1" in refs and "because" in refs and "Model summary" in refs
    assert "the abstract of U" in refs and "From the abstract" in refs     # unreviewed: abstract instead
    assert "2,000" in sieve and ">30<" in sieve
    assert conv[-1] == {"role": "assistant", "content": "A [1]"}
    assert results == {"refs": [ref, other], "summarized": 1}


def test_references_sort_filter_and_page_keep_numbers():
    from ai_news.ui import has_more, references_html, view_refs
    refs = [{"title": f"T{i}", "url": f"https://x/{i}", "source": "arXiv" if i % 2 else "OpenAI Blog",
             "source_type": "paper" if i % 2 else "blog", "published": f"2026-10-{10 + i:02d}T00:00:00+00:00",
             "upvotes": i, "summary": "s"} for i in range(1, 7)]
    assert [n for n, _ in view_refs(refs)] == [1, 2, 3, 4, 5, 6]
    assert [n for n, _ in view_refs(refs, sort="date")] == [6, 5, 4, 3, 2, 1]
    assert [n for n, _ in view_refs(refs, sort="oldest")] == [1, 2, 3, 4, 5, 6]
    assert [n for n, _ in view_refs(refs, sort="upvotes", source_type="blog")] == [6, 4, 2]
    assert [n for n, _ in view_refs(refs, sources=["arXiv"])] == [1, 3, 5]
    page = references_html(refs, shown=2, source_type="blog", summarized=2)
    assert page.count("<li") == 2 and "[2]" in page and "[4]" in page and "showing 2" in page
    assert "(1 cited in the summary is hidden)" in page               # [1] is a paper, filtered out
    assert has_more(refs, 2, source_type="blog") and not has_more(refs, 3, source_type="blog")


def test_sieve_bars_shrink_with_counts():
    import re

    from ai_news.ui import STEPS_EMPTY, sieve_html
    html = sieve_html(3000, [{"node": "retrieve", "count": 300}, {"node": "rerank", "count": 40},
                             {"node": "grade_documents", "count": 8}])
    kept = [float(w) for w in re.findall(r'class="kept" style="width:([\d.]+)%', html)]
    assert kept == sorted(kept, reverse=True) and kept[0] == 100.0
    assert html.count('class="was"') == 2                       # retrieve and rerank filter; grading reviews
    assert sieve_html(10, []) == STEPS_EMPTY


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


def test_references_show_match_scores_and_warn_on_low_matches():
    from ai_news.ui import references_html
    base = {"url": "https://x/1", "source": "arXiv", "source_type": "paper",
            "published": "2026-10-01T00:00:00+00:00", "summary": "s"}
    refs = [{**base, "title": "Good", "relevance": 0.87},
            {**base, "title": "Close", "relevance": 0.012, "weak": True},
            {**base, "title": "Off", "relevance": 0.001, "weak": True, "rejected": True}]
    html = references_html(refs)
    assert "Match 87%" in html and "Low match 1%" in html and "Low match &lt;1%" not in html
    assert "Low match <1% · the model judged it off-topic" in html
    assert "Only 1 reference matched your question well. 2 more with low match scores are shown" in html
    assert "nv-warn" not in references_html(refs[:1])
    assert "No reference matched your question well" in references_html(refs[1:])
