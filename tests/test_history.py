from datetime import datetime, timezone

from ai_news.history import (MAX_ENTRIES, MAX_REFS, REF_SUMMARY_CHARS, add_entry, find, group_entries,
                             history_html, make_entry, restore)
from ai_news.ui import references_html


def ts(*args) -> float:
    return datetime(*args, tzinfo=timezone.utc).timestamp()


def entry(q, when):
    return make_entry(q, f"Answer to {q}", {"refs": [], "summarized": 0}, now=when)


def labels(history, now, tz=0):
    return [(label, is_open, [e["q"] for e in es]) for label, is_open, es in group_entries(history, now, tz)]


def test_groups_by_day_week_month_and_year():
    now = ts(2026, 11, 7, 12)  # a Saturday; that week's Monday is 2 Nov
    hist = [entry("today", ts(2026, 11, 7, 10)), entry("yesterday", ts(2026, 11, 6, 9)),
            entry("wednesday", ts(2026, 11, 4, 9)), entry("last sunday", ts(2026, 11, 1, 9)),
            entry("last week", ts(2026, 10, 27, 9)), entry("september", ts(2026, 9, 3, 9)),
            entry("last january", ts(2025, 1, 15, 9)), entry("long ago", ts(2023, 5, 1, 9))]
    assert labels(hist, now) == [
        ("Today", True, ["today"]), ("Yesterday", True, ["yesterday"]), ("Wednesday", True, ["wednesday"]),
        ("Last week", False, ["last sunday", "last week"]), ("September 2026", False, ["september"]),
        ("January 2025", False, ["last january"]), ("2023", False, ["long ago"])]
    later = ts(2026, 11, 25, 12)
    assert labels([entry("early november", ts(2026, 11, 5, 9))], later) == [
        ("Earlier in November", False, ["early november"])]


def test_groups_use_the_visitors_time_zone():
    now = ts(2026, 11, 7, 0, 30)
    hist = [entry("late", ts(2026, 11, 6, 23, 50))]
    assert labels(hist, now)[0][0] == "Yesterday"
    assert labels(hist, now, tz=-60)[0][0] == "Today"  # UTC+1: both are on 7 Nov locally


def test_add_entry_dedupes_and_caps():
    a, b = entry("Speculative decoding", 1), entry("RAG", 2)
    again = entry("speculative  decoding", 3)
    hist = add_entry(add_entry(add_entry([], a), b), again)
    assert [e["q"] for e in hist] == ["speculative decoding", "RAG"]
    for i in range(MAX_ENTRIES + 10):
        hist = add_entry(hist, entry(f"q{i}", i))
    assert len(hist) == MAX_ENTRIES and hist[0]["q"] == f"q{MAX_ENTRIES + 9}"


def test_make_entry_trims_refs():
    refs = [{"id": str(i), "title": f"T{i}", "url": "https://x", "source": "arXiv", "source_type": "paper",
             "published": "2026-11-01T00:00:00+00:00", "summary": "s" * 2000, "relevance": 0.5, "extra": 1}
            for i in range(MAX_REFS + 5)]
    e = make_entry("q", "a", {"refs": refs, "summarized": 4})
    assert len(e["refs"]) == MAX_REFS and e["total"] == MAX_REFS + 5
    assert len(e["refs"][0]["summary"]) == REF_SUMMARY_CHARS and "extra" not in e["refs"][0]


def test_restore_round_trips_into_the_reference_list():
    refs = [{"id": "1", "title": "Paper one", "url": "https://arxiv.org/abs/1", "source": "arXiv",
             "source_type": "paper", "published": "2026-11-01T00:00:00+00:00", "summary": "About it.",
             "relevance": 0.8, "score": 0.8, "digest": "Does X."}] * 3
    e = make_entry("my question", "The answer [1].", {"refs": refs, "summarized": 1}, now=ts(2026, 11, 4, 14, 5))
    body, results, conv, note = restore(e)
    assert body == "The answer [1]." and conv[0] == {"role": "user", "content": "my question"}
    assert "Saved answer from Wed 4 Nov, 14:05." in note and "Press Ask" in note
    assert "Paper one" in references_html(results["refs"], summarized=results["summarized"])
    assert find([e], e["id"]) is e and find([e], "nope") is None


def test_history_html_escapes_and_marks_active():
    e = entry("<b>bold</b> question", ts(2026, 11, 7, 10))
    out = history_html([e], now=ts(2026, 11, 7, 12), active=e["id"])
    assert "&lt;b&gt;bold&lt;/b&gt;" in out and "<b>bold" not in out
    assert f'data-nv-entry="{e["id"]}"' in out and "nv-hist-q on" in out and "10:00" in out
    assert "<details open><summary>Today</summary>" in out
    assert "saved in this browser only" in history_html([])
