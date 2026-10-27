import re
from datetime import date, timedelta

import numpy as np
from langchain_core.embeddings import DeterministicFakeEmbedding

from ai_news.atlas import build_map, collection_stats, research_map
from ai_news.store import CorpusStore
from ai_news.ui import landing_html, map_intro_html, papers_chart, trend_html
from conftest import make_item

TOPIC_WORDS = ["diffusion video editing", "reinforcement learning policy", "retrieval augmented generation",
               "robot manipulation grasping", "speech recognition audio"]


def _papers(n):
    return [make_item(f"{TOPIC_WORDS[i % 5]} study number {i}", age_days=i % 20, upvotes=i % 3,
                      url=f"https://arxiv.org/abs/2610.{10000 + i}") for i in range(n)]


def _clustered_vectors(n, d=32, seed=0):
    """Five well-separated groups, matching the topic of each title in _papers."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(5, d)) * 4
    return np.stack([centres[i % 5] + rng.normal(size=d) for i in range(n)])


def test_empty_map():
    assert build_map([], None) == {"topics": [], "papers": [], "links": []}


def test_map_structure_and_topics():
    items = _papers(60)
    m = build_map(items, _clustered_vectors(60))
    assert len(m["papers"]) == 60
    n_topics = len(m["topics"])
    assert 4 <= n_topics <= 36
    assert all(0 <= p["g"] < n_topics for p in m["papers"])
    assert sum(t["size"] for t in m["topics"]) == 60
    assert all(t["label"] for t in m["topics"])
    pairs = [(a, b) for a, b, _ in m["links"]]
    assert all(0 <= a < b < 60 for a, b in pairs), "links are between known papers, ordered, never to themselves"
    assert len(pairs) == len(set(pairs)), "no duplicate links"
    # papers from the same group should land in the same topic and be each other's neighbours
    same = sum(items[a].title.split()[0] == items[b].title.split()[0] for a, b in pairs)
    assert same / len(pairs) > 0.9
    labels = " ".join(t["label"] for t in m["topics"])
    assert any(w in labels for w in ("diffusion", "reinforcement", "retrieval", "robot", "speech"))


def test_research_map_papers_only_capped_and_cached(tmp_path):
    store = CorpusStore(tmp_path, DeterministicFakeEmbedding(size=16))
    store.add(_papers(30) + [make_item("A lab blog post", source="Some Blog", source_type="blog")])
    m = research_map(store, max_nodes=20)
    assert len(m["papers"]) == 20
    assert all(p["s"] == "arXiv" for p in m["papers"])
    assert research_map(store, max_nodes=20) is m  # unchanged store: cached
    store.add([make_item("A brand new paper on diffusion", url="https://arxiv.org/abs/2610.99999")])
    assert research_map(store, max_nodes=20) is not m


def test_research_map_empty_store(tmp_path):
    m = research_map(CorpusStore(tmp_path, DeterministicFakeEmbedding(size=8)))
    assert m["papers"] == [] and m["topics"] == []


def test_collection_stats(store):
    today = date.today()
    s = collection_stats(store, days=30, today=today)
    assert s["total"] == 5 and s["papers"] == 4 and s["posts"] == 1
    assert len(s["days"]) == 30 and s["days"][-1][0] == today
    # fixture ages: 1 (blog, not counted), 2 (paper) this week; 10 (paper) last week; 40 and 200 days are outside
    assert (s["week"], s["prev_week"]) == (1, 1)
    assert sum(p for _, p, _ in s["days"]) == 2 and sum(b for _, _, b in s["days"]) == 1


def test_landing_html(store, tmp_path):
    empty = collection_stats(CorpusStore(tmp_path / "e", DeterministicFakeEmbedding(size=8)))
    assert "being gathered" in landing_html(empty)
    assert trend_html(empty) == ""
    out = landing_html(collection_stats(store))
    assert out.count('class="nv-tile"') == 4 and "In the collection" in out


def test_map_intro_html():
    out = map_intro_html([{"label": "retrieval · rag", "size": 3, "pos": [0, 0, 0]}])
    assert "Map of the papers" in out and 'data-nv-topic="0"' in out and "retrieval · rag" in out
    assert "nv-topics" not in map_intro_html([])


def test_trend_html(store):
    out = trend_html(collection_stats(store))
    assert "last 14 days" in out and "Last 7 days" in out and "The 7 days before" in out
    assert "same as the week before (1)" in out
    assert len(re.findall(r'class="nv-bar[" ]', out)) == 14


def test_papers_chart_bars_and_table():
    days = [(date(2026, 10, 1) + timedelta(days=i), i * 3, i % 2) for i in range(14)]
    out = papers_chart(days)
    assert len(re.findall(r'class="nv-bar[" ]', out)) == 14
    assert "so far today" in out  # the last day is today, still filling
    assert "<table>" in out
