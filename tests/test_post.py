from langchain_core.embeddings import DeterministicFakeEmbedding

from ai_news.post import (LINKEDIN_LIMIT, category_count, OTHER, camel_tags, draft_post, fold_preview, keep_links, takeaway,
                          topic_matches, topic_spec)
from ai_news.store import CorpusStore
from conftest import make_item


def _store(tmp_path, items):
    s = CorpusStore(tmp_path, DeterministicFakeEmbedding(size=32))
    s.add(items)
    return s


def test_topic_matches_keeps_the_week_and_blog_posts(tmp_path):
    seg = make_item("Open-vocabulary image segmentation with diffusion features",
                    "We segment images with a vision backbone.", age_days=2)
    seg.categories = ["cs.CV"]
    blog = make_item("Our new image segmentation model", "A blog post on image segmentation.",
                     source="Hugging Face Blog", source_type="blog", age_days=1)
    old = make_item("Image segmentation from last month", "Image segmentation again.", age_days=20)
    off = make_item("Tokenizer-free language modelling", "Byte-level language models.", age_days=3)
    s = _store(tmp_path, [seg, blog, old, off])
    name, spec = topic_spec("Computer vision")
    got = [it.id for it, _ in topic_matches(s, "image segmentation", spec["categories"])]
    assert old.id not in got
    assert blog.id in got and seg.id in got
    assert got[0] == seg.id  # keyword match plus the cs.CV nudge


def test_topic_matches_empty_query_or_week(tmp_path):
    s = _store(tmp_path, [make_item("Old", "old", age_days=30)])
    assert topic_matches(s, "anything") == [] and topic_matches(s, "  ") == []


def test_topic_spec_custom():
    assert topic_spec(OTHER, "  protein   design ") == ("protein design",
                                                         {"query": "protein design", "categories": [],
                                                          "tags": ["#ProteinDesign"]})
    assert camel_tags("graph neural networks") == ["#GraphNeuralNetworks"] and camel_tags("!!") == []


def test_takeaway_first_sentence_and_cut():
    it = make_item("T", "We propose X. It beats Y by 3 points.")
    assert takeaway(it) == "We propose X."
    bg = make_item("T", "Video models are popular. Yet they are slow. In this paper, we introduce FastV, a cache. "
                        "It is 3x faster.")
    assert takeaway(bg) == "We introduce FastV, a cache."
    assert takeaway(make_item("T", "Background only. More background.")) == "Background only."
    long = make_item("T", "word " * 100)
    out = takeaway(long, 50)
    assert out.endswith("…") and len(out) <= 51


def test_draft_post_contents_and_limit():
    a = make_item("Paper A", "A does vision. More text.", url="https://arxiv.org/abs/2611.00001")
    b = make_item("Post B", "B is a blog.", source="Hugging Face Blog", source_type="blog", url="https://hf.co/b")
    out = draft_post("Computer vision", [a, b], ["#ComputerVision"], total=412)
    assert out.startswith("What's new in computer vision this week")
    assert "412 new computer vision papers went up on arXiv in the past 7 days. Here are 2 worth" in out
    assert "1. Paper A (arXiv)\nA does vision.\nhttps://arxiv.org/abs/2611.00001" in out
    assert "2. Post B (Hugging Face Blog)" in out and out.endswith("#ComputerVision #AIResearch")
    assert "Here is one from the past 7 days" in draft_post("AI agents", [a], [])
    assert "What's new in AI agents" in draft_post("AI agents", [a], [])

    many = [make_item(f"Paper {i} " + "long title " * 10, "x " * 400, url=f"https://arxiv.org/abs/2611.{i:05d}")
            for i in range(30)]
    big = draft_post("Computer vision", many, ["#ComputerVision"])
    assert len(big) <= LINKEDIN_LIMIT and "1. Paper 0" in big


def test_draft_post_nothing_picked():
    assert "Tick a few" in draft_post("Computer vision", [], [])
    assert draft_post("", [make_item("Paper A")], []) == ""


def test_category_count(tmp_path):
    cv = make_item("CV paper", age_days=2)
    cv.categories = ["cs.CV", "cs.LG"]
    old = make_item("Old CV paper", age_days=20)
    old.categories = ["cs.CV"]
    s = _store(tmp_path, [cv, old, make_item("Other", age_days=1)])
    assert category_count(s, ["cs.CV"]) == 1 and category_count(s, []) is None


def test_fold_preview_and_keep_links():
    assert fold_preview("short") == "short"
    assert fold_preview("word " * 100).endswith("…") and len(fold_preview("word " * 100)) <= 211
    draft = "Intro\nhttps://a.org/1\nhttps://b.org/2"
    assert keep_links(draft, "Nicer https://a.org/1 and https://b.org/2") == (
        "Nicer https://a.org/1 and https://b.org/2", "")
    text, note = keep_links(draft, "Nicer https://a.org/1")
    assert text == draft and "dropped some links" in note
    assert keep_links(draft, "  ")[0] == draft
