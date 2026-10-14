import json

from ai_news.fetchers import arxiv, hf_papers, rss
from ai_news.schema import canonical_url, make_id
from conftest import FIXTURES


def test_canonical_url_unifies_arxiv_variants():
    ids = {make_id(u) for u in ["http://arxiv.org/abs/2610.16201v1", "https://arxiv.org/pdf/2610.16201",
                                "https://huggingface.co/papers/2610.16201"]}
    assert len(ids) == 1
    assert canonical_url("https://x.com/a/?utm_source=rss") == "https://x.com/a"


def test_parse_arxiv_feed():
    items = arxiv.parse_feed((FIXTURES / "arxiv.xml").read_bytes())
    assert len(items) == 2
    it = items[0]
    assert it.source == "arXiv" and it.source_type == "paper"
    assert it.url.startswith("https://arxiv.org/abs/") and "v1" not in it.url
    assert it.pdf_url.startswith("https://arxiv.org/pdf/")
    assert it.authors and it.summary and "cs.CL" in it.categories
    assert it.published.startswith("20")


def test_parse_hf_daily_papers():
    items = hf_papers.parse(json.loads((FIXTURES / "daily_papers.json").read_text()))
    assert items and all(i.source == "HF Daily Papers" for i in items)
    assert items[0].url.startswith("https://arxiv.org/abs/")
    assert isinstance(items[0].upvotes, int)


def test_parse_rss_strips_html_and_skips_undated():
    items = rss.parse_feed((FIXTURES / "blog.xml").read_bytes(), "Example")
    assert len(items) == 1
    it = items[0]
    assert it.title == "Scaling test-time compute with verifiers"
    assert "<" not in it.summary and "reasoning" in it.summary
    assert it.source_type == "blog" and it.categories == ["reasoning"]


def test_parse_arxiv_rss_skips_replacements():
    items = arxiv.parse_rss((FIXTURES / "arxiv_rss.xml").read_bytes())
    assert [i.title for i in items] == ["Speculative Decoding at Scale"]
    it = items[0]
    assert it.summary == "We speed up LLM inference with draft models."
    assert it.authors == ["Ada Lovelace", "Alan Turing"]
    assert it.pdf_url == "https://arxiv.org/pdf/2610.11111"
