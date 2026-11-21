import time

from ai_news.cache import QueryCache
from ai_news.rag.graph import build_graph
from ai_news.rag.nodes import RagNodes, fast_analyze_query, format_docs_for_generate


def test_fast_analyze_query_extracts_window_and_type_without_llm():
    r = fast_analyze_query("recent papers on speculative decoding this week")
    assert r is not None
    assert r["days"] == 7 and r["source_type"] == "paper"
    assert r["queries"] == ["recent papers on speculative decoding this week"]

    r_blog = fast_analyze_query("OpenAI blog posts from this month")
    assert r_blog is not None and r_blog["days"] == 30 and r_blog["source_type"] == "blog"

    # Multi-question or overly long queries fall back to the LLM
    assert fast_analyze_query("What is LoRA? And how does it compare to QLoRA?") is None
    assert fast_analyze_query("a" * 130) is None


def test_fast_analyze_and_confidence_gated_grading_cut_llm_calls_to_one(store):
    calls: list[tuple[str, int | None]] = []

    class SpyLLM:
        def __init__(self, bound_max=None):
            self.bound_max = bound_max

        def bind(self, **kwargs):
            return SpyLLM(bound_max=kwargs.get("max_new_tokens"))

        def invoke(self, msgs):
            calls.append((msgs[0].content, self.bound_max))

            class R:
                content = "Concise answer citing [1] and [2]. **Start with** [1] because it introduces the core method."

            return R()

    class HighConfidenceReranker:
        def score(self, query, passages):
            return [0.92 for _ in passages]

    nodes = RagNodes(
        SpyLLM(),
        store,
        reranker=HighConfidenceReranker(),
        fast_analyze=True,
        grade_top=6,
        grade_skip_threshold=0.75,
    )
    out = build_graph(nodes).invoke({"question": "test-time compute scaling"})

    # Both analyze_query (fast-path) and grade_documents (confidence >= 0.75) skipped LLM -> ONLY 1 LLM call!
    assert len(calls) == 1
    prompt_text, max_tok = calls[0]
    assert max_tok == 420
    assert any("fast-path" in t for t in out["trace"])
    assert any("LLM skipped" in t for t in out["trace"])
    assert "[1]" in out["answer"]
    assert len(prompt_text) < 2000


def test_format_docs_for_generate_prefers_compact_digest():
    long_abstract = "Word " * 200
    cands = [
        {"title": "Paper A", "source": "arXiv", "published": "2026-10-01T00:00:00+00:00",
         "summary": long_abstract, "digest": "Compact 20-word digest of Paper A."},
        {"title": "Paper B", "source": "arXiv", "published": "2026-10-02T00:00:00+00:00",
         "summary": long_abstract, "digest": ""},
    ]
    out = format_docs_for_generate(cands, fallback_chars=120)
    assert "Compact 20-word digest of Paper A." in out
    assert len(out) < 400  # much shorter than 2 * 700-char abstracts


def test_query_cache_lru_ttl_and_normalization():
    cache = QueryCache(maxsize=2, ttl_seconds=60)
    k1 = QueryCache.make_key("  Speculative   Decoding  ", {}, corpus_tag=1)
    k1_same = QueryCache.make_key("speculative decoding", {}, corpus_tag=1)
    k2 = QueryCache.make_key("LoRA fine-tuning", {}, corpus_tag=1)
    k3 = QueryCache.make_key("RAG citations", {}, corpus_tag=1)

    cache.put(k1, {"answer": "A1"})
    assert cache.get(k1_same) == {"answer": "A1"}
    cache.put(k2, {"answer": "A2"})
    cache.put(k3, {"answer": "A3"})  # evicts LRU (k1)
    assert cache.get(k1) is None
    assert cache.get(k2) == {"answer": "A2"}
    assert cache.get(k3) == {"answer": "A3"}

    # Expired TTL returns None
    short = QueryCache(maxsize=4, ttl_seconds=0.01)
    short.put(k2, {"answer": "A2"})
    time.sleep(0.02)
    assert short.get(k2) is None
