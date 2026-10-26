import json

from langchain_core.language_models import FakeListChatModel

from ai_news.rag import nodes as rag_nodes
from ai_news.rag.graph import build_graph
from ai_news.rag.nodes import RagNodes, extract_json
from conftest import make_item


def analyze(queries):
    return json.dumps({"intent": "x", "search_queries": queries, "date_window_days": None, "source_type": "any"})


def grade(n_docs, relevant):
    return json.dumps([{"doc": i, "relevant": i in relevant, "digest": f"digest {i}" if i in relevant else ""}
                       for i in range(1, n_docs + 1)])


def test_extract_json_tolerates_fences():
    assert extract_json('Sure!\n```json\n{"a": [1]}\n```') == {"a": [1]}
    assert extract_json('[{"doc": 1}]') == [{"doc": 1}]


def test_enough_relevant_goes_straight_to_generate(store):
    llm = FakeListChatModel(responses=[analyze(["reasoning", "compute", "model"]),
                                       grade(8, {1, 2, 3}), "Answer citing [1]."])
    out = build_graph(RagNodes(llm, store)).invoke({"question": "test-time compute"})
    assert out["answer"] == "Answer citing [1]."
    refs = out["references"]
    assert len(refs) == 5 and not any(r.get("weak") for r in refs[:3])   # 3 relevant + 2 padded to reach 5
    assert all(r["weak"] and r["rejected"] for r in refs[3:])
    steps = [t.split("**")[1] for t in out["trace"]]
    assert steps == ["Analyze", "Retrieve", "Rerank", "Grade", "Generate"]


def test_few_relevant_rewrites_then_live_searches(store):
    live = [make_item("Mixture of experts routing", "Sparse MoE routing for LLMs.", age_days=3)]
    llm = FakeListChatModel(responses=[
        analyze(["mixture experts"]), grade(8, set()),                    # first pass: nothing relevant
        json.dumps({"search_queries": ["sparse moe"]}),                   # rewrite; the re-retrieved items
        grade(8, {1}),                                                   # were all reviewed, so the next
        "MoE answer [1]."])                                              # LLM grading is the live result
    out = build_graph(RagNodes(llm, store, live_search=lambda q: live)).invoke({"question": "MoE routing"})
    steps = [t.split("**")[1] for t in out["trace"]]
    assert steps == ["Analyze", "Retrieve", "Rerank", "Grade", "Rewrite", "Retrieve", "Rerank", "Grade",
                     "Live arXiv search", "Rerank", "Grade", "Generate"]
    assert live[0].id in store.items                       # live results grow the corpus
    assert out["answer"] == "MoE answer [1]."
    refs = out["references"]
    assert refs[0]["title"] == "Mixture of experts routing" and not refs[0].get("weak")
    assert len(refs) == 5 and all(r["weak"] and r["rejected"] for r in refs[1:])   # padded, flagged
    assert out["summarized"] == 1                                    # the summary uses only the real match


def test_references_are_every_match_reviewed_first(store, monkeypatch):
    monkeypatch.setattr(rag_nodes, "GRADE_TOP", 2)
    monkeypatch.setattr(rag_nodes, "MIN_RELEVANT", 2)
    llm = FakeListChatModel(responses=[analyze(["model"]), grade(2, {1, 2}), "Summary [1] [2]."])
    out = build_graph(RagNodes(llm, store)).invoke({"question": "model"})
    refs = out["references"]
    assert len(refs) == len(store.items) and out["summarized"] == 2   # no fixed cap on the list
    assert [r.get("digest") for r in refs[:2]] == ["digest 1", "digest 2"]
    assert all(not r.get("digest") for r in refs[2:])                  # the rest were not reviewed


def test_ui_filters_restrict_retrieval(store):
    llm = FakeListChatModel(responses=[analyze(["model"]), grade(8, {1}), "a"])
    nodes = RagNodes(llm, store)
    out = build_graph(nodes).invoke({"question": "model", "filters": {"source_type": "blog"}})
    assert all(c["source_type"] == "blog" for c in out["candidates"])


def test_reranker_scores_become_match_scores_and_low_ones_pad(store):
    class Scores:  # one strong match, the rest far below MIN_RELEVANCE
        def score(self, query, passages):
            return [0.9 if p.startswith("Test-time") else 0.001 for p in passages]

    llm = FakeListChatModel(responses=[analyze(["compute"]), grade(5, {1}), "Answer [1]."])
    out = build_graph(RagNodes(llm, store, reranker=Scores())).invoke({"question": "test-time compute"})
    refs = out["references"]
    assert refs[0]["title"].startswith("Test-time") and refs[0]["relevance"] == 0.9 and not refs[0].get("weak")
    assert len(refs) == 5 and all(r["weak"] and r["relevance"] == 0.001 for r in refs[1:])
