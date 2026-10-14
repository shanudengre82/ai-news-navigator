import json

from langchain_core.language_models import FakeListChatModel

from ai_news.rag.graph import build_graph
from ai_news.rag.nodes import RagNodes, extract_json
from conftest import make_item


def analyze(queries):
    return json.dumps({"intent": "x", "search_queries": queries, "date_window_days": None, "source_type": "any"})


def grade(n_docs, relevant):
    return json.dumps([{"doc": i, "relevant": i in relevant, "reason": "r"} for i in range(1, n_docs + 1)])


def test_extract_json_tolerates_fences():
    assert extract_json('Sure!\n```json\n{"a": [1]}\n```') == {"a": [1]}
    assert extract_json('[{"doc": 1}]') == [{"doc": 1}]


def test_enough_relevant_goes_straight_to_generate(store):
    llm = FakeListChatModel(responses=[analyze(["reasoning", "compute", "model"]),
                                       grade(8, {1, 2, 3}), "Answer citing [1]."])
    out = build_graph(RagNodes(llm, store)).invoke({"question": "test-time compute"})
    assert out["answer"] == "Answer citing [1]."
    assert len(out["references"]) == 3
    steps = [t.split("**")[1] for t in out["trace"]]
    assert steps == ["Analyze", "Retrieve", "Rerank", "Grade", "Generate"]


def test_few_relevant_rewrites_then_live_searches(store):
    live = [make_item("Mixture of experts routing", "Sparse MoE routing for LLMs.", age_days=3)]
    llm = FakeListChatModel(responses=[
        analyze(["mixture experts"]), grade(8, set()),               # first pass: nothing relevant
        json.dumps({"search_queries": ["sparse moe"]}), grade(8, set()),  # rewrite: still nothing
        grade(8, {1}),                                                 # after live search
        "MoE answer [1]."])
    out = build_graph(RagNodes(llm, store, live_search=lambda q: live)).invoke({"question": "MoE routing"})
    steps = [t.split("**")[1] for t in out["trace"]]
    assert steps == ["Analyze", "Retrieve", "Rerank", "Grade", "Rewrite", "Retrieve", "Rerank", "Grade",
                     "Live arXiv search", "Rerank", "Grade", "Generate"]
    assert live[0].id in store.items                       # live results grow the corpus
    assert out["answer"] == "MoE answer [1]."


def test_ui_filters_restrict_retrieval(store):
    llm = FakeListChatModel(responses=[analyze(["model"]), grade(8, {1}), "a"])
    nodes = RagNodes(llm, store)
    out = build_graph(nodes).invoke({"question": "model", "filters": {"source_type": "blog"}})
    assert all(c["source_type"] == "blog" for c in out["candidates"])
