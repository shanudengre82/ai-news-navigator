"""Corrective / adaptive RAG graph:

analyze → retrieve → rerank → grade ─┬─(≥3 relevant)──────────────→ generate
                                      ├─(few, first try)→ rewrite → retrieve …
                                      └─(still few)→ live arXiv search → rerank …
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ai_news.rag.nodes import RagNodes
from ai_news.rag.state import GraphState


def build_graph(nodes: RagNodes):
    g = StateGraph(GraphState)
    g.add_node("analyze_query", nodes.analyze_query)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("rerank", nodes.rerank)
    g.add_node("grade_documents", nodes.grade_documents)
    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("live_search", nodes.live_search)
    g.add_node("generate", nodes.generate)

    g.add_edge(START, "analyze_query")
    g.add_edge("analyze_query", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "grade_documents")
    g.add_conditional_edges("grade_documents", nodes.route_after_grade,
                            ["generate", "rewrite_query", "live_search"])
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("live_search", "rerank")
    g.add_edge("generate", END)
    return g.compile()
