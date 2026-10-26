"""LangGraph state for the research assistant."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Candidate(TypedDict, total=False):
    id: str
    title: str
    url: str
    source: str
    source_type: str
    published: str
    summary: str
    upvotes: int
    score: float
    relevance: float            # the cross-encoder's probability that the item is relevant (0-1)
    digest: str                 # the LLM's short summary of the item, for the items it reviewed
    weak: bool                  # a low match, listed only to reach MIN_RESULTS
    rejected: bool              # the LLM judged it off-topic


class GraphState(TypedDict, total=False):
    question: str
    history: list[str]          # previous user questions, for follow-ups
    filters: dict               # UI filters: sources, source_type, days
    intent: str
    queries: list[str]
    source_type: str | None
    days: int | None
    candidates: list[Candidate]  # the retrieval pool of the latest retrieve / live search
    matches: list[Candidate]     # everything the reranker kept, across rewrites, best first
    ranked: list[Candidate]      # the best-scored items of every pool, kept or not (to pad the results)
    verdicts: dict               # id -> {"relevant": bool, "digest": str} for items the LLM reviewed
    relevant: list[Candidate]
    retries: int
    live_searched: bool
    answer: str
    references: list[Candidate]
    summarized: int             # how many of the references the summary was written from
    trace: Annotated[list[str], operator.add]
