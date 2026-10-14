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
    reason: str


class GraphState(TypedDict, total=False):
    question: str
    history: list[str]          # previous user questions, for follow-ups
    filters: dict               # UI filters: sources, source_type, days
    intent: str
    queries: list[str]
    source_type: str | None
    days: int | None
    candidates: list[Candidate]
    relevant: list[Candidate]
    retries: int
    live_searched: bool
    answer: str
    references: list[Candidate]
    trace: Annotated[list[str], operator.add]
