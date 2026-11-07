"""Weekly LinkedIn roundup: the last week's papers and posts on a topic, and a ready-to-edit draft post.

Everything here runs on the CPU from the live collection: topic matching uses the stored vectors and BM25,
and the draft is a template. Only the optional "Rewrite with AI" step (in app.py) uses the GPU.
"""

from __future__ import annotations

import math
import re

import numpy as np

from ai_news.schema import NewsItem
from ai_news.search import passes_filters
from ai_news.store import CorpusStore, faiss_id

WEEK_DAYS = 7
LINKEDIN_LIMIT = 3000   # characters LinkedIn allows in a post
FOLD_CHARS = 210        # roughly what LinkedIn shows before "see more"
DEFAULT_PICKS = 5
RRF_K = 60              # same fusion constant as the assistant's retrieval (rag/nodes.py)
CATEGORY_BONUS = 0.012  # about the RRF weight of a top-20 rank in one ranking
OTHER = "Other"

TOPICS: dict[str, dict] = {
    "Computer vision": {
        "query": "computer vision image video object detection segmentation visual recognition 3D scene",
        "categories": ["cs.CV"], "tags": ["#ComputerVision", "#DeepLearning"]},
    "Time series forecasting": {
        "query": "time series forecasting temporal sequence prediction anomaly detection",
        "categories": [], "tags": ["#TimeSeries", "#Forecasting"]},
    "LLMs and language": {
        "query": "large language models LLM text generation natural language processing pretraining",
        "categories": ["cs.CL"], "tags": ["#LLM", "#NLP"]},
    "AI agents": {
        "query": "AI agents tool use web browsing multi-agent planning autonomous agent",
        "categories": ["cs.MA"], "tags": ["#AIAgents", "#LLM"]},
    "Reinforcement learning": {
        "query": "reinforcement learning policy optimization reward RLHF",
        "categories": [], "tags": ["#ReinforcementLearning"]},
    "Multimodal models": {
        "query": "multimodal vision-language model image text audio video understanding",
        "categories": [], "tags": ["#Multimodal", "#VisionLanguage"]},
    "Retrieval and RAG": {
        "query": "retrieval-augmented generation RAG dense retrieval search reranking embeddings",
        "categories": ["cs.IR"], "tags": ["#RAG", "#InformationRetrieval"]},
    "Efficient ML": {
        "query": "efficient inference quantization distillation pruning speculative decoding KV cache",
        "categories": [], "tags": ["#EfficientAI", "#MLOps"]},
    "Safety and alignment": {
        "query": "AI safety alignment jailbreak red teaming interpretability harmful outputs",
        "categories": [], "tags": ["#AISafety", "#Alignment"]},
}
COMMON_TAG = "#AIResearch"


def topic_spec(topic: str, custom: str = "") -> tuple[str, dict]:
    """The topic's display name and search spec; "Other" uses the visitor's own words."""
    if topic in TOPICS:
        return topic, TOPICS[topic]
    name = " ".join((custom or "").split())[:80]
    return name, {"query": name, "categories": [], "tags": camel_tags(name)}


def camel_tags(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9]+", text)
    return ["#" + "".join(w[:1].upper() + w[1:] for w in words)] if words else []


def week_items(store: CorpusStore, days: int = WEEK_DAYS) -> list[NewsItem]:
    """Papers and blog posts published in the last `days` days."""
    return [it for it in store.all_items() if passes_filters(it, None, None, days)]


def topic_matches(store: CorpusStore, query: str, categories: list[str] | None = None, days: int = WEEK_DAYS,
                  limit: int = 25) -> list[tuple[NewsItem, float]]:
    """The week's items on a topic, best first: meaning (stored vectors) and keywords (BM25) fused by rank,
    with a nudge for matching arXiv categories and a tie-break on upvotes."""
    week = {it.id: it for it in week_items(store, days)}
    if not week or not query.strip():
        return []
    fused: dict[str, float] = {}

    def add(ranked_ids: list[str]) -> None:
        for rank, i in enumerate(ranked_ids):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)

    with_vec = [i for i in week if faiss_id(i) in store._by_faiss_id]
    if with_vec:
        qv = np.asarray(store.embeddings.embed_query(query), dtype="float32")
        sims = store.vectors.get(with_vec) @ qv
        order = np.argsort(-sims)
        # the vector ranking always has every item; keep only its upper half so the tail adds no weight
        add([with_vec[k] for k in order[: max(limit, len(order) // 2)]])
    add([it.id for it, _ in store.bm25_scores(query) if it.id in week])
    cats = set(categories or [])
    scores = {i: s + (CATEGORY_BONUS if cats & set(week[i].categories) else 0.0)
              + 0.001 * math.log1p(week[i].upvotes) for i, s in fused.items()}
    best = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [(week[i], scores[i]) for i in best]


CONTRIBUTION = re.compile(r"\b(?:we|this (?:paper|work)|our (?:method|approach|model))\s+(?:\w+\s+){0,2}?"
                          r"(?:propose|introduce|present|develop|show|release|demonstrate|find)s?\b", re.I)


def takeaway(it: NewsItem, chars: int = 220) -> str:
    """The sentence that states the contribution ("We propose…") if there is one among the first few,
    else the first sentence; cut on a word boundary."""
    text = " ".join(it.summary.split())
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if text else [""]
    first = next((x for x in sentences[:5] if CONTRIBUTION.search(x)), sentences[0])
    # "To address this, we present X." leans on the sentence before it; start at "We" instead
    lead = re.match(r"[^,]{1,40},\s*(?=we\b)", first, re.I)
    if lead:
        first = first[lead.end()].upper() + first[lead.end() + 1:]
    if len(first) <= chars:
        return first
    return first[:chars].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def category_count(store: CorpusStore, categories: list[str] | None, days: int = WEEK_DAYS) -> int | None:
    """How many papers in these arXiv categories came out this week: a real total for the post's opening line.
    None when the topic has no category (a ranked match list has no honest total)."""
    cats = set(categories or [])
    if not cats:
        return None
    return sum(1 for it in week_items(store, days) if it.source_type == "paper" and cats & set(it.categories))


def in_sentence(topic: str) -> str:
    """'Computer vision' -> 'computer vision' mid-sentence, but 'LLMs and language' and 'AI agents' stay."""
    return topic[:1].lower() + topic[1:] if topic[1:2].islower() else topic


def draft_post(topic: str, picked: list[NewsItem], tags: list[str], total: int | None = None,
               limit: int = LINKEDIN_LIMIT) -> str:
    """A LinkedIn post listing the picked items, kept within LinkedIn's length limit."""
    if not topic:
        return ""
    mid = in_sentence(topic)
    head = f"What's new in {mid} this week\n\n"
    if not picked:
        return head + "Tick a few papers or posts on the left to build the post."
    some = "Here is one" if len(picked) == 1 else f"Here are {len(picked)}"
    if total:
        lead = f"{total:,} new {mid} papers went up on arXiv in the past 7 days. {some} worth your time:\n\n"
    else:
        lead = f"{some} from the past 7 days worth your time:\n\n"
    tail = "\n\nWhich one would you try first?\n\n" + " ".join(dict.fromkeys(tags + [COMMON_TAG]))

    def build(items: list[NewsItem], chars: int) -> str:
        blocks = []
        for n, it in enumerate(items, 1):
            line = takeaway(it, chars) if chars else ""
            blocks.append(f"{n}. {it.title} ({it.source})\n" + (f"{line}\n" if line else "") + it.url)
        return head + lead + "\n\n".join(blocks) + tail

    items = list(picked)
    while items:
        for chars in (220, 140, 80, 0):
            text = build(items, chars)
            if len(text) <= limit:
                return text
        items = items[:-1]  # still too long: drop the last pick
    return (head + tail)[:limit]


def fold_preview(text: str, chars: int = FOLD_CHARS) -> str:
    """What readers see before LinkedIn's "see more"."""
    text = text.strip()
    return text if len(text) <= chars else text[:chars].rsplit(" ", 1)[0] + "…"


URL_RE = re.compile(r"https?://\S+")


def keep_links(draft: str, polished: str) -> tuple[str, str]:
    """Use the AI rewrite only if it kept every link; otherwise keep the draft and say why."""
    polished = (polished or "").strip()
    if not polished:
        return draft, "The rewrite came back empty, so the draft was kept."
    missing = [u for u in URL_RE.findall(draft) if u not in polished]
    if missing:
        return draft, "The rewrite dropped some links, so the draft was kept. Edit the text by hand instead."
    return polished, ""
