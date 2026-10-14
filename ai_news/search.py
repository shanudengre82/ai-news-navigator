"""Plain keyword search (BM25) with filters — the non-LLM search mode."""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone

from ai_news.fetchers.common import parse_iso
from ai_news.schema import NewsItem
from ai_news.store import CorpusStore
from ai_news.text import tokenize

DATE_WINDOWS = {"Any time": None, "Last 24 hours": 1, "Last 3 days": 3, "Last week": 7,
                "Last month": 30, "Last 3 months": 90}


def passes_filters(it: NewsItem, sources: list[str] | None, source_type: str | None,
                   days: int | None, now: datetime | None = None) -> bool:
    if sources and it.source not in sources:
        return False
    if source_type and source_type not in ("all", "any") and it.source_type != source_type:
        return False
    if days:
        now = now or datetime.now(timezone.utc)
        if parse_iso(it.published) < now - timedelta(days=days):
            return False
    return True


def keyword_search(store: CorpusStore, query: str, sources: list[str] | None = None,
                   source_type: str | None = None, days: int | None = None,
                   sort: str = "relevance", limit: int = 50) -> list[tuple[NewsItem, float]]:
    """BM25 over title+summary. Quoted phrases ("like this") must appear verbatim."""
    phrases = [p.lower() for p in re.findall(r'"([^"]+)"', query)]
    if query.strip():
        hits = store.bm25_scores(query.replace('"', " "))
    else:
        hits = [(it, 0.0) for it in store.all_items()]
        sort = "date" if sort == "relevance" else sort
    hits = [(it, s) for it, s in hits if passes_filters(it, sources, source_type, days)
            and all(p in it.page_text().lower() for p in phrases)]
    if sort == "date":
        hits.sort(key=lambda h: h[0].published, reverse=True)
    elif sort == "upvotes":
        hits.sort(key=lambda h: (h[0].upvotes, h[1]), reverse=True)
    return hits[:limit]


def highlight(text: str, query: str) -> str:
    """HTML-escape text and wrap query terms in <mark>."""
    text = html.escape(text)
    terms = sorted({t for t in tokenize(query.replace('"', " ")) if len(t) > 2}, key=len, reverse=True)
    if not terms:
        return text
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", text)
