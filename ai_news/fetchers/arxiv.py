"""arXiv API fetcher (https://info.arxiv.org/help/api). Free; asks for a 3 s delay between calls."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime

import feedparser

from ai_news.fetchers.common import http_get, parse_iso, struct_to_iso
from ai_news.schema import NewsItem

log = logging.getLogger(__name__)
API = "https://export.arxiv.org/api/query"
RSS = "https://rss.arxiv.org/rss/"
_RSS_PREFIX = re.compile(r"^arXiv:\S+\s+Announce Type:\s*\S+\s+Abstract:\s*", re.IGNORECASE)


def parse_feed(xml: str | bytes) -> list[NewsItem]:
    feed = feedparser.parse(xml)
    items = []
    for e in feed.entries:
        pdf = next((l.href for l in e.get("links", []) if l.get("type") == "application/pdf"), "")
        published = struct_to_iso(e.get("published_parsed")) or struct_to_iso(e.get("updated_parsed"))
        if not published:
            continue
        items.append(NewsItem(
            title=e.get("title", ""),
            url=e.get("id") or e.get("link", ""),
            source="arXiv",
            source_type="paper",
            published=published,
            summary=e.get("summary", ""),
            authors=[a.get("name", "") for a in e.get("authors", [])],
            pdf_url=pdf,
            categories=[t.get("term") for t in e.get("tags", []) if t.get("term")],
        ))
    return items


def fetch_recent(categories: list[str], since: datetime, max_results: int = 800,
                 page_size: int = 100, delay: float = 3.0) -> list[NewsItem]:
    """Newest papers per category (one small query each — big OR-queries get throttled with HTTP 429),
    stopping once we page past `since`. Total is capped at `max_results`."""
    per_cat = max(page_size, max_results // max(len(categories), 1))
    out: dict[str, NewsItem] = {}
    errors = []
    for cat in categories:
        try:
            for start in range(0, per_cat, page_size):
                time.sleep(delay)  # arXiv asks for >= 3 s between requests
                r = http_get(API, params={"search_query": f"cat:{cat}", "sortBy": "submittedDate",
                                          "sortOrder": "descending", "start": start,
                                          "max_results": min(page_size, per_cat - start)},
                             timeout=120, retries=3, backoff=10)
                page = parse_feed(r.content)
                out.update((i.id, i) for i in page if parse_iso(i.published) >= since)
                if not page or parse_iso(page[-1].published) < since:
                    break
        except Exception as e:  # keep what other categories returned
            log.warning("arXiv %s failed: %s", cat, e)
            errors.append(cat)
    if errors and not out:
        raise RuntimeError(f"arXiv API failed for {errors}")
    log.info("arXiv: %d papers", len(out))
    return list(out.values())


def search(query: str, max_results: int = 15) -> list[NewsItem]:
    """Live relevance search over all of arXiv (used by the agent as a fallback)."""
    words = [re.sub(r"[^\w\-]", "", w) for w in query.split()]
    terms = " AND ".join(f"all:{w}" for w in words if len(w) > 2) or f"all:{query}"
    r = http_get(API, params={"search_query": terms, "sortBy": "relevance", "max_results": max_results * 2},
                 timeout=45, retries=1, backoff=5)
    # Filter to CS / stat.ML after the fact (wildcard category clauses get throttled).
    papers = [p for p in parse_feed(r.content)
              if any(c.startswith("cs.") or c == "stat.ML" for c in p.categories)]
    return papers[:max_results]


def parse_rss(xml: str | bytes) -> list[NewsItem]:
    """rss.arxiv.org daily announcement feed (cheap; complements the slower API)."""
    feed = feedparser.parse(xml)
    items = []
    for e in feed.entries:
        published = struct_to_iso(e.get("published_parsed")) or struct_to_iso(e.get("updated_parsed"))
        if not e.get("link") or not published:
            continue
        if "replace" in (e.get("arxiv_announce_type") or ""):
            continue  # revised versions of older papers
        authors = [a.strip() for a in (e.get("author") or "").split(",") if a.strip()]
        items.append(NewsItem(
            title=e.get("title", ""), url=e["link"], source="arXiv", source_type="paper", published=published,
            summary=_RSS_PREFIX.sub("", e.get("summary", "")), authors=authors,
            pdf_url=e["link"].replace("/abs/", "/pdf/"),
            categories=[t.get("term") for t in e.get("tags", []) if t.get("term")],
        ))
    return items


def fetch_rss(categories: list[str]) -> list[NewsItem]:
    items = parse_rss(http_get(RSS + "+".join(categories), timeout=60).content)
    log.info("arXiv RSS: %d papers", len(items))
    return items
