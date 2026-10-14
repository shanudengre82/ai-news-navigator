"""Generic RSS/Atom fetcher for AI lab and research blogs."""

from __future__ import annotations

import logging
from datetime import datetime

import feedparser

from ai_news.fetchers.common import http_get, parse_iso, strip_html, struct_to_iso
from ai_news.schema import NewsItem

log = logging.getLogger(__name__)


def parse_feed(xml: str | bytes, source: str) -> list[NewsItem]:
    feed = feedparser.parse(xml)
    items = []
    for e in feed.entries:
        link = e.get("link")
        published = struct_to_iso(e.get("published_parsed")) or struct_to_iso(e.get("updated_parsed"))
        if not link or not published:
            continue
        summary = e.get("summary") or ""
        if not summary and e.get("content"):
            summary = e.content[0].get("value", "")
        items.append(NewsItem(
            title=strip_html(e.get("title", ""), 300),
            url=link,
            source=source,
            source_type="blog",
            published=published,
            summary=strip_html(summary),
            authors=[a.get("name", "") for a in e.get("authors", []) if a.get("name")],
            categories=[t.get("term") for t in e.get("tags", []) if t.get("term")][:8],
        ))
    return items


def fetch(name: str, url: str, since: datetime | None = None) -> list[NewsItem]:
    items = parse_feed(http_get(url, timeout=30).content, name)
    if since:
        items = [i for i in items if parse_iso(i.published) >= since]
    log.info("%s: %d posts", name, len(items))
    return items
