"""Hugging Face Daily Papers (community-curated, upvoted). Free JSON API."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from ai_news.fetchers.common import http_get
from ai_news.schema import NewsItem

log = logging.getLogger(__name__)
API = "https://huggingface.co/api/daily_papers"


def parse(records: list[dict[str, Any]]) -> list[NewsItem]:
    items = []
    for rec in records:
        p = rec.get("paper", rec)
        pid = p.get("id")
        if not pid:
            continue
        items.append(NewsItem(
            title=p.get("title") or rec.get("title", ""),
            url=f"https://arxiv.org/abs/{pid}",
            source="HF Daily Papers",
            source_type="paper",
            published=(p.get("publishedAt") or rec.get("publishedAt") or "").replace(".000Z", "+00:00"),
            summary=p.get("summary") or rec.get("summary", ""),
            authors=[a.get("name", "") for a in p.get("authors", []) if not a.get("hidden")],
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            categories=["hf-daily"],
            upvotes=int(p.get("upvotes") or 0),
        ))
    return [i for i in items if i.published]


def fetch_recent(days: int = 14) -> list[NewsItem]:
    out: list[NewsItem] = []
    today = date.today()
    for d in range(days):
        day = (today - timedelta(days=d)).isoformat()
        try:
            out.extend(parse(http_get(API, params={"date": day, "limit": 100}).json()))
        except Exception as e:  # one bad day shouldn't drop the rest
            log.warning("HF daily papers %s failed: %s", day, e)
    log.info("HF Daily Papers: %d papers", len(out))
    return out
