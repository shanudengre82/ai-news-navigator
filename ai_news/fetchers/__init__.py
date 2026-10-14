"""Fetch every enabled source in sources.yaml. A failing source is logged and skipped."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from ai_news.fetchers import arxiv, hf_papers, rss
from ai_news.schema import NewsItem

log = logging.getLogger(__name__)


def load_sources(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def fetch_all(sources: dict, lookback_days: int) -> tuple[list[NewsItem], dict[str, str]]:
    """Returns (items, status) where status maps source name -> "ok (n)" or the error."""
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    jobs = {}
    if sources.get("arxiv", {}).get("enabled"):
        cfg = sources["arxiv"]
        jobs["arXiv"] = lambda: arxiv.fetch_recent(cfg["categories"], since, cfg.get("max_results", 800),
                                                   cfg.get("page_size", 100))
        jobs["arXiv RSS"] = lambda: arxiv.fetch_rss(cfg["categories"])
    if sources.get("hf_daily_papers", {}).get("enabled"):
        days = min(sources["hf_daily_papers"].get("days", 14), lookback_days)
        jobs["HF Daily Papers"] = lambda: hf_papers.fetch_recent(days)
    for feed in sources.get("rss", []):
        jobs[feed["name"]] = (lambda f=feed: rss.fetch(f["name"], f["url"], since))

    items: list[NewsItem] = []
    status: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {name: pool.submit(fn) for name, fn in jobs.items()}
        for name, fut in futures.items():
            try:
                got = fut.result()
                items.extend(got)
                status[name] = f"ok ({len(got)})"
            except Exception as e:
                log.warning("source %s failed: %s", name, e)
                status[name] = f"error: {type(e).__name__}: {e}"[:200]
    return items, status
