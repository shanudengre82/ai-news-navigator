"""Hugging Face Daily Papers & historical Hugging Face document fetchers. Free JSON/RSS APIs."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Any

from ai_news.fetchers import rss
from ai_news.fetchers.common import http_get, parse_iso
from ai_news.schema import NewsItem

log = logging.getLogger(__name__)
API = "https://huggingface.co/api/daily_papers"
HF_BLOG_FEED = "https://huggingface.co/blog/feed.xml"
HF_DAILY_PAPERS_LAUNCH = date(2023, 5, 1)  # Hugging Face Daily Papers launched in May 2023
DEFAULT_AI_CATS = frozenset({"cs.AI", "cs.CL", "cs.LG", "cs.CV", "cs.MA", "cs.IR", "stat.ML"})


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


def _fetch_day(day: date) -> list[NewsItem]:
    iso = day.isoformat()
    try:
        resp = http_get(API, params={"date": iso, "limit": 100})
        data = resp.json()
        return parse(data) if isinstance(data, list) else []
    except Exception as e:  # one bad day shouldn't drop the rest
        log.debug("HF daily papers %s failed: %s", iso, e)
        return []


def fetch_recent(days: int = 14) -> list[NewsItem]:
    out: list[NewsItem] = []
    today = date.today()
    for d in range(days):
        out.extend(_fetch_day(today - timedelta(days=d)))
    log.info("HF Daily Papers: %d papers", len(out))
    return out


def fetch_all_hf_papers(
    since_year: int = 2010,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    max_items: int | None = None,
    workers: int = 6,
) -> list[NewsItem]:
    """Fetch all curated Hugging Face Daily Papers from launch to today, keeping papers since `since_year`.

    Deduplicates by canonical item ID (keeping the highest upvote count and longest summary).
    """
    end = end_date or date.today()
    floor = start_date or max(HF_DAILY_PAPERS_LAUNCH, date(since_year, 1, 1))
    if floor > end:
        return []

    days = [end - timedelta(days=i) for i in range((end - floor).days + 1)]
    seen: dict[str, NewsItem] = {}
    min_dt = datetime(since_year, 1, 1, tzinfo=timezone.utc)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_fetch_day, d): d for d in days}
        done = 0
        for fut in as_completed(futures):
            done += 1
            for it in fut.result():
                try:
                    if parse_iso(it.published) < min_dt:
                        continue
                except Exception:
                    continue
                old = seen.get(it.id)
                if old is None:
                    seen[it.id] = it
                else:
                    old.upvotes = max(old.upvotes, it.upvotes)
                    if len(it.summary) > len(old.summary):
                        old.summary = it.summary
            if done % 150 == 0 or done == len(days):
                log.info("HF Daily Papers backfill: %d/%d days scanned, %d unique papers", done, len(days), len(seen))
            if max_items and len(seen) >= max_items:
                break

    items = sorted(seen.values(), key=lambda x: x.published, reverse=True)
    if max_items:
        items = items[:max_items]
    log.info("HF Daily Papers backfill complete: %d unique papers (since %d)", len(items), since_year)
    return items


def fetch_hf_blog_archive(since_year: int = 2010) -> list[NewsItem]:
    """Fetch all available Hugging Face Blog posts since `since_year`."""
    since = datetime(since_year, 1, 1, tzinfo=timezone.utc)
    try:
        items = rss.fetch("Hugging Face Blog", HF_BLOG_FEED, since)
        log.info("HF Blog archive: %d posts (since %d)", len(items), since_year)
        return items
    except Exception as e:
        log.warning("HF Blog archive failed: %s", e)
        return []


def fetch_hf_arxiv_dataset(
    since_year: int = 2010,
    *,
    categories: list[str] | None = None,
    max_items: int | None = None,
    dataset_id: str = "librarian-bots/arxiv-metadata-snapshot",
) -> list[NewsItem]:
    """Optional streaming loader for an arXiv metadata dataset hosted on Hugging Face Hub (since `since_year`)."""
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    wanted = set(categories) if categories else set(DEFAULT_AI_CATS)
    api = HfApi()
    files = [f for f in api.list_repo_files(dataset_id, repo_type="dataset") if f.endswith(".parquet")]
    if not files:
        log.warning("No parquet files found in HF dataset %s", dataset_id)
        return []

    out: list[NewsItem] = []
    for fname in sorted(files):
        local = hf_hub_download(dataset_id, fname, repo_type="dataset")
        df = pd.read_parquet(local)
        for rec in df.to_dict("records"):
            cats_raw = rec.get("categories") or ""
            cats = cats_raw.split() if isinstance(cats_raw, str) else list(cats_raw)
            if wanted and not (wanted & set(cats)):
                continue
            pub = str(rec.get("update_date") or rec.get("published") or "")
            if len(pub) < 4 or not pub[:4].isdigit() or int(pub[:4]) < since_year:
                continue
            pid = str(rec.get("id") or "").strip()
            title = " ".join(str(rec.get("title") or "").split())
            abstract = " ".join(str(rec.get("abstract") or rec.get("summary") or "").split())
            if not pid or not title:
                continue
            iso_pub = f"{pub[:10]}T00:00:00+00:00" if len(pub) == 10 else pub
            authors_raw = rec.get("authors") or ""
            authors = ([a.strip() for a in authors_raw.split(",") if a.strip()]
                       if isinstance(authors_raw, str) else [str(a) for a in authors_raw])
            out.append(NewsItem(
                title=title,
                url=f"https://arxiv.org/abs/{pid}",
                source="arXiv",
                source_type="paper",
                published=iso_pub,
                summary=abstract,
                authors=authors[:12],
                pdf_url=f"https://arxiv.org/pdf/{pid}",
                categories=cats[:6],
                upvotes=0,
            ))
            if max_items and len(out) >= max_items:
                log.info("HF arXiv dataset (%s): reached cap %d items", dataset_id, max_items)
                return out
    log.info("HF arXiv dataset (%s): loaded %d items since %d", dataset_id, len(out), since_year)
    return out
