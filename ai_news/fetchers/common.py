"""Shared HTTP + parsing helpers for fetchers."""

from __future__ import annotations

import html
import re
import time
from datetime import datetime, timezone

import httpx

USER_AGENT = "ai-news-navigator/1.0 (+https://huggingface.co/spaces; open-source research reader)"
_TAG = re.compile(r"<[^>]+>")


def http_get(url: str, *, params: dict | None = None, timeout: float = 30.0, retries: int = 2,
             backoff: float = 3.0) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = httpx.get(url, params=params, timeout=timeout, follow_redirects=True,
                          headers={"User-Agent": USER_AGENT})
            if r.status_code in (429, 503) and attempt < retries:
                retry_after = r.headers.get("Retry-After", "")
                time.sleep(float(retry_after) if retry_after.isdigit() else backoff * 2 ** attempt)
                continue
            r.raise_for_status()
            return r
        except httpx.HTTPError as e:
            last = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise last  # type: ignore[misc]


def strip_html(text: str, limit: int = 2000) -> str:
    text = html.unescape(_TAG.sub(" ", text or ""))
    text = " ".join(text.split())
    return text[:limit]


def struct_to_iso(t) -> str | None:
    """feedparser's time.struct_time (UTC) -> ISO string."""
    if not t:
        return None
    return datetime(*t[:6], tzinfo=timezone.utc).isoformat()


def parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
