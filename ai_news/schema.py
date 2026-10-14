"""The normalized record every fetcher produces."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

_ARXIV_VERSION = re.compile(r"v\d+$")


def canonical_url(url: str) -> str:
    """Normalize a URL so the same item from two sources dedupes to one id."""
    url = url.strip().split("#")[0]
    url = re.sub(r"^http://", "https://", url)
    # arXiv: abs/pdf/huggingface paper pages and version suffixes all map to one abs URL.
    m = re.search(r"(?:arxiv\.org/(?:abs|pdf)|huggingface\.co/papers)/(\d{4}\.\d{4,5})", url)
    if m:
        return f"https://arxiv.org/abs/{m.group(1)}"
    url = re.sub(r"[?&]utm_[^&]+", "", url)
    return url.rstrip("/")


def make_id(url: str) -> str:
    return hashlib.sha1(canonical_url(url).encode()).hexdigest()[:16]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    source_type: str  # "paper" | "blog"
    published: str  # ISO-8601 UTC
    summary: str = ""
    authors: list[str] = field(default_factory=list)
    pdf_url: str = ""
    categories: list[str] = field(default_factory=list)
    upvotes: int = 0
    fetched_at: str = field(default_factory=utcnow_iso)
    id: str = ""

    def __post_init__(self) -> None:
        self.url = canonical_url(self.url) if "arxiv.org" in self.url else self.url.strip()
        if not self.id:
            self.id = make_id(self.url)
        self.title = " ".join(self.title.split())
        self.summary = " ".join(self.summary.split())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NewsItem":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        for k in ("authors", "categories"):
            v = known.get(k)
            known[k] = list(v) if v is not None else []
        known["upvotes"] = int(known.get("upvotes") or 0)
        return cls(**known)

    def page_text(self) -> str:
        """Text that is embedded / BM25-indexed."""
        return f"{self.title}\n{self.summary}"
