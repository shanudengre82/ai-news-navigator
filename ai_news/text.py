"""Tokenization shared by BM25 and highlighting."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9][a-z0-9\-\+\.]*[a-z0-9]|[a-z0-9]")
STOPWORDS = frozenset("""a an and are as at be by for from has have in into is it its of on or our that the
their this to was we were what which with about how new using via based towards toward can do does i me my
show find latest recent papers paper blog posts post""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]
