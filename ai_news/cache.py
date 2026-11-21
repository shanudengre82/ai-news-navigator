"""Thread-safe, PII-safe LRU query cache for single-turn RAG answers.

Avoids re-running LLM inference (and ZeroGPU allocation) for repeat questions and the preset
landing-page example queries. Never caches requests that contain PII, are blocked by guardrails,
or depend on multi-turn conversation history.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any


def normalize_query(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


class QueryCache:
    def __init__(self, maxsize: int = 64, ttl_seconds: float = 7 * 86400.0):
        self.maxsize = max(0, int(maxsize))
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._lock = threading.Lock()
        self._data: OrderedDict[tuple, tuple[float, dict]] = OrderedDict()

    @staticmethod
    def make_key(question: str, filters: dict | None = None, corpus_tag: Any = 0) -> tuple:
        f = filters or {}
        sources = tuple(sorted(f.get("sources") or []))
        return (
            normalize_query(question),
            sources,
            f.get("source_type") or "any",
            f.get("days"),
            corpus_tag,
        )

    def get(self, key: tuple) -> dict | None:
        if self.maxsize <= 0:
            return None
        now = time.time()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            ts, payload = entry
            if self.ttl_seconds > 0 and (now - ts) >= self.ttl_seconds:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return deepcopy(payload)

    def put(self, key: tuple, payload: dict) -> None:
        if self.maxsize <= 0:
            return
        now = time.time()
        with self._lock:
            self._data[key] = (now, deepcopy(payload))
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
