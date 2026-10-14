"""Fetch all sources -> merge into the store -> save -> push to the HF Dataset repo."""

from __future__ import annotations

import logging
import threading
import time

from ai_news.fetchers import fetch_all, load_sources
from ai_news.store import CorpusStore

log = logging.getLogger(__name__)
_refresh_lock = threading.Lock()


def refresh(store: CorpusStore, sources_file, lookback_days: int, push: bool = True) -> str:
    """Run one refresh. Returns a human-readable summary. Concurrent calls are skipped."""
    if not _refresh_lock.acquire(blocking=False):
        return "A refresh is already running."
    try:
        t0 = time.time()
        items, status = fetch_all(load_sources(sources_file), lookback_days)
        added = store.add(items)
        pruned = store.prune()
        store.mark_refreshed(status)
        store.save()
        pushed = ""
        if push and store.dataset_repo:
            try:
                store.push()
                pushed = f", pushed to {store.dataset_repo}"
            except Exception as e:
                log.warning("push failed: %s", e)
                pushed = f", push failed: {e}"
        msg = (f"Fetched {len(items)} items, added {len(added)} new, pruned {pruned}; "
               f"corpus now {len(store.items)} items ({time.time() - t0:.0f}s{pushed}).")
        log.info(msg)
        return msg
    finally:
        _refresh_lock.release()
