"""Fetch all sources -> merge into the store -> save -> push to the HF Dataset repo."""

from __future__ import annotations

import logging
import threading
import time

from ai_news.fetchers import fetch_all, load_sources
from ai_news.store import META_FILE, CorpusStore

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


def sync_from_hub(store: CorpusStore) -> str:
    """Pull mode: download the latest snapshot from the dataset repo and reload it if it changed."""
    if not store.dataset_repo:
        return "No DATASET_REPO configured; nothing to sync."
    if not _refresh_lock.acquire(blocking=False):
        return "A sync is already running."
    try:
        meta = store.data_dir / META_FILE
        before = meta.read_text() if meta.exists() else ""
        if not store.pull():
            return f"Could not download {store.dataset_repo}; keeping the current corpus."
        if meta.exists() and meta.read_text() != before:
            store.load()
            msg = f"Pulled new snapshot from {store.dataset_repo}: {len(store.items)} items."
        else:
            msg = f"Already up to date with {store.dataset_repo} ({len(store.items)} items)."
        msg += f" Last refresh: {store.meta.get('last_refresh') or 'unknown'} (UTC)."
        log.info(msg)
        return msg
    finally:
        _refresh_lock.release()
