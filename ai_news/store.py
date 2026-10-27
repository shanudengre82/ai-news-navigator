"""Corpus store: Parquet records + FAISS vectors + BM25, synced to a free HF Dataset repo."""

from __future__ import annotations

import json
import logging
import os
import threading
import weakref
from datetime import datetime, timedelta, timezone
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from ai_news.fetchers.common import parse_iso
from ai_news.schema import NewsItem, utcnow_iso
from ai_news.text import tokenize

log = logging.getLogger(__name__)

ITEMS_FILE = "items.parquet"
FAISS_FILE = "index.faiss"
META_FILE = "meta.json"


def faiss_id(item_id: str) -> int:
    """Item ids are 16 hex chars; the first 15 fit in a positive int64."""
    return int(item_id[:15], 16)


class VectorIndex:
    """Cosine-similarity FAISS index (inner product on normalized vectors) keyed by item id."""

    def __init__(self, index: faiss.Index | None = None):
        self.index = index

    @property
    def ids(self) -> set[int]:
        return set(faiss.vector_to_array(self.index.id_map).tolist()) if self.index is not None else set()

    def add(self, item_ids: list[str], vectors: list[list[float]]) -> None:
        x = np.asarray(vectors, dtype="float32")
        if self.index is None:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(x.shape[1]))
        self.index.add_with_ids(x, np.array([faiss_id(i) for i in item_ids], dtype="int64"))

    def get(self, item_ids: list[str]) -> np.ndarray:
        """The stored vectors for these items, in the same order (no re-embedding)."""
        if self.index is None or not item_ids:
            return np.zeros((0, self.index.d if self.index is not None else 0), dtype="float32")
        return np.vstack([self.index.reconstruct(faiss_id(i)) for i in item_ids]).astype("float32")

    def remove(self, item_ids: list[str]) -> None:
        if self.index is not None and item_ids:
            self.index.remove_ids(np.array([faiss_id(i) for i in item_ids], dtype="int64"))

    def search(self, vector: list[float], k: int) -> list[tuple[int, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        scores, ids = self.index.search(np.asarray([vector], dtype="float32"), min(k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def __len__(self) -> int:
        return self.index.ntotal if self.index is not None else 0


_STORES: "weakref.WeakSet[CorpusStore]" = weakref.WeakSet()


def _after_fork_in_child() -> None:
    """ZeroGPU forks a GPU worker from the main process. If a background refresh held a store lock at
    that moment, the child would inherit it locked forever — give each store a fresh lock instead."""
    for s in list(_STORES):
        s._lock = threading.RLock()
    faiss.omp_set_num_threads(1)  # OpenMP thread pools don't survive fork


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


class CorpusStore:
    def __init__(self, data_dir: Path, embeddings: Embeddings, dataset_repo: str | None = None,
                 hf_token: str | None = None, retention_days: int = 180):
        self.data_dir = Path(data_dir)
        self.embeddings = embeddings
        self.dataset_repo = dataset_repo
        self.hf_token = hf_token
        self.retention_days = retention_days
        self.items: dict[str, NewsItem] = {}
        self.vectors = VectorIndex()
        self._by_faiss_id: dict[int, str] = {}
        self.meta: dict = {}
        self._lock = threading.RLock()
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._snapshot_mtime = 0.0  # mtime of the on-disk snapshot this process last loaded or wrote
        self.version = 0  # bumped whenever the items change; keys caches such as the research map
        _STORES.add(self)

    # ---------- persistence ----------
    def pull(self) -> bool:
        """Download the latest corpus snapshot from the dataset repo, if configured."""
        if not self.dataset_repo:
            return False
        from huggingface_hub import snapshot_download
        try:
            snapshot_download(self.dataset_repo, repo_type="dataset", local_dir=self.data_dir,
                              token=self.hf_token, allow_patterns=[ITEMS_FILE, FAISS_FILE, META_FILE])
            return True
        except Exception as e:
            log.warning("could not pull %s (%s); starting empty", self.dataset_repo, e)
            return False

    def _disk_mtime(self) -> float:
        p = self.data_dir / META_FILE
        return p.stat().st_mtime if p.exists() else 0.0

    def reload_if_stale(self) -> bool:
        """ZeroGPU runs GPU functions in a forked worker that may outlive a refresh done by the main
        process; reload the snapshot from disk if it is newer than what this process has."""
        if self._disk_mtime() > self._snapshot_mtime:
            self.load()
            return True
        return False

    def load(self) -> None:
        self._snapshot_mtime = self._disk_mtime()
        path = self.data_dir / ITEMS_FILE
        items: dict[str, NewsItem] = {}
        if path.exists():
            for rec in pd.read_parquet(path).to_dict("records"):
                it = NewsItem.from_dict(rec)
                items[it.id] = it
        meta_path = self.data_dir / META_FILE
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        vectors = VectorIndex()
        if (self.data_dir / FAISS_FILE).exists():
            vectors = VectorIndex(faiss.read_index(str(self.data_dir / FAISS_FILE)))
            if vectors.ids != {faiss_id(i) for i in items}:
                log.warning("FAISS index out of sync with items; rebuilding")
                vectors = VectorIndex()
        with self._lock:
            self.items, self.meta, self.vectors = items, meta, vectors
            self._by_faiss_id = {faiss_id(i): i for i in items}
            if not len(self.vectors) and self.items:
                self._rebuild_vectors()
            self._rebuild_bm25()
            self.version += 1
        log.info("corpus loaded: %d items", len(self.items))

    def save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            pd.DataFrame([i.to_dict() for i in self.items.values()]).to_parquet(self.data_dir / ITEMS_FILE, index=False)
            if len(self.vectors):
                faiss.write_index(self.vectors.index, str(self.data_dir / FAISS_FILE))
            (self.data_dir / META_FILE).write_text(json.dumps(self.meta, indent=2))
            self._snapshot_mtime = self._disk_mtime()

    def push(self) -> None:
        if not self.dataset_repo:
            return
        from huggingface_hub import HfApi
        api = HfApi(token=self.hf_token)
        api.create_repo(self.dataset_repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(repo_id=self.dataset_repo, repo_type="dataset", folder_path=str(self.data_dir),
                          allow_patterns=[ITEMS_FILE, FAISS_FILE, META_FILE],
                          commit_message=f"Refresh corpus ({len(self.items)} items)")
        log.info("pushed corpus to %s", self.dataset_repo)

    # ---------- mutation ----------
    def add(self, new_items: list[NewsItem]) -> list[NewsItem]:
        """Merge items; embed only those not seen before. Returns the newly added ones."""
        fresh: dict[str, NewsItem] = {}
        with self._lock:
            for it in new_items:
                old = self.items.get(it.id) or fresh.get(it.id)
                if old is None:
                    fresh[it.id] = it
                    continue
                # Same paper from arXiv + HF Daily Papers: keep the popularity signal and both labels.
                old.upvotes = max(old.upvotes, it.upvotes)
                old.categories = list(dict.fromkeys([*old.categories, *it.categories]))
                if len(it.summary) > len(old.summary):
                    old.summary = it.summary
        if not fresh:
            return []
        vectors = self.embeddings.embed_documents([i.page_text() for i in fresh.values()])  # slow; outside the lock
        with self._lock:
            self.items.update(fresh)
            self._by_faiss_id.update({faiss_id(i): i for i in fresh})
            self.vectors.add(list(fresh), vectors)
            self._rebuild_bm25()
            self.version += 1
        return list(fresh.values())

    def prune(self) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        with self._lock:
            old = [i for i, it in self.items.items() if parse_iso(it.published) < cutoff]
            if old:
                for i in old:
                    del self.items[i]
                    self._by_faiss_id.pop(faiss_id(i), None)
                self.vectors.remove(old)
                self._rebuild_bm25()
                self.version += 1
        return len(old)

    def _rebuild_vectors(self) -> None:
        ids = list(self.items)
        self.vectors = VectorIndex()
        self.vectors.add(ids, self.embeddings.embed_documents([self.items[i].page_text() for i in ids]))

    def _rebuild_bm25(self) -> None:
        ids = list(self.items)
        self._bm25_ids = ids
        self._bm25 = BM25Okapi([tokenize(self.items[i].page_text()) for i in ids]) if ids else None

    def mark_refreshed(self, status: dict[str, str]) -> None:
        self.meta["last_refresh"] = utcnow_iso()
        self.meta["source_status"] = status

    # ---------- queries ----------
    def bm25_scores(self, query: str) -> list[tuple[NewsItem, float]]:
        with self._lock:
            if self._bm25 is None:
                return []
            tokens = tokenize(query)
            if not tokens:
                return []
            scores = self._bm25.get_scores(tokens)
            ids = self._bm25_ids
        order = np.argsort(-scores)
        return [(self.items[ids[k]], float(scores[k])) for k in order if scores[k] > 0]

    def vector_search(self, query: str, k: int = 40) -> list[tuple[NewsItem, float]]:
        """Semantic search; scores are cosine similarities (higher is better)."""
        if not len(self.vectors):
            return []
        qv = self.embeddings.embed_query(query)
        with self._lock:
            hits = self.vectors.search(qv, k)
            return [(self.items[self._by_faiss_id[h]], s) for h, s in hits if h in self._by_faiss_id]

    def all_items(self) -> list[NewsItem]:
        with self._lock:
            return list(self.items.values())

    def stats(self) -> dict:
        items = self.all_items()
        by_source: dict[str, int] = {}
        for it in items:
            by_source[it.source] = by_source.get(it.source, 0) + 1
        return {"total": len(items), "by_source": dict(sorted(by_source.items(), key=lambda kv: -kv[1])),
                "last_refresh": self.meta.get("last_refresh"), "source_status": self.meta.get("source_status", {})}
