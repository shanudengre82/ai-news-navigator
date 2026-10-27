"""The research map and the landing-page stats.

The map follows Karpathy's "LLM wiki" pattern, where notes link to the concept pages they belong to and to
each other, but it is built from the collection itself: k-means over the stored embeddings finds the topics
(the concept pages), a few title terms name them, and each paper links to its topic and to its nearest
neighbours by meaning. Everything runs on CPU from the FAISS vectors the store already has.
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from datetime import date, datetime, timedelta, timezone

import faiss
import numpy as np

from ai_news.fetchers.common import parse_iso
from ai_news.schema import NewsItem
from ai_news.store import CorpusStore
from ai_news.text import tokenize

MAX_NODES = 3000
NEIGHBOURS = 4
# Two papers are "near" when they are closer than this share of random pairs. Embedding models differ a lot in
# their cosine range (bge-small scores unrelated abstracts ~0.7), so the cut is measured on the corpus itself.
NEAR_PERCENTILE = 95
SPREAD = 300.0  # the PCA start fills roughly ±SPREAD in each axis

_cache: dict[int, tuple[int, dict]] = {}
_cache_lock = threading.Lock()


def _day(it: NewsItem) -> date | None:
    try:
        return parse_iso(it.published).astimezone(timezone.utc).date()
    except Exception:
        return None


def _pick(items: list[NewsItem], max_nodes: int) -> list[NewsItem]:
    """Papers only; past the cap keep the upvoted ones (Daily Papers picks), then the newest."""
    papers = [it for it in items if it.source_type == "paper"]
    papers.sort(key=lambda it: (it.upvotes > 0, it.published), reverse=True)
    return papers[:max_nodes]


def _n_topics(n: int) -> int:
    if n < 2:
        return n
    return max(1, min(36, max(4, round(math.sqrt(n / 2))), n // 2))


def _cluster(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.zeros(len(x), dtype=int)
    km = faiss.Kmeans(x.shape[1], k, niter=20, seed=0, spherical=True, min_points_per_centroid=1)
    km.train(x)
    return km.index.search(x, 1)[1][:, 0].astype(int)


def _terms(title: str) -> set[str]:
    toks = [t for t in tokenize(title) if not t.replace(".", "").isdigit() and len(t) > 1]
    return set(toks) | {f"{a} {b}" for a, b in zip(toks, toks[1:])}


def _labels(titles: list[str], groups: np.ndarray, k: int, per_topic: int = 3) -> list[str]:
    """Name each topic by the title terms most frequent in it and rare elsewhere (tf-idf over titles)."""
    docs = [_terms(t) for t in titles]
    df = Counter(term for d in docs for term in d)
    n = len(docs)
    labels = []
    for g in range(k):
        members = [docs[i] for i in np.flatnonzero(groups == g)]
        tf = Counter(term for d in members for term in d)
        floor = 2 if len(members) >= 3 else 1
        ranked = sorted((c / len(members) * math.log(n / df[t]) * (1.3 if " " in t else 1.0), t)
                        for t, c in tf.items() if c >= floor and df[t] < n)
        chosen: list[str] = []
        for _, term in reversed(ranked):
            if any(set(term.split()) & set(c.split()) or term in c or c in term for c in chosen):
                continue  # "diffusion" next to "video diffusion" says nothing new
            chosen.append(term)
            if len(chosen) == per_topic:
                break
        labels.append(" · ".join(chosen) if chosen else f"Topic {g + 1}")
    return labels


def _pca3(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0)
    if len(x) < 2:
        return np.zeros((len(x), 3))
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pos = centered @ vt[:3].T
    if pos.shape[1] < 3:
        pos = np.pad(pos, ((0, 0), (0, 3 - pos.shape[1])))
    scale = np.abs(pos).max(axis=0)
    return pos / np.where(scale > 0, scale, 1) * SPREAD


def _near_cut(x: np.ndarray, sample: int = 400) -> float:
    """Cosine above which a pair is closer than NEAR_PERCENTILE % of random pairs."""
    rows = x[np.random.default_rng(0).choice(len(x), min(sample, len(x)), replace=False)]
    sims = rows @ rows.T
    return float(np.percentile(sims[~np.eye(len(rows), dtype=bool)], NEAR_PERCENTILE))


def _neighbour_links(x: np.ndarray, k: int, min_sim: float | None = None) -> list[tuple[int, int, float]]:
    if len(x) < 2:
        return []
    min_sim = _near_cut(x) if min_sim is None else min_sim
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    sims, nbrs = index.search(x, min(k + 1, len(x)))
    seen: dict[tuple[int, int], float] = {}
    for i in range(len(x)):
        for j, s in zip(nbrs[i], sims[i]):
            if j < 0 or j == i or s < min_sim:
                continue
            seen.setdefault((min(i, int(j)), max(i, int(j))), float(s))
    return [(a, b, s) for (a, b), s in seen.items()]


def build_map(items: list[NewsItem], vectors: np.ndarray, k: int = NEIGHBOURS,
              min_sim: float | None = None) -> dict:
    """The 3D graph, compact for the wire: `topics` are the hubs, `papers[i]` is paper i, and `links` holds
    [i, j, cosine] between near papers. Each paper also links to its topic `g`; the page draws that link
    itself. Positions are a PCA start for the force layout. `vectors` are row-aligned with `items`."""
    if not items:
        return {"topics": [], "papers": [], "links": []}
    x = np.array(vectors, dtype="float32")  # a copy: normalized in place below
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n_topics = _n_topics(len(items))
    groups = _cluster(x, n_topics)
    labels = _labels([it.title for it in items], groups, n_topics)
    pos = np.rint(_pca3(x)).astype(int)

    topics = []
    for g in range(n_topics):
        members = np.flatnonzero(groups == g)
        centre = pos[members].mean(axis=0) if len(members) else np.zeros(3)
        topics.append({"label": labels[g], "size": int(len(members)), "pos": [int(c) for c in np.rint(centre)]})
    papers = []
    for i, it in enumerate(items):
        title = it.title if len(it.title) <= 140 else it.title[:139].rsplit(" ", 1)[0] + "…"
        d = _day(it)
        papers.append({"t": title, "u": it.url, "s": it.source, "d": d.isoformat() if d else "",
                       "v": it.upvotes, "g": int(groups[i]), "pos": [int(c) for c in pos[i]]})
    links = [[a, b, round(s, 3)] for a, b, s in _neighbour_links(x, k, min_sim)]
    return {"topics": topics, "papers": papers, "links": links}


def research_map(store: CorpusStore, max_nodes: int = MAX_NODES) -> dict:
    """The map for the store's current items, rebuilt only when the store changes."""
    key = id(store)
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] == store.version:
            return hit[1]
    version = store.version
    with store._lock:
        items = _pick(store.all_items(), max_nodes)
        vectors = store.vectors.get([it.id for it in items]) if len(store.vectors) else None
    data = build_map(items, vectors) if vectors is not None and len(vectors) else build_map([], None)
    data["version"] = version
    with _cache_lock:
        _cache[key] = (version, data)
    return data


def collection_stats(store: CorpusStore, days: int = 14, today: date | None = None) -> dict:
    """Totals, papers this week against last week, and papers / posts per publication day (UTC)."""
    today = today or datetime.now(timezone.utc).date()
    items = store.all_items()
    first = today - timedelta(days=days - 1)
    per_day = {first + timedelta(days=i): [0, 0] for i in range(days)}
    week = prev = 0
    for it in items:
        d = _day(it)
        if d is None:
            continue
        if d in per_day:
            per_day[d][0 if it.source_type == "paper" else 1] += 1
        if it.source_type != "paper":
            continue
        age = (today - d).days
        if 0 <= age < 7:
            week += 1
        elif 7 <= age < 14:
            prev += 1
    papers = sum(it.source_type == "paper" for it in items)
    return {"total": len(items), "papers": papers, "posts": len(items) - papers,
            "sources": len({it.source for it in items}), "week": week, "prev_week": prev,
            "last_refresh": store.meta.get("last_refresh"),
            "days": [(d, p, b) for d, (p, b) in sorted(per_day.items())]}
