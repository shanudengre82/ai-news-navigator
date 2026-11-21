"""Multi-stage Sentence-Transformers re-reranking strategy.

Pipeline:
1. Stage 1 (Coarse Cross-Encoder): Fast scoring of the full RRF candidate pool (e.g. up to 300 items)
   using a compact cross-encoder (`BAAI/bge-reranker-base`, max_length=512).
2. Stage 2 (Fine Re-Reranker): High-precision re-scoring of only the top Stage-1 shortlist
   (`stage1_top_k`, default 40) using a stronger cross-encoder (`BAAI/bge-reranker-v2-m3`, max_length=1024),
   incorporating the LLM-extracted query intent when it adds context, and fusing Stage-1 + Stage-2 scores.
   Cleanly skipped when `stage2_model_id` is None/empty.
3. Stage 3 (MMR Diversity): Maximal Marginal Relevance over the top boosted matches using the
   already-stored normalized document embeddings (zero extra model cost), preventing near-duplicate
   stories from crowding out the top LLM-judged slots.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

log = logging.getLogger(__name__)


def _default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class PassageScorer(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderScorer:
    """Thin wrapper over `sentence_transformers.CrossEncoder` returning relevance probabilities in [0, 1]."""

    def __init__(self, model_id: str, device: str | None = None, max_length: int = 512, batch_size: int = 16):
        import torch
        from sentence_transformers import CrossEncoder
        self.model_id = model_id
        self.device = device or _default_device()
        self.batch_size = batch_size
        kwargs: dict = {"device": self.device, "max_length": max_length}
        if self.device == "cuda":
            # Half the VRAM and ZeroGPU torch.move() transfer time compared to float32
            kwargs["model_kwargs"] = {"dtype": torch.bfloat16}
        try:
            self.model = CrossEncoder(model_id, **kwargs)
        except TypeError:
            kwargs.pop("model_kwargs", None)
            self.model = CrossEncoder(model_id, **kwargs)

    def score(self, query: str, passages: list[str]) -> list[float]:
        import torch
        if not passages:
            return []
        preds = self.model.predict(
            [(query, p) for p in passages],
            batch_size=self.batch_size,
            activation_fn=torch.nn.Sigmoid(),
        )
        return [min(max(float(s), 0.0), 1.0) for s in preds]


# Backward-compatible alias for single-stage callers
Reranker = CrossEncoderScorer


@dataclass
class RerankOutcome:
    """Result of multi-stage re-reranking."""
    candidates: list[dict]
    stage1_count: int
    stage2_count: int
    mmr_applied: bool


def build_stage2_query(question: str, intent: str | None = None) -> str:
    """Combine raw question with LLM-extracted intent when intent adds meaningful context."""
    q = (question or "").strip()
    it = (intent or "").strip()
    if not it or it.lower() == q.lower() or it.lower() in q.lower():
        return q
    return f"{q} ({it})"


def mmr_reorder(
    items: list[dict],
    vectors: dict[str, np.ndarray],
    *,
    mmr_lambda: float = 0.85,
    window: int = 25,
) -> tuple[list[dict], bool]:
    """Apply Maximal Marginal Relevance over the top `window` items using normalized embeddings.

    Items without a stored vector (e.g. brand-new live search items before indexing) are kept in their
    original relevance order. Returns `(reordered_items, mmr_applied)`.
    """
    if len(items) < 2 or mmr_lambda >= 1.0 or not vectors:
        return list(items), False

    head = list(items[:window])
    tail = list(items[window:])

    # Collect indices in `head` that have valid normalized vectors
    vecs: list[np.ndarray | None] = []
    have_vec = 0
    for c in head:
        v = vectors.get(c["id"])
        if v is not None and v.size > 0:
            norm = float(np.linalg.norm(v))
            vecs.append((v / norm).astype("float32") if norm > 0 else None)
            if norm > 0:
                have_vec += 1
        else:
            vecs.append(None)

    if have_vec < 2:
        return list(items), False

    # Scale scores by the top score in head so relative relevance ratios are preserved in [0, 1]
    raw_scores = [max(0.0, float(c.get("score", c.get("relevance", 0.0)))) for c in head]
    s_max = max(raw_scores)
    norm_scores = [s / s_max if s_max > 1e-9 else 1.0 for s in raw_scores]

    remaining = list(range(len(head)))
    selected: list[int] = [remaining.pop(0)]  # always keep the #1 highest-scoring item first

    while remaining:
        best_pos = 0
        best_val = -float("inf")
        for pos, idx in enumerate(remaining):
            v = vecs[idx]
            if v is None:
                max_sim = 0.0
            else:
                sims = [float(np.dot(v, vecs[j])) for j in selected if vecs[j] is not None]
                max_sim = max(sims) if sims else 0.0
            val = mmr_lambda * norm_scores[idx] - (1.0 - mmr_lambda) * max_sim
            if val > best_val:
                best_val = val
                best_pos = pos
        selected.append(remaining.pop(best_pos))

    return [head[i] for i in selected] + tail, True


class TwoStageReranker:
    """Two-stage Sentence-Transformers cross-encoder re-reranker with optional MMR diversity."""

    def __init__(
        self,
        stage1_model_id: str | None = "BAAI/bge-reranker-base",
        stage2_model_id: str | None = "BAAI/bge-reranker-v2-m3",
        *,
        stage1_top_k: int = 40,
        stage2_weight: float = 0.7,
        mmr_lambda: float = 0.85,
        mmr_window: int = 25,
        device: str | None = None,
        stage1_device: str | None = None,
        stage2_device: str | None = None,
        stage1_scorer: PassageScorer | None = None,
        stage2_scorer: PassageScorer | None = None,
    ):
        self.stage1_top_k = max(1, int(stage1_top_k))
        self.stage2_weight = min(max(float(stage2_weight), 0.0), 1.0)
        self.mmr_lambda = min(max(float(mmr_lambda), 0.0), 1.0)
        self.mmr_window = max(2, int(mmr_window))
        s1_dev = stage1_device or device
        s2_dev = stage2_device or device

        if stage1_scorer is not None:
            self.stage1: PassageScorer = stage1_scorer
        else:
            assert stage1_model_id, "stage1_model_id or stage1_scorer is required"
            self.stage1 = CrossEncoderScorer(stage1_model_id, device=s1_dev, max_length=512, batch_size=32)

        if stage2_scorer is not None:
            self.stage2: PassageScorer | None = stage2_scorer
        elif stage2_model_id:
            try:
                max_len = 512 if s2_dev == "cpu" else 1024
                self.stage2 = CrossEncoderScorer(stage2_model_id, device=s2_dev, max_length=max_len, batch_size=16)
            except Exception as e:
                log.warning("Stage-2 reranker %s unavailable (%s); falling back to Stage 1 only", stage2_model_id, e)
                self.stage2 = None
        else:
            self.stage2 = None

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Backward-compatible single-stage scoring interface."""
        return self.stage1.score(query, passages)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        *,
        intent: str | None = None,
        vectors: dict[str, np.ndarray] | None = None,
        recency_fn: Callable[[str], float] | None = None,
    ) -> RerankOutcome:
        """Run Stage 1 coarse scoring -> Stage 2 fine re-reranking -> recency/upvote boost -> Stage 3 MMR."""
        if not candidates:
            return RerankOutcome(candidates=[], stage1_count=0, stage2_count=0, mmr_applied=False)

        # ---- Stage 1: Coarse cross-encoder on full pool ----
        s1_passages = [f"{c['title']}. {c['summary'][:1000]}" for c in candidates]
        s1_scores = [min(max(float(s), 0.0), 1.0) for s in self.stage1.score(query, s1_passages)]
        for c, s1 in zip(candidates, s1_scores):
            c["stage1_relevance"] = s1
            c["relevance"] = s1

        # ---- Stage 2: Fine cross-encoder on top-K shortlist ----
        stage2_count = 0
        if self.stage2 is not None:
            order = sorted(range(len(candidates)), key=lambda i: candidates[i]["stage1_relevance"], reverse=True)
            shortlist_idx = order[: self.stage1_top_k]
            s2_query = build_stage2_query(query, intent)
            s2_passages = [f"{candidates[i]['title']}. {candidates[i]['summary'][:1800]}" for i in shortlist_idx]
            s2_scores = [min(max(float(s), 0.0), 1.0) for s in self.stage2.score(s2_query, s2_passages)]
            w2 = self.stage2_weight
            w1 = 1.0 - w2
            for idx, s2 in zip(shortlist_idx, s2_scores):
                c = candidates[idx]
                c["stage2_relevance"] = s2
                c["relevance"] = w1 * c["stage1_relevance"] + w2 * s2
            stage2_count = len(shortlist_idx)

        # ---- Multiplicative recency + popularity boost ----
        for c in candidates:
            rec = recency_fn(c["published"]) if recency_fn else 0.0
            c["score"] = c["relevance"] * (1.0 + 0.25 * rec + 0.05 * math.log1p(c.get("upvotes", 0)))

        # Sort by boosted score before MMR
        ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)

        # ---- Stage 3: MMR diversity over top window ----
        mmr_applied = False
        if vectors and self.mmr_lambda < 1.0:
            ordered, mmr_applied = mmr_reorder(ordered, vectors, mmr_lambda=self.mmr_lambda, window=self.mmr_window)

        return RerankOutcome(
            candidates=ordered,
            stage1_count=len(candidates),
            stage2_count=stage2_count,
            mmr_applied=mmr_applied,
        )
