import numpy as np

from ai_news.reranker import TwoStageReranker, build_stage2_query, mmr_reorder


class FakeScorer:
    def __init__(self, scores_by_prefix: dict[str, float], default: float = 0.1):
        self.scores_by_prefix = scores_by_prefix
        self.default = default
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        out = []
        for p in passages:
            matched = next((v for k, v in self.scores_by_prefix.items() if p.startswith(k)), self.default)
            out.append(matched)
        return out


def _cand(cid: str, title: str, summary: str = "abstract", upvotes: int = 0) -> dict:
    return {
        "id": cid,
        "title": title,
        "summary": summary,
        "published": "2026-10-01T00:00:00+00:00",
        "upvotes": upvotes,
    }


def test_build_stage2_query_appends_distinct_intent():
    assert build_stage2_query("open-weight ones?", "open-weight multimodal models") == (
        "open-weight ones? (open-weight multimodal models)"
    )
    assert build_stage2_query("speculative decoding", "speculative decoding") == "speculative decoding"
    assert build_stage2_query("speculative decoding", None) == "speculative decoding"


def test_two_stage_reranker_shortlists_and_fuses_scores():
    s1 = FakeScorer({"Alpha": 0.8, "Beta": 0.7, "Gamma": 0.2})
    # Stage 2 prefers Beta strongly over Alpha; Gamma is outside top_k=2 so Stage 2 must NOT score it
    s2 = FakeScorer({"Alpha": 0.3, "Beta": 0.95, "Gamma": 0.99})

    rr = TwoStageReranker(
        stage1_scorer=s1,
        stage2_scorer=s2,
        stage1_top_k=2,
        stage2_weight=0.7,
        mmr_lambda=1.0,  # disable MMR to test pure Stage 1 + Stage 2 fusion
    )
    cands = [_cand("1", "Alpha"), _cand("2", "Beta"), _cand("3", "Gamma")]
    outcome = rr.rerank("test query", cands, intent="detailed technical intent")

    assert outcome.stage1_count == 3
    assert outcome.stage2_count == 2
    assert outcome.mmr_applied is False
    # Stage 2 received only 2 passages and included the intent in its query
    assert len(s2.calls) == 1
    assert len(s2.calls[0][1]) == 2
    assert "detailed technical intent" in s2.calls[0][0]

    # Beta: 0.3*0.7 + 0.7*0.95 = 0.875 > Alpha: 0.3*0.8 + 0.7*0.3 = 0.45 > Gamma: 0.2 (Stage 1 only)
    assert [c["title"] for c in outcome.candidates] == ["Beta", "Alpha", "Gamma"]
    assert abs(outcome.candidates[0]["relevance"] - 0.875) < 1e-6
    assert abs(outcome.candidates[1]["relevance"] - 0.45) < 1e-6
    assert "stage2_relevance" not in outcome.candidates[2]


def test_single_stage_fallback_and_backward_compatible_score():
    s1 = FakeScorer({"Alpha": 0.9, "Beta": 0.4})
    rr = TwoStageReranker(stage1_model_id=None, stage2_model_id=None, stage1_scorer=s1)
    assert rr.score("q", ["Alpha. x", "Beta. y"]) == [0.9, 0.4]

    outcome = rr.rerank("q", [_cand("1", "Beta"), _cand("2", "Alpha")])
    assert outcome.stage2_count == 0
    assert [c["title"] for c in outcome.candidates] == ["Alpha", "Beta"]


def test_mmr_diversity_promotes_distinct_item_over_near_duplicate():
    items = [
        {**_cand("a", "Paper A"), "score": 0.95},
        {**_cand("a_dup", "Blog recap of Paper A"), "score": 0.93},
        {**_cand("b", "Distinct Paper B"), "score": 0.88},
    ]
    # a and a_dup have nearly identical embeddings; b is orthogonal
    vecs = {
        "a": np.array([1.0, 0.0, 0.0], dtype="float32"),
        "a_dup": np.array([0.99, 0.05, 0.0], dtype="float32"),
        "b": np.array([0.0, 1.0, 0.0], dtype="float32"),
    }
    reordered, applied = mmr_reorder(items, vecs, mmr_lambda=0.6, window=10)
    assert applied is True
    assert [c["id"] for c in reordered] == ["a", "b", "a_dup"]
