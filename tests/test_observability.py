from dataclasses import dataclass, field

from ai_news.config import Settings
from ai_news.observability import (
    RagTracer,
    _reset_client_after_fork,
    compute_rag_scores,
    is_enabled,
    summarize_node_object,
    trace_polish_run,
)


class FakeSpan:
    def __init__(self, name, inp=None, output=None, metadata=None, trace_id="trace-1"):
        self.name = name
        self.input = inp
        self.output = output
        self.metadata = metadata
        self.trace_id = trace_id
        self.children = []
        self.ended = False

    def start_span(self, name, input=None, output=None, metadata=None):
        child = FakeSpan(name, inp=input, output=output, metadata=metadata, trace_id=self.trace_id)
        self.children.append(child)
        return child

    def update(self, output=None, metadata=None):
        if output is not None:
            self.output = output
        if metadata is not None:
            self.metadata = metadata

    def end(self):
        self.ended = True


class FakeLangfuse:
    def __init__(self):
        self.spans: list[FakeSpan] = []
        self.scores: list[dict] = []
        self.flushed = 0

    def start_span(self, name, input=None, output=None, metadata=None):
        sp = FakeSpan(name, inp=input, output=output, metadata=metadata)
        self.spans.append(sp)
        return sp

    def create_score(self, trace_id, name, value, comment=None):
        self.scores.append({"trace_id": trace_id, "name": name, "value": value, "comment": comment})

    def flush(self):
        self.flushed += 1


def test_is_enabled_respects_keys_and_flag():
    assert not is_enabled(Settings(langfuse_public_key=None, langfuse_secret_key=None, langfuse_enabled=True))
    assert not is_enabled(Settings(langfuse_public_key="pk", langfuse_secret_key="sk", langfuse_enabled=False))
    assert is_enabled(Settings(langfuse_public_key="pk", langfuse_secret_key="sk", langfuse_enabled=True))


def test_rag_tracer_noop_when_unconfigured():
    cfg = Settings(langfuse_public_key=None, langfuse_secret_key=None, langfuse_enabled=False)
    with RagTracer("test question", cfg=cfg) as tracer:
        assert not tracer.active
        assert tracer.langchain_callbacks() == []
        tracer.record_node("retrieve", {"candidates": []})
        scores = tracer.finish({"guardrail_blocked": False, "answer": "ok"})
    assert scores["guardrail_passed"][0] == 1.0


def test_rag_tracer_records_redacted_objects_scores_and_flushes():
    fake = FakeLangfuse()
    cfg = Settings(langfuse_public_key="pk", langfuse_secret_key="sk", langfuse_enabled=True)
    final_state = {
        "guardrail_blocked": False,
        "pii_findings": [{"entity_type": "EMAIL", "count": 1}],
        "pool_size": 10,
        "rerank_kept": 4,
        "matches": [
            {"id": "1", "title": "Speculative Decoding", "relevance": 0.9, "score": 1.1},
            {"id": "2", "title": "Draft Models", "relevance": 0.7, "score": 0.8},
        ],
        "verdicts": {"1": {"relevant": True, "digest": "d1"}, "2": {"relevant": False, "digest": ""}},
        "references": [
            {"id": "1", "title": "Speculative Decoding", "summary": "Fast inference with draft models.", "weak": False},
            {"id": "2", "title": "Other", "summary": "Unrelated.", "weak": True},
        ],
        "summarized": 1,
        "retries": 0,
        "live_searched": False,
        "answer": "Speculative decoding accelerates inference with draft models [1]. **Start with** [1].",
        "output_guardrail": {"valid_citations": [1, 1], "invalid_citations": [], "hallucinated_urls": []},
    }

    with RagTracer("Papers for alice@example.com on speculative decoding", corpus_size=500, cfg=cfg, client=fake) as tr:
        tr.record_node("input_guardrail", {"guardrail_blocked": False, "pii_findings": final_state["pii_findings"]})
        tr.record_node("rerank", {"matches": final_state["matches"], "pool_size": 10, "rerank_kept": 4})
        tr.finish(final_state)

    assert fake.flushed == 1
    assert len(fake.spans) == 1
    root = fake.spans[0]
    assert root.ended
    assert "alice@example.com" not in root.input["question"]
    assert "[REDACTED_EMAIL]" in root.input["question"]
    assert [c.name for c in root.children] == ["node:input_guardrail", "node:rerank"]
    assert root.children[1].output["relevance_stats"]["max"] == 0.9

    score_map = {s["name"]: s["value"] for s in fake.scores}
    assert score_map["guardrail_passed"] == 1.0
    assert score_map["pii_detected"] == 1.0
    assert score_map["retrieval_top1_relevance"] == 0.9
    assert score_map["retrieval_rerank_keep_rate"] == 0.4
    assert score_map["retrieval_grade_pass_rate"] == 0.5
    assert score_map["retrieval_strong_ref_ratio"] == 0.5
    assert score_map["retrieval_retry_count"] == 0.0
    assert score_map["answer_citation_validity"] == 1.0
    assert score_map["answer_citation_coverage"] == 1.0
    assert score_map["answer_has_start_with"] == 1.0
    assert score_map["answer_lexical_grounding"] > 0.5


def test_compute_rag_scores_for_blocked_input():
    scores = compute_rag_scores({"guardrail_blocked": True, "guardrail_reason": "prompt injection"})
    assert scores["guardrail_passed"][0] == 0.0
    assert "retrieval_top1_relevance" not in scores


def test_trace_polish_run_and_fork_reset():
    fake = FakeLangfuse()
    trace_polish_run(
        "Draft with secret@example.com https://arxiv.org/abs/1",
        "Polished https://arxiv.org/abs/1",
        "",
        blocked=False,
        pii_detected=True,
        client=fake,
    )
    assert fake.flushed == 1
    assert "secret@example.com" not in fake.spans[0].input["draft_preview"]
    score_map = {s["name"]: s["value"] for s in fake.scores}
    assert score_map == {
        "polish_guardrail_passed": 1.0,
        "polish_pii_detected": 1.0,
        "polish_links_preserved": 1.0,
    }
    _reset_client_after_fork()  # should execute cleanly without error
