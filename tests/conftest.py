import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_news.schema import NewsItem  # noqa: E402
from ai_news.store import CorpusStore  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def days_ago(n: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).replace(microsecond=0).isoformat()


def make_item(title, summary="", source="arXiv", source_type="paper", age_days=1.0, url=None, upvotes=0):
    return NewsItem(title=title, summary=summary, source=source, source_type=source_type,
                    published=days_ago(age_days), url=url or f"https://example.com/{abs(hash(title))}",
                    upvotes=upvotes)


@pytest.fixture
def items():
    return [
        make_item("Test-time compute scaling for reasoning models",
                  "We scale inference compute with verifiers and search to improve math reasoning.", age_days=2),
        make_item("Efficient LoRA fine-tuning of vision transformers",
                  "Low-rank adapters for ViT image classification.", age_days=10),
        make_item("Introducing our new open-weight multimodal model",
                  "A blog post announcing a vision-language model with open weights.",
                  source="Hugging Face Blog", source_type="blog", age_days=1),
        make_item("Retrieval-augmented generation survey",
                  "A survey of RAG: retrievers, rerankers, and generation with citations.", age_days=40),
        make_item("Diffusion models for video editing",
                  "Soft masks for pixel-level video diffusion editing.", age_days=200),
    ]


@pytest.fixture
def store(tmp_path, items):
    s = CorpusStore(tmp_path, DeterministicFakeEmbedding(size=32))
    s.add(items)
    return s
