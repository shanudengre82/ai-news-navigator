"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN"))
    # HF Dataset repo that persists the corpus + FAISS index, e.g. "user/ai-news-corpus".
    dataset_repo: str | None = field(default_factory=lambda: os.getenv("DATASET_REPO") or None)
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", ROOT / "data")))
    sources_file: Path = field(default_factory=lambda: Path(os.getenv("SOURCES_FILE", ROOT / "config" / "sources.yaml")))

    # "local" = transformers pipeline in-process (ZeroGPU / local GPU / CPU); "api" = HF Inference API.
    llm_backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "local"))
    llm_model_id: str = field(default_factory=lambda: os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"))
    llm_max_new_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_NEW_TOKENS", "900")))
    embed_model_id: str = field(default_factory=lambda: os.getenv("EMBED_MODEL_ID", "BAAI/bge-small-en-v1.5"))
    rerank_model_id: str = field(default_factory=lambda: os.getenv("RERANK_MODEL_ID", "BAAI/bge-reranker-base"))

    lookback_days: int = field(default_factory=lambda: int(os.getenv("LOOKBACK_DAYS", "30")))
    refresh_hours: float = field(default_factory=lambda: float(os.getenv("REFRESH_HOURS", "6")))
    refresh_on_start: bool = field(default_factory=lambda: _bool("REFRESH_ON_START", True))
    push_to_hub: bool = field(default_factory=lambda: _bool("PUSH_TO_HUB", True))


settings = Settings()
