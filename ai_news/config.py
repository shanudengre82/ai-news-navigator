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
    llm_max_new_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_NEW_TOKENS", "700")))
    embed_model_id: str = field(default_factory=lambda: os.getenv("EMBED_MODEL_ID", "BAAI/bge-small-en-v1.5"))
    rerank_model_id: str = field(default_factory=lambda: os.getenv("RERANK_MODEL_ID", "BAAI/bge-reranker-base"))
    # Stage-2 fine cross-encoder re-reranker (set to "" or "none" to disable Stage 2 on small GPUs).
    rerank_stage2_model_id: str | None = field(default_factory=lambda: (
        None if os.getenv("RERANK_STAGE2_MODEL_ID", "BAAI/bge-reranker-v2-m3").strip().lower() in {"", "none", "false", "0"}
        else os.getenv("RERANK_STAGE2_MODEL_ID", "BAAI/bge-reranker-v2-m3").strip()
    ))
    rerank_stage1_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_STAGE1_TOP_K", "40")))
    rerank_stage2_weight: float = field(default_factory=lambda: float(os.getenv("RERANK_STAGE2_WEIGHT", "0.7")))
    rerank_mmr_lambda: float = field(default_factory=lambda: float(os.getenv("RERANK_MMR_LAMBDA", "0.85")))

    # "fetch": the Space fetches feeds itself; "pull": it only re-downloads the snapshot that the
    # GitHub Action (.github/workflows/refresh-corpus.yml) pushes to DATASET_REPO.
    refresh_mode: str = field(default_factory=lambda: os.getenv("REFRESH_MODE", "fetch").strip().lower())
    lookback_days: int = field(default_factory=lambda: int(os.getenv("LOOKBACK_DAYS", "30")))
    refresh_hours: float = field(default_factory=lambda: float(os.getenv("REFRESH_HOURS", "6")))
    refresh_on_start: bool = field(default_factory=lambda: _bool("REFRESH_ON_START", True))
    push_to_hub: bool = field(default_factory=lambda: _bool("PUSH_TO_HUB", True))

    # Guardrails & PII checks
    guardrails_enabled: bool = field(default_factory=lambda: _bool("GUARDRAILS_ENABLED", True))
    guardrails_pii_mode: str = field(default_factory=lambda: os.getenv("GUARDRAILS_PII_MODE", "redact").strip().lower())
    guardrails_max_query_chars: int = field(default_factory=lambda: int(os.getenv("GUARDRAILS_MAX_QUERY_CHARS", "2000")))

    # Provider-agnostic observability ("auto", "none", "langfuse", or a registered custom backend).
    # In "auto" mode, Langfuse is skipped completely whenever its credentials are missing.
    observability_backend: str = field(default_factory=lambda: os.getenv("OBSERVABILITY_BACKEND", "auto").strip().lower())
    langfuse_public_key: str | None = field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY") or None)
    langfuse_secret_key: str | None = field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY") or None)
    langfuse_host: str = field(default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
    langfuse_enabled: bool = field(default_factory=lambda: _bool(
        "LANGFUSE_ENABLED", bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))))
    langfuse_llm_judge: bool = field(default_factory=lambda: _bool("LANGFUSE_LLM_JUDGE", False))


settings = Settings()
