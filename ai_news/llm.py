"""LLM / reranker factories. Everything is open-weight and runs for free.

- local: transformers pipeline in-process. On a ZeroGPU Space the model is moved to "cuda" at import time
  and ZeroGPU attaches a GPU only while a @spaces.GPU function runs.
- api:   Hugging Face Inference API (uses HF_TOKEN's free monthly credits) — handy for local dev without a GPU.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

log = logging.getLogger(__name__)


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_chat_model(backend: str, model_id: str, max_new_tokens: int = 900, hf_token: str | None = None) -> BaseChatModel:
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline

    if backend == "api":
        llm = HuggingFaceEndpoint(repo_id=model_id, task="text-generation", max_new_tokens=max_new_tokens,
                                  temperature=0.1, huggingfacehub_api_token=hf_token)
        return ChatHuggingFace(llm=llm)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    device = _device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    log.info("loading %s on %s (%s)", model_id, device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, token=hf_token).to(device)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens,
                    do_sample=False, return_full_text=False)
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe), tokenizer=tokenizer, model_id=model_id)


class Reranker:
    """Cross-encoder reranker (BAAI/bge-reranker-base)."""

    def __init__(self, model_id: str, device: str | None = None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_id, device=device or _device(), max_length=512)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self.model.predict([(query, p) for p in passages], batch_size=16)]


def load_embeddings(model_id: str):
    """CPU embeddings: background refreshes run outside any GPU context."""
    from langchain_huggingface import HuggingFaceEmbeddings
    query_kwargs = {"normalize_embeddings": True}
    if "bge-" in model_id and "-en" in model_id:  # BGE v1.5 retrieves better with its query instruction
        query_kwargs["prompt"] = "Represent this sentence for searching relevant passages: "
    return HuggingFaceEmbeddings(model_name=model_id, model_kwargs={"device": "cpu"},
                                 encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
                                 query_encode_kwargs=query_kwargs)
