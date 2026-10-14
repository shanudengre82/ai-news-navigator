---
title: AI News & Research Navigator
emoji: 🛰️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
python_version: "3.12"
short_description: Agentic RAG search over the latest AI papers & news
startup_duration_timeout: 1h
license: apache-2.0
tags:
  - rag
  - langgraph
  - langchain
  - arxiv
  - mcp-server
---

# 🛰️ AI News & Research Navigator

Keep up with AI research without paying for anything. The app collects new papers and posts from open sources, then lets you search them in two ways:

- **🔎 Keyword Search**: BM25 search with filters for source, type, date and sort order. Put a phrase in `"quotes"` to require it word for word.
- **🤖 Research Assistant**: an agentic RAG pipeline in **LangGraph**, using **LangChain** components and an open LLM (Qwen2.5-7B-Instruct on ZeroGPU). It filters the documents and returns the best references, each with a reason it was chosen and citations.

## Sources (all free and open; edit `config/sources.yaml`)
| Kind | Sources |
|---|---|
| Papers | arXiv API (cs.AI, cs.CL, cs.LG, cs.CV, cs.MA, cs.IR, stat.ML), Hugging Face Daily Papers (community upvotes) |
| Blogs / news | Hugging Face, Google Research, Google DeepMind, OpenAI, Microsoft Research, BAIR, MIT News AI, The Gradient, Lil'Log, Ahead of AI, Simon Willison, NVIDIA, AWS ML, Import AI, MarkTechPost, TechCrunch AI |

## How the assistant works
```
analyze_query ─► retrieve ─► rerank ─► grade_documents ─┬─ ≥3 relevant ───────────► generate
 (LLM: intent,    (BM25 +     (bge-reranker  (LLM judge   ├─ too few, 1st try ─► rewrite_query ─► retrieve …
  queries, dates)  FAISS, RRF)  + recency)    per doc)    └─ still too few ───► live arXiv search ─► rerank …
```
- **Embeddings**: `BAAI/bge-small-en-v1.5` on CPU, so background refreshes need no GPU.
- **Vector index**: FAISS (cosine similarity). **Keyword index**: `rank_bm25`. The two are fused with reciprocal rank fusion.
- **Reranker**: `BAAI/bge-reranker-base` cross-encoder, plus a small boost for recency and HF upvotes.
- **LLM**: `Qwen/Qwen2.5-7B-Instruct` through `transformers` + `langchain-huggingface`. Set `LLM_MODEL_ID` to change it.
- **Grounding**: the answer may cite only the retrieved sources. The reference list is built from the corpus, never written by the LLM, so its links can't be invented.

## Storage and refresh (free)
The corpus (`items.parquet`, `index.faiss`, `meta.json`) lives in a **Hugging Face Dataset repo**:
1. On startup the Space downloads it.
2. While the Space is awake, it refreshes every `REFRESH_HOURS` (6 by default) and pushes the result back.
3. The **⚙️ Status** tab has a "Refresh now" button.

## Configuration (Space secrets / variables)
| Name | Default | Purpose |
|---|---|---|
| `HF_TOKEN` (secret) | – | Write token, used to push corpus refreshes to the dataset repo |
| `DATASET_REPO` | – | e.g. `your-name/ai-news-corpus` |
| `LLM_BACKEND` | `local` | `local` runs a transformers model in-process; `api` uses the HF Inference API (free credits) |
| `LLM_MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` | Any open chat model |
| `LOOKBACK_DAYS` | `30` | How many days of history a refresh fetches |
| `REFRESH_MODE` | `fetch` | `fetch`: the Space fetches the feeds itself. `pull`: it only downloads the snapshot that the GitHub Action pushes |
| `REFRESH_HOURS` | `6` | How often to fetch or pull; `0` turns it off |

## Run locally
```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install gradio spaces huggingface_hub -r requirements.txt pytest
pytest -q
python scripts/build_corpus.py --days 14 --no-push      # build the local corpus in ./data
# A small model fits a 12 GB GPU; on CPU-only machines use LLM_BACKEND=api with HF_TOKEN set.
LLM_MODEL_ID=Qwen/Qwen2.5-3B-Instruct REFRESH_HOURS=0 python app.py
```

## Deploy with GitHub Actions (free)
```
GitHub (push to main) ──► sync-to-hf.yml     ──► HF Space shanudengre82/ai-news-navigator (ZeroGPU)
GitHub (every 12 h)   ──► refresh-corpus.yml ──► HF Dataset shanudengre82/ai-news-corpus ──► Space pulls it
```
A free account can host Gradio Spaces only on **ZeroGPU**: up to 2 Spaces, and the account needs a verified email and must be older than 30 days.

**One-time setup**
1. **GitHub repo:** go to **Settings → Secrets and variables → Actions**.
   - Add the secret `HF_TOKEN`: a Hugging Face **write** token from https://huggingface.co/settings/tokens.
   - Optionally add the variables `HF_SPACE` and `DATASET_REPO`. The workflows default to the names above.
2. **HF Space:** go to **Settings → Variables and secrets** and add these variables:
   - `DATASET_REPO=shanudengre82/ai-news-corpus`
   - `REFRESH_MODE=pull`
   - `REFRESH_HOURS=1`

   If the dataset is private, also add the secret `HF_TOKEN`. Under **Settings → Hardware**, check that the hardware is **ZeroGPU**.
3. Seed the data: on GitHub, open **Actions → Refresh corpus → Run workflow** (about 15–20 minutes).
4. Push to `main`, or run **Actions → Sync to Hugging Face Space** by hand. The Space rebuilds with the new code.

**What each workflow does**
- `sync-to-hf.yml`: on every push to `main`, it uploads the code to the Space with `hf upload`. `tests/`, `data/` and `.github/` are not uploaded.
- `refresh-corpus.yml`: every 12 hours, it fetches all sources on a CPU runner and embeds only the new items. It then pushes `items.parquet`, `index.faiss` and `meta.json` to the dataset. The Space picks up the new snapshot within `REFRESH_HOURS`.

**Without GitHub**
```bash
hf auth login
python scripts/build_corpus.py --days 30 --dataset-repo <you>/ai-news-corpus
hf upload <you>/ai-news-navigator . --repo-type space \
   --exclude ".venv/**" --exclude "data/**" --exclude "**/__pycache__/**" --exclude "tests/**"
```
The app also runs as an **MCP server** (`/gradio_api/mcp/sse`), so agents can call its search and assistant tools directly.
