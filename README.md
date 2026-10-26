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

- **Ask**: an agentic RAG pipeline in **LangGraph**, using **LangChain** components and an open LLM (Qwen2.5-7B-Instruct on ZeroGPU). You get an overall summary with citations, then every matching reference (not a fixed top-N), each with a short summary; sort and filter the list afterwards by relevance, date, upvotes, type or source. A live *sieve* on the right shows how the collection was narrowed at each step: pulled, reranked, reviewed, listed.
- **Search**: BM25 keyword search over every match (50 at a time) with filters for source, type, date and sort order. Put a phrase in `"quotes"` to require it word for word.
- **Latest**: the newest papers and posts, filtered by source and type.

## Sources (all free and open; edit `config/sources.yaml`)
| Kind | Sources |
|---|---|
| Papers | arXiv API (cs.AI, cs.CL, cs.LG, cs.CV, cs.MA, cs.IR, stat.ML), Hugging Face Daily Papers (community upvotes) |
| Blogs / news | Hugging Face, Google Research, Google DeepMind, OpenAI, Microsoft Research, BAIR, MIT News AI, The Gradient, Lil'Log, Ahead of AI, Simon Willison, NVIDIA, AWS ML, Import AI, MarkTechPost, TechCrunch AI |

<!-- about:start -->
## How it's built

AI Research Navigator collects new AI papers and posts from free, open sources every day. It then helps you
decide what to read. Search finds items by keyword. The assistant reads your question, searches the
collection, keeps everything that is relevant and writes a short summary that cites the best of it,
followed by every matching reference.

Everything here is open source and runs on free tiers: Hugging Face Spaces with ZeroGPU for the app, a
Hugging Face dataset for storage and GitHub Actions for the daily collection. No paid API is involved.

### How a question becomes a reading list

1. **Understand the question.** The LLM (Qwen2.5-7B-Instruct) turns your question into one to three search
   queries. It also notes any time window ("this month") and whether you want papers or blog posts.
2. **Find candidates.** Each query runs through two searches: keyword search (BM25) and meaning-based search
   (FAISS over BGE embeddings). The two rankings are merged with Reciprocal Rank Fusion into a wide pool of
   up to 300 items (a limit on compute, not on results).
3. **Rerank.** A cross-encoder (bge-reranker-base) reads the question and each candidate together and scores
   how well they match. Every item above a relevance threshold is kept, so a broad question can return a
   hundred matches and a narrow one a handful. Recent items and Hugging Face upvotes get a small boost.
   Each reference shows its **match %**: the reranker's estimate that it is relevant to your question.
4. **Review.** The LLM reads the 10 best matches, drops any that are off-topic and writes a short summary
   of each one it keeps.
5. **Try again if needed.** If fewer than 3 items pass the review, the LLM rewords the search and steps 2–4
   repeat (only new items are reviewed). If that still isn't enough, the app searches arXiv live.
6. **Answer.** The LLM writes an overall summary from the reviewed items and cites them as [1], [2] and so
   on. Below it comes every match: reviewed items with the model's summary, the rest with their abstract.
   You always get at least 5 references: if fewer match, the closest low-scoring items fill the list,
   marked "Low match" under a warning, so a weak result is never presented as a strong one.
   The list is built straight from the collection, so its titles and links can't be made up.

On the Ask tab your question and the list's filters sit on the left, the results in the center and the
assistant's steps on the right. The steps panel (the "sieve") shows the real count at each step. Sort and
filter the references afterwards (relevance, newest, oldest, most upvoted; type, date, source) without
asking again; the [n] numbers stay attached to their items.

### Interface

| Component | What it is | What it does in this project | Where |
|---|---|---|---|
| **Gradio** | A Python library for building web apps around ML models | Builds the whole interface: the tabs, the three-pane Ask desk, the reference list with its sort and filters, and the sieve. `gr.State` keeps the conversation per visitor without exposing it in the public API. Theme and CSS give the custom look. | `ai_news/ui.py`, `app.py` |
| **Gradio MCP server** + **`mcp`** | Model Context Protocol support built into Gradio; `mcp` is the protocol library it needs | Exposes the search and assistant functions as tools that other AI agents, such as Claude or Cursor, can call. The tool descriptions come from the functions' docstrings. | `app.py` (`mcp_server=True`) |
| **Google Fonts** (Newsreader, Hanken Grotesk) | Free web fonts | Newsreader, a serif made for screen reading, is used for titles, summaries and references. Hanken Grotesk is used for controls and metadata. | `ai_news/ui.py` |
| **Iframe height reporter** | A small script in the page head (`FIT_JS`) | On huggingface.co the app runs inside an iframe that only grows when the app reports its height. Gradio's own reporter can stop growing after a few quick changes (a streaming answer, fonts loading), which left the page unable to scroll. The script measures the real content height and reports it whenever it changes. | `ai_news/ui.py`, `app.py` |

### Orchestration: LangChain and LangGraph

| Component | What it is | What it does in this project | Where |
|---|---|---|---|
| **LangGraph** | A library for building LLM workflows as graphs of steps with shared state | Runs the assistant as a `StateGraph` with seven nodes: analyze, retrieve, rerank, grade, rewrite, live search and generate. **Conditional edges** decide what happens after grading: answer, reword or search arXiv. **Streaming** sends each finished step to the page as it happens, which drives the sieve. | `ai_news/rag/graph.py`, `ai_news/rag/nodes.py`, `ai_news/rag/state.py` |
| **LangChain Core** (`langchain-core`) | The common interfaces of the LangChain ecosystem: messages, chat models and embeddings | Gives every model the same interface. The graph calls `llm.invoke([HumanMessage(...)])` whether the model runs locally or through an API. Tests swap in `FakeListChatModel` and `DeterministicFakeEmbedding`, so the whole pipeline is tested without downloading a model. | `ai_news/rag/nodes.py`, `tests/` |
| **`langchain-huggingface`** | LangChain's official Hugging Face integration | `HuggingFacePipeline` + `ChatHuggingFace` wrap the local transformers model with its chat template. `HuggingFaceEndpoint` is the alternative that calls the HF Inference API, for local development without a GPU. `HuggingFaceEmbeddings` produces the vectors used for search. | `ai_news/llm.py` |

### Models (all open weights, from the Hugging Face Hub)

| Component | What it is | What it does in this project | Where |
|---|---|---|---|
| **Qwen2.5-7B-Instruct** | A 7-billion-parameter open chat LLM from Alibaba's Qwen team | The "brain" of the assistant: it plans the searches, judges relevance, rewords searches and writes the cited answer. It returns structured JSON for the planning and judging steps. You can switch it with `LLM_MODEL_ID`, for example to Qwen2.5-3B on smaller GPUs. | `ai_news/llm.py`, `ai_news/rag/prompts.py` |
| **BAAI/bge-small-en-v1.5** | A small (33M parameter) embedding model that turns text into 384-number vectors | Embeds every title + abstract, and every search query, for meaning-based search. It is small enough to run on CPU, so the background collection needs no GPU. Queries get BGE's recommended retrieval prefix. | `ai_news/llm.py` (`load_embeddings`) |
| **BAAI/bge-reranker-base** | A cross-encoder that scores how well a passage answers a query | Scores every candidate in the pool by actually reading the question and each abstract together. This is more accurate than comparing vectors; everything above the relevance threshold becomes a match. | `ai_news/llm.py` (`Reranker`) |
| **Transformers** | Hugging Face's library for loading and running models | Loads Qwen and its tokenizer and runs text generation. | `ai_news/llm.py` |
| **Accelerate** | Hugging Face's helper for placing models on devices | Used by transformers to load the large model efficiently onto the GPU. | (used by transformers) |
| **PyTorch** (`torch`) | The deep-learning framework all three models run on | Runs the neural networks, on the GPU in the Space and on CPU for the embeddings. | (used by all models) |
| **Sentence Transformers** | A library for embedding and cross-encoder models | Runs the BGE embedding model (through `HuggingFaceEmbeddings`) and the BGE reranker (`CrossEncoder`). | `ai_news/llm.py` |

### Search and ranking

| Component | What it is | What it does in this project | Where |
|---|---|---|---|
| **FAISS** (`faiss-cpu`) | Meta's library for fast similarity search over vectors | Stores one vector per item in an `IndexIDMap2(IndexFlatIP)`, which gives exact cosine similarity and lets items be added and removed by id. It finds the items closest in meaning to a query. The index is saved as `index.faiss` next to the data. | `ai_news/store.py` (`VectorIndex`) |
| **`rank-bm25`** | An implementation of BM25, the classic keyword-ranking formula used by search engines | Powers the Search tab, and the keyword half of the assistant's search. It catches exact model names and acronyms that embeddings can blur. | `ai_news/store.py`, `ai_news/search.py`, `ai_news/text.py` |
| **Reciprocal Rank Fusion** | A simple method for merging several rankings (score = sum of 1 / (60 + rank)) | Combines the BM25 and FAISS results into one candidate list without having to calibrate their very different scores. | `ai_news/rag/nodes.py` (`retrieve`) |
| **Recency and popularity boost** | Our own scoring | After reranking, an item's relevance is scaled up a little for being recent (half-life of 30 days) and for Hugging Face upvotes, so among equally relevant items fresh and widely discussed work comes first. | `ai_news/rag/nodes.py` (`rerank`) |

### Data sources and ingestion

| Component | What it is | What it does in this project | Where |
|---|---|---|---|
| **arXiv API** + **arXiv RSS** | arXiv's free query API and its daily announcement feeds | Provide the newest papers in cs.AI, cs.CL, cs.LG, cs.CV, cs.MA, cs.IR and stat.ML. Queries go one category at a time with a 3-second pause, because larger queries get rate-limited. The API also powers the assistant's live search. | `ai_news/fetchers/arxiv.py` |
| **Hugging Face Daily Papers API** | The community-curated paper feed at huggingface.co/papers | Adds hand-picked papers and their upvotes, which feed the popularity boost. | `ai_news/fetchers/hf_papers.py` |
| **Blog RSS feeds** | Feeds from Hugging Face, Google Research, DeepMind, OpenAI, Microsoft Research, BAIR, NVIDIA, AWS and others | Cover announcements and explainers that never appear on arXiv. The list is in `config/sources.yaml`, and new feeds can be added there. | `ai_news/fetchers/rss.py` |
| **feedparser** | A robust RSS/Atom parser | Reads the arXiv and blog feeds, including dates and authors. | `ai_news/fetchers/` |
| **HTTPX** | A modern HTTP client | Downloads every source with timeouts and retries, and backs off politely on HTTP 429 or 503, honouring `Retry-After`. | `ai_news/fetchers/common.py` |
| **PyYAML** (`pyyaml`) | A YAML parser | Reads `config/sources.yaml`, the editable list of sources. | `ai_news/fetchers/__init__.py` |
| **pandas** + **PyArrow** | Data-frame and columnar-file libraries | Save and load the collection as `items.parquet`, a compact, fast file format. | `ai_news/store.py` |
| **Deduplication** | Our own logic | One canonical URL per item, so arXiv versions, PDF links and HF paper pages of the same paper merge into a single entry, keeping the highest upvote count. Items older than 180 days are pruned. | `ai_news/schema.py`, `ai_news/store.py` |

### Hosting, storage and automation

| Component | What it is | What it does in this project | Where |
|---|---|---|---|
| **Hugging Face Spaces** | Free hosting for ML web apps | Hosts the Gradio app. | `README.md` (Space settings) |
| **ZeroGPU** (`spaces`) | A free, shared GPU pool for Spaces | Lends the app a GPU only while `@spaces.GPU` code runs. Here that is one assistant answer, with 75 seconds reserved, and each visitor uses their own daily GPU quota. The GPU code runs in a separate process, so the app resets its locks after the fork and sends results back as simple messages. | `app.py`, `ai_news/assistant.py`, `ai_news/store.py` |
| **Hugging Face Dataset repo** | A free versioned file store on the Hub | Keeps the latest `items.parquet`, `index.faiss` and `meta.json` at `shanudengre82/ai-news-corpus`. A restarted Space downloads it in seconds instead of rebuilding everything. | `ai_news/store.py` (`pull` / `push`) |
| **`huggingface_hub`** + **`hf` CLI** | The Python client and command-line tool for the Hub | `snapshot_download` and `upload_folder` move the data to and from the dataset; the CLI uploads the code to the Space. | `ai_news/store.py`, `.github/workflows/` |
| **GitHub Actions** | Free CI/CD from GitHub | `sync-to-hf.yml` deploys the code to the Space on every push. `refresh-corpus.yml` collects new items every 12 hours on a CPU runner and pushes them to the dataset. | `.github/workflows/` |
| **Refresh scheduler** | A background thread in the app | In `pull` mode (the setting used on the Space), it downloads the newest dataset copy every hour. In `fetch` mode it collects the sources itself. | `ai_news/scheduler.py`, `ai_news/ingest.py` |

### Quality checks

| Component | What it is | What it does in this project | Where |
|---|---|---|---|
| **pytest** | Python's standard testing tool | Tests the parsers on saved feed samples, the deduplication, search filters, the graph's routing (fake LLM), the error handling, the sieve and the fork safety. It also checks that this page names every dependency. | `tests/` |
| **`gradio_client`** | Gradio's Python client | `scripts/smoke_test.py` calls the running app or Space end to end: status, search and a full assistant answer. | `scripts/smoke_test.py` |
| **Playwright** | Browser automation (development only, not deployed) | Took screenshots at desktop and phone sizes in light and dark mode while the interface was being designed. | (local tooling) |

### Why these choices

- **Free and open.** Every model, library and service here is open source or has a free tier, so the
  project costs nothing to run.
- **Hybrid search.** BM25 finds exact terms, embeddings find related ideas, and the reranker reads both
  closely. Together they find better candidates than any one of them alone.
- **The LLM judges before it answers.** Grading each result first, and retrying when too few pass, keeps the
  answer grounded in sources that actually fit the question.
- **Trade-offs.** CPU embeddings keep collection free but slow, so a full refresh takes about 7–15 minutes on
  GitHub's runners. ZeroGPU is free but limited per visitor, so the assistant's GPU time is kept short;
  logged-in visitors get more. A 7B model gives good answers on the Space, while a 3B model is enough for
  local testing on a 12 GB GPU.
<!-- about:end -->

## Storage and refresh (free)
The corpus (`items.parquet`, `index.faiss`, `meta.json`) lives in a **Hugging Face Dataset repo**:
1. On startup the Space downloads it.
2. While the Space is awake, it refreshes every `REFRESH_HOURS` (6 by default) and pushes the result back.
3. The **Sources** tab shows the health of each source and has a button to update the collection now.

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
