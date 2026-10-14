## How it's built

AI Research Navigator collects new AI papers and posts from free, open sources every day. It then helps you
decide what to read. Search finds items by keyword. The assistant reads your question, searches the
collection, checks which results are actually relevant and writes a short answer that cites them.

Everything here is open source and runs on free tiers: Hugging Face Spaces with ZeroGPU for the app, a
Hugging Face dataset for storage and GitHub Actions for the daily collection. No paid API is involved.

### How a question becomes a reading list

1. **Understand the question.** The LLM (Qwen2.5-7B-Instruct) turns your question into one to three search
   queries. It also notes any time window ("this month") and whether you want papers or blog posts.
2. **Find candidates.** Each query runs through two searches: keyword search (BM25) and meaning-based search
   (FAISS over BGE embeddings). The two rankings are merged with Reciprocal Rank Fusion, and the top 30
   items move on.
3. **Rerank.** A cross-encoder (bge-reranker-base) reads the question and each candidate together and scores
   how well they match. A small boost for recent items and for Hugging Face upvotes is added. The top 8 are
   kept.
4. **Judge relevance.** The LLM reads those 8 and marks each one relevant or not, with a one-line reason.
5. **Try again if needed.** If fewer than 3 items are relevant, the LLM rewords the search and steps 2–4
   repeat. If that still isn't enough, the app searches arXiv live and adds what it finds.
6. **Answer.** The LLM writes a short summary that cites only the relevant items, as [1], [2] and so on. The
   reading list next to it is built straight from the collection, so its titles and links can't be made up.

The "sieve" on the Ask tab shows these steps with the real counts from your question.

### Interface

| Component | What it is | What it does in this project | Where |
|---|---|---|---|
| **Gradio** | A Python library for building web apps around ML models | Builds the whole interface: the tabs, inputs, answer pane, reading list and sieve. `gr.State` keeps the conversation per visitor without exposing it in the public API. Theme and CSS give the custom look. | `ai_news/ui.py`, `app.py` |
| **Gradio MCP server** + **`mcp`** | Model Context Protocol support built into Gradio; `mcp` is the protocol library it needs | Exposes the search and assistant functions as tools that other AI agents, such as Claude or Cursor, can call. The tool descriptions come from the functions' docstrings. | `app.py` (`mcp_server=True`) |
| **Google Fonts** (Newsreader, Hanken Grotesk) | Free web fonts | Newsreader, a serif made for screen reading, is used for titles, answers and the reading list. Hanken Grotesk is used for controls and metadata. | `ai_news/ui.py` |

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
| **BAAI/bge-reranker-base** | A cross-encoder that scores how well a passage answers a query | Reorders the 30 candidates by actually reading the question and each abstract together. This is more accurate than comparing vectors, so the best 8 rise to the top. | `ai_news/llm.py` (`Reranker`) |
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
| **Recency and popularity boost** | Our own scoring | After reranking, items get a little extra score for being recent (half-life of 30 days) and for Hugging Face upvotes, so fresh and widely discussed work comes first. | `ai_news/rag/nodes.py` (`rerank`) |

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
