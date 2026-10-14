"""Gradio Blocks UI."""

from __future__ import annotations

import html
from typing import Callable

import gradio as gr

from ai_news.rag.state import Candidate
from ai_news.schema import NewsItem
from ai_news.search import DATE_WINDOWS, highlight, keyword_search
from ai_news.store import CorpusStore

TYPE_CHOICES = [("Papers & blogs", "any"), ("Papers", "paper"), ("Blogs / news", "blog")]
EXAMPLES = [
    "What are the latest approaches to scaling test-time compute for reasoning models?",
    "Which new open-weight multimodal models were released this month?",
    "Recent papers on making RAG more faithful with citations",
    "Best recent work on efficient fine-tuning (LoRA and variants) for LLMs",
    "What's new in AI agents that use tools and browse the web?",
]

CSS = """
.card {border:1px solid var(--border-color-primary); border-radius:10px; padding:12px 14px; margin:0 0 10px;
       background:var(--block-background-fill);}
.card a.title {font-weight:600; font-size:1.02em; text-decoration:none;}
.card .meta {font-size:.82em; opacity:.75; margin:4px 0 6px;}
.card .badge {display:inline-block; padding:1px 8px; border-radius:999px; font-size:.78em; margin-right:6px;
              background:var(--color-accent-soft); color:var(--body-text-color);}
.card .summary {font-size:.9em; line-height:1.45;}
.card mark {padding:0 2px; border-radius:3px;}
"""


def card(it: NewsItem, query: str = "", score: float | None = None, chars: int = 360) -> str:
    summary = it.summary[:chars] + ("…" if len(it.summary) > chars else "")
    up = f" · 👍 {it.upvotes}" if it.upvotes else ""
    pdf = f' · <a href="{html.escape(it.pdf_url)}" target="_blank">PDF</a>' if it.pdf_url else ""
    authors = ", ".join(it.authors[:4]) + (" et al." if len(it.authors) > 4 else "")
    return (f'<div class="card"><a class="title" href="{html.escape(it.url)}" target="_blank">'
            f'{highlight(it.title, query)}</a>'
            f'<div class="meta"><span class="badge">{html.escape(it.source)}</span>{it.published[:10]}{up}{pdf}'
            f'{" · " + html.escape(authors) if authors else ""}</div>'
            f'<div class="summary">{highlight(summary, query)}</div></div>')


def cards(items: list[NewsItem], query: str = "", header: str = "") -> str:
    if not items:
        return "<p>No results. Try fewer filters or different words.</p>"
    return f"<p>{header}</p>" + "".join(card(it, query) for it in items)


def format_references(refs: list[Candidate]) -> str:
    if not refs:
        return "_No references yet._"
    lines = []
    for n, c in enumerate(refs, 1):
        up = f" · 👍 {c['upvotes']}" if c.get("upvotes") else ""
        lines.append(f"**[{n}] [{c['title']}]({c['url']})**  \n"
                     f"{c['source']} · {c['published'][:10]}{up}"
                     + (f"  \n_Why: {c['reason']}_" if c.get("reason") else ""))
    return "\n\n".join(lines)


def build_ui(store: CorpusStore, source_names: list[str], assistant_fn: Callable,
             refresh_fn: Callable[[], str], pull_mode: bool = False) -> gr.Blocks:

    def search(query: str, sources: list[str], source_type: str, window: str, sort: str) -> str:
        """Keyword (BM25) search over collected AI papers and posts. Quote a phrase to require it verbatim.

        Args:
            query: keywords, e.g. 'mixture of experts "load balancing"'
            sources: restrict to these sources (empty = all)
            source_type: "any", "paper" or "blog"
            window: one of "Any time", "Last 24 hours", "Last 3 days", "Last week", "Last month", "Last 3 months"
            sort: "relevance", "date" or "upvotes"
        """
        hits = keyword_search(store, query, sources or None, source_type, DATE_WINDOWS.get(window), sort)
        return cards([h[0] for h in hits], query, f"{len(hits)} result(s)")

    def latest(sources: list[str], source_type: str, n: float) -> str:
        """The newest items in the corpus, optionally filtered by source and type."""
        hits = keyword_search(store, "", sources or None, source_type, None, "date", int(n))
        return cards([h[0] for h in hits])

    def status() -> str:
        """Corpus size, last refresh time and per-source health."""
        s = store.stats()
        rows = "\n".join(f"| {name} | {s['by_source'].get(name, 0)} | {s['source_status'].get(name, '—')} |"
                         for name in sorted(set(s["by_source"]) | set(s["source_status"])))
        return (f"**{s['total']} items** · last refresh: {s['last_refresh'] or 'never'} (UTC)\n\n"
                f"| Source | Items | Last fetch |\n|---|---:|---|\n{rows}")

    def refresh_now() -> tuple[str, str]:
        """Update the corpus now: fetch all sources (fetch mode, takes minutes) or download the latest
        snapshot from the HF Dataset repo (pull mode)."""
        return refresh_fn(), status()

    with gr.Blocks(title="AI News & Research Navigator") as demo:
        gr.Markdown("# 🛰️ AI News & Research Navigator\n"
                    "Fresh AI papers and posts from arXiv, Hugging Face Daily Papers and leading AI lab blogs. "
                    "Search by keyword, or ask the **Research Assistant**: an open LLM agent "
                    "(LangGraph corrective RAG over a FAISS + BM25 index) that finds, checks and explains the best references.")
        with gr.Tabs():
            with gr.Tab("🤖 Research Assistant"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chat = gr.Chatbot(height=520, label="Assistant", show_label=False)
                        with gr.Row():
                            msg = gr.Textbox(placeholder="Ask about any AI topic, e.g. 'latest work on speculative decoding'",
                                             show_label=False, scale=5, submit_btn=True)
                            clear = gr.Button("Clear", scale=1)
                        with gr.Accordion("Filters", open=False):
                            a_sources = gr.Dropdown(source_names, multiselect=True, label="Sources (empty = all)")
                            with gr.Row():
                                a_type = gr.Radio(TYPE_CHOICES, value="any", label="Type")
                                a_window = gr.Dropdown(list(DATE_WINDOWS), value="Any time", label="Published")
                        gr.Examples(EXAMPLES, inputs=msg)
                    with gr.Column(scale=2):
                        refs = gr.Markdown("_Ask a question to get ranked references._", label="Recommended reading")
                        with gr.Accordion("🧭 Agent trace", open=False):
                            trace = gr.Markdown()
                msg.submit(assistant_fn, [msg, chat, a_sources, a_type, a_window], [chat, refs, trace, msg])
                clear.click(lambda: ([], "_Ask a question to get ranked references._", "", ""), None,
                            [chat, refs, trace, msg], api_visibility="private")

            with gr.Tab("🔎 Keyword Search"):
                with gr.Row():
                    q = gr.Textbox(placeholder='Keywords, e.g. diffusion "video editing"', show_label=False, scale=5)
                    go = gr.Button("Search", variant="primary", scale=1)
                with gr.Row():
                    s_sources = gr.Dropdown(source_names, multiselect=True, label="Sources (empty = all)")
                    s_type = gr.Radio(TYPE_CHOICES, value="any", label="Type")
                    s_window = gr.Dropdown(list(DATE_WINDOWS), value="Any time", label="Published")
                    s_sort = gr.Radio(["relevance", "date", "upvotes"], value="relevance", label="Sort by")
                results = gr.HTML()
                inputs = [q, s_sources, s_type, s_window, s_sort]
                go.click(search, inputs, results)
                q.submit(search, inputs, results, api_visibility="private")

            with gr.Tab("📰 Latest"):
                with gr.Row():
                    l_sources = gr.Dropdown(source_names, multiselect=True, label="Sources (empty = all)")
                    l_type = gr.Radio(TYPE_CHOICES, value="any", label="Type")
                    l_n = gr.Slider(10, 200, value=50, step=10, label="How many")
                feed = gr.HTML()
                for comp in (l_sources, l_type, l_n):
                    comp.change(latest, [l_sources, l_type, l_n], feed, api_visibility="private")
                demo.load(latest, [l_sources, l_type, l_n], feed, api_name="latest_feed")

            with gr.Tab("⚙️ Status"):
                stat = gr.Markdown()
                if pull_mode:
                    gr.Markdown("New items are collected every 12 hours by a GitHub Action and saved to the "
                                "HF Dataset repo. This Space downloads the latest snapshot automatically.")
                btn = gr.Button("🔄 Sync now" if pull_mode else "🔄 Refresh now")
                out = gr.Markdown()
                btn.click(refresh_now, None, [out, stat])
                demo.load(status, None, stat, api_name="status")
    return demo
