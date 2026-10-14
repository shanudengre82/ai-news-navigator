"""Gradio Blocks UI: a reading desk for AI research.

Design: serif (Newsreader) for what you read, sans (Hanken Grotesk) for what you operate. The one
expressive element is the sieve, which shows how the collection narrows to the few items worth reading.
Teal means "kept", ochre means "filtered out"; no other accent colors.
"""

from __future__ import annotations

import html
import math
import re
from datetime import datetime, timezone
from typing import Callable

import gradio as gr

from ai_news.fetchers.common import parse_iso
from ai_news.rag.state import Candidate
from ai_news.schema import NewsItem
from ai_news.search import DATE_WINDOWS, highlight, keyword_search
from ai_news.store import CorpusStore

TYPE_CHOICES = [("All", "any"), ("Papers", "paper"), ("Blog posts", "blog")]
EXAMPLES = [
    "What are the latest approaches to scaling test-time compute for reasoning models?",
    "Which new open-weight multimodal models were released this month?",
    "Recent papers on making RAG more faithful with citations",
    "Efficient fine-tuning of LLMs: LoRA and its newer variants",
    "What's new in AI agents that use tools and browse the web?",
]
ASK_PLACEHOLDER = "What do you want to read about?"
ASK_EMPTY = ("Ask a question above. The assistant searches the collection, keeps only what is relevant, "
             "and gives you a short reading list with the reason for each pick.")

FONTS_HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
              '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
              '<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;'
              '0,6..72,500;0,6..72,600;1,6..72,400&display=swap" rel="stylesheet">')

CSS = """
:root {
  --paper:#F5F7F9; --ink:#1B2440; --graphite:#5B6478; --rule:#D7DCE5; --field:#FFFFFF;
  --kept:#0F7B6C; --dropped:#B7791F;
  --serif:"Newsreader", Georgia, "Times New Roman", serif;
}
.dark {
  --paper:#141B2D; --ink:#E6EAF2; --graphite:#9AA4B8; --rule:#2A3350; --field:#1B2338;
  --kept:#4FC7B4; --dropped:#E0A84A;
}
.gradio-container { width:100% !important; max-width:1180px !important; margin:0 auto !important;
  padding:0 16px !important; }
.gradio-container a { color:var(--ink); text-decoration-color:var(--rule); text-underline-offset:3px; }
.gradio-container a:hover { text-decoration-color:var(--kept); }
.gradio-container a:focus-visible, .gradio-container button:focus-visible {
  outline:2px solid var(--kept); outline-offset:2px; border-radius:2px; }

/* masthead */
.nv-masthead { padding:28px 0 6px; }
.nv-masthead h1 { font-family:var(--serif); font-weight:500; font-size:34px; line-height:1.15;
  letter-spacing:-0.01em; color:var(--ink); margin:0 0 6px; }
.nv-masthead p { font-size:15px; color:var(--graphite); margin:0; max-width:68ch; }

/* question box is the hero */
.nv-ask textarea { font-family:var(--serif) !important; font-size:21px !important; line-height:1.4 !important;
  padding:14px 16px !important; }
.nv-ask .submit-button { background:var(--kept) !important; color:#fff !important; font-weight:600;
  border-radius:6px !important; padding:0 20px !important; margin-left:10px; min-height:44px; align-self:center; }
.dark .nv-ask .submit-button { color:#0E1A2B !important; }
.nv-ask .submit-button:hover { filter:brightness(.92); }
.nv-filters { gap:12px !important; }
.nv-hint { color:var(--graphite); font-size:15px; max-width:68ch; }

/* the sieve */
.nv-sieve { margin:8px 0 4px; max-width:860px; }
.nv-step { display:grid; grid-template-columns:minmax(150px, 15rem) 1fr 4.5rem; align-items:center;
  gap:14px; padding:5px 0; font-size:15px; color:var(--ink); }
.nv-step .label { color:var(--graphite); }
.nv-step .count { text-align:right; font-variant-numeric:tabular-nums; font-weight:600; }
.nv-rail { position:relative; height:14px; }
.nv-rail .was { position:absolute; inset:0 auto 0 0; border:1.5px solid var(--dropped); border-radius:2px;
  opacity:.75; }
.nv-rail .kept { position:absolute; inset:0 auto 0 0; background:var(--kept); border-radius:2px; }
.nv-step.note .label { grid-column:1 / span 3; font-style:italic; font-family:var(--serif); font-size:17px; }
.nv-step.new .kept { animation:grow .6s cubic-bezier(.2,.7,.2,1) both; transform-origin:left; }
@keyframes grow { from { transform:scaleX(0); } to { transform:scaleX(1); } }
@media (prefers-reduced-motion: reduce) { .nv-step.new .kept { animation:none; } }
.nv-sieve details { margin-top:8px; font-size:13px; color:var(--graphite); }
.nv-sieve details summary { cursor:pointer; font-size:15px; }
.nv-sieve details ol { padding-left:20px; max-width:80ch; }
.nv-legend { font-size:13px; color:var(--graphite); margin-top:4px; }
.nv-legend i { display:inline-block; width:18px; height:8px; vertical-align:middle; margin:0 4px 0 0;
  border-radius:2px; }
.nv-legend .k { background:var(--kept); }
.nv-legend .d { border:1.5px solid var(--dropped); margin-left:12px; }

/* answer prose */
.nv-answer { font-family:var(--serif); font-size:17px; line-height:1.55; color:var(--ink); }
.nv-answer .nv-prose { max-width:68ch; }
.nv-answer .nv-prose p, .nv-answer .nv-prose li { font-family:var(--serif) !important; font-size:17px !important;
  line-height:1.55 !important; }
.nv-answer > .nv-hint, .nv-answer .nv-hint { font-family:var(--serif); font-size:17px !important; }
.nv-reset button { border:1px solid var(--rule) !important; padding:6px 14px !important; font-size:13px; }
.nv-answer h3 { font-family:var(--serif); font-weight:500; font-size:27px; line-height:1.2; margin:0 0 12px; }
.nv-answer h2, .nv-answer h4 { font-family:var(--serif); font-weight:600; font-size:21px; }
.nv-answer details { margin-top:24px; font-size:15px; color:var(--graphite); }
.nv-answer details summary { cursor:pointer; font-family:var(--body-font, inherit); }

/* reading list */
.nv-reading h2 { font-family:var(--serif); font-weight:500; font-size:21px; margin:0 0 10px; color:var(--ink); }
.nv-reading ol { list-style:none; counter-reset:r; padding:0; margin:0; }
.nv-reading li { counter-increment:r; display:grid; grid-template-columns:1.8rem 1fr; padding:12px 0;
  border-top:1px solid var(--rule); }
.nv-reading li::before { content:counter(r); font-family:var(--serif); font-size:21px; color:var(--kept);
  line-height:1.2; }
.nv-reading .t { font-family:var(--serif); font-size:17px; line-height:1.35; font-weight:500; }
.nv-item a.t, .nv-reading a.t { display:block; padding:0 !important; text-decoration:none; color:var(--ink);
  text-align:left; }
.nv-item, .nv-reading li, .nv-step { text-align:left; }
.nv-item a.t:hover, .nv-reading a.t:hover { text-decoration:underline; text-decoration-color:var(--kept); }
.nv-item .m a.pdf { margin-left:12px; padding:0 !important; }
.nv-reading .m, .nv-item .m { font-size:13px; color:var(--graphite); margin:4px 0 0; }
.nv-reading .why { font-size:15px; margin:6px 0 0; color:var(--ink); }
.nv-reading .empty, .nv-rows .empty { color:var(--graphite); font-size:15px; }

/* result rows (search, latest) */
.nv-rows { max-width:860px; }
.nv-rows .nv-count { font-size:15px; color:var(--graphite); margin:4px 0 8px; }
.nv-item { padding:14px 0; border-top:1px solid var(--rule); }
.nv-item .t { font-family:var(--serif); font-size:21px; line-height:1.3; font-weight:500; }
.nv-item .s { font-size:15px; line-height:1.55; color:var(--ink); margin:6px 0 0; max-width:75ch; }
.nv-item mark, .nv-reading mark { background:color-mix(in srgb, var(--kept) 14%, transparent); color:inherit;
  padding:0; border-radius:2px; }
.nv-kind { display:inline-block; width:.9em; text-align:center; margin-right:4px; color:var(--graphite); }

/* sources table */
.nv-sources table { border-collapse:collapse; font-size:15px; max-width:760px; width:100%; }
.nv-sources th { text-align:left; font-weight:600; color:var(--graphite); padding:6px 12px 6px 0; }
.nv-sources td { padding:7px 12px 7px 0; border-top:1px solid var(--rule); vertical-align:top; }
.nv-sources td.n { text-align:right; font-variant-numeric:tabular-nums; }
.nv-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; }
.nv-dot.ok { background:var(--kept); } .nv-dot.bad { background:var(--dropped); }

@media (max-width: 640px) {
  .nv-masthead h1 { font-size:27px; }
  .nv-ask textarea { font-size:17px !important; }
  .nv-step { grid-template-columns:1fr 3.5rem; gap:4px 10px; }
  .nv-step .nv-rail { grid-column:1 / span 2; grid-row:2; }
  .nv-item .t { font-size:17px; }
}
"""


def theme() -> gr.themes.Base:
    t = gr.themes.Base(
        font=[gr.themes.GoogleFont("Hanken Grotesk"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=["ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_sm,
    )
    return t.set(
        body_background_fill="#F5F7F9", body_background_fill_dark="#141B2D",
        body_text_color="#1B2440", body_text_color_dark="#E6EAF2",
        body_text_color_subdued="#5B6478", body_text_color_subdued_dark="#9AA4B8",
        background_fill_primary="#F5F7F9", background_fill_primary_dark="#141B2D",
        background_fill_secondary="#EDF0F4", background_fill_secondary_dark="#1B2338",
        block_background_fill="#F5F7F9", block_background_fill_dark="#141B2D",
        block_border_width="0px", block_shadow="none", block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        panel_background_fill="#F5F7F9", panel_background_fill_dark="#141B2D",
        border_color_primary="#D7DCE5", border_color_primary_dark="#2A3350",
        input_background_fill="#FFFFFF", input_background_fill_dark="#1B2338",
        input_border_color="#C9D0DC", input_border_color_dark="#334064",
        input_border_color_focus="#0F7B6C", input_border_color_focus_dark="#4FC7B4",
        color_accent="#0F7B6C", color_accent_soft="#DDEFEC", color_accent_soft_dark="#1E3A3A",
        link_text_color="#1B2440", link_text_color_dark="#E6EAF2",
        button_primary_background_fill="#0F7B6C", button_primary_background_fill_dark="#4FC7B4",
        button_primary_background_fill_hover="#0B6457", button_primary_background_fill_hover_dark="#3DB3A0",
        button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#0E1A2B",
        button_secondary_background_fill="transparent", button_secondary_background_fill_dark="transparent",
        button_secondary_border_color="#C9D0DC", button_secondary_border_color_dark="#334064",
        checkbox_label_background_fill_selected="#DDEFEC", checkbox_label_background_fill_selected_dark="#1E3A3A",
        shadow_drop="none", layout_gap="18px",
    )


# ---------------------------------------------------------------- small formatting helpers
def human_date(iso: str) -> str:
    try:
        return parse_iso(iso).strftime("%-d %b %Y")
    except Exception:
        return iso[:10]


def ago(iso: str | None) -> str:
    if not iso:
        return "not yet"
    mins = int((datetime.now(timezone.utc) - parse_iso(iso)).total_seconds() // 60)
    if mins < 2:
        return "just now"
    if mins < 90:
        return f"{mins} minutes ago"
    hours = round(mins / 60)
    return f"{hours} hours ago" if hours < 36 else f"{round(hours / 24)} days ago"


def _meta(source: str, source_type: str, published: str, upvotes: int = 0, extra: str = "") -> str:
    kind = ('<span class="nv-kind" title="Paper" aria-label="Paper">●</span>' if source_type == "paper"
            else '<span class="nv-kind" title="Blog post or news" aria-label="Blog post or news">◆</span>')
    parts = [html.escape(source), human_date(published)]
    if upvotes:
        parts.append(f"{upvotes} upvote{'s' if upvotes != 1 else ''}")
    return f'<p class="m">{kind}{", ".join(parts)}{extra}</p>'


def masthead_html(store: CorpusStore) -> str:
    s = store.stats()
    if not s["total"]:
        line = ("The collection is being gathered. This takes a few minutes after the app starts; "
                "Keyword search and the assistant will work once it is ready.")
    else:
        line = (f"{s['total']:,} papers and posts from arXiv, Hugging Face Daily Papers and AI lab blogs, "
                f"updated {ago(s['last_refresh'])}.")
    return f'<header class="nv-masthead"><h1>AI Research Navigator</h1><p>{line}</p></header>'


# ---------------------------------------------------------------- result rows
def row(it: NewsItem, query: str = "", chars: int = 380) -> str:
    summary = it.summary[:chars] + ("…" if len(it.summary) > chars else "")
    authors = ", ".join(it.authors[:3]) + (" and others" if len(it.authors) > 3 else "")
    extra = f", {html.escape(authors)}" if authors else ""
    if it.pdf_url:
        extra += f'<a class="pdf" href="{html.escape(it.pdf_url)}" target="_blank" rel="noopener">PDF</a>'

    return (f'<article class="nv-item"><a class="t" href="{html.escape(it.url)}" target="_blank" rel="noopener">'
            f'{highlight(it.title, query)}</a>{_meta(it.source, it.source_type, it.published, it.upvotes, extra)}'
            f'<p class="s">{highlight(summary, query)}</p></article>')


def rows(items: list[NewsItem], query: str = "", count_line: str = "") -> str:
    if not items:
        return ('<div class="nv-rows"><p class="empty">Nothing matches. Try fewer words, a longer date range, '
                'or clear the source filter.</p></div>')
    head = f'<p class="nv-count">{count_line}</p>' if count_line else ""
    return f'<div class="nv-rows">{head}{"".join(row(it, query) for it in items)}</div>'


# ---------------------------------------------------------------- assistant outputs
def reading_list_html(refs: list[Candidate]) -> str:
    if not refs:
        return ""
    items = []
    for c in refs:
        why = f'<p class="why">{html.escape(c["reason"])}</p>' if c.get("reason") else ""
        items.append(f'<li><div><a class="t" href="{html.escape(c["url"])}" target="_blank" rel="noopener">'
                     f'{html.escape(c["title"])}</a>'
                     f'{_meta(c["source"], c.get("source_type", "paper"), c["published"], c.get("upvotes", 0))}'
                     f'{why}</div></li>')
    return f'<section class="nv-reading"><h2>Read these first</h2><ol>{"".join(items)}</ol></section>'


# kept for callers that want markdown (e.g. MCP clients reading the API output)
def format_references(refs: list[Candidate]) -> str:
    return "\n\n".join(f"[{n}] [{c['title']}]({c['url']}) ({c['source']}, {human_date(c['published'])})"
                       + (f"\n{c['reason']}" if c.get("reason") else "") for n, c in enumerate(refs, 1))


STEP_LABEL = {
    "retrieve": "Matched your question",
    "rerank": "Best matches after reranking",
    "grade_documents": "Judged relevant by the model",
    "live_search": "Added from a live arXiv search",
    "generate": "Used in the answer",
}
STEP_NOTE = {
    "analyze_query": "Reading your question",
    "rewrite_query": "Too few were relevant, so the question was reworded and searched again",
}


def _bar(count: int, was: int | None, total: int) -> str:
    scale = math.log10(max(total, 10) + 1)

    def pct(n: int) -> float:
        return round(100 * math.log10(n + 1) / scale, 1) if n > 0 else 0.0

    was_el = f'<span class="was" style="width:{pct(was)}%"></span>' if was is not None and was > count else ""
    return f'<span class="nv-rail" aria-hidden="true">{was_el}<span class="kept" style="width:{pct(count)}%"></span></span>'


def sieve_html(total: int, steps: list[dict], trace: list[str] | None = None, working: bool = False) -> str:
    """The narrowing from the whole collection to the items used in the answer, one row per step."""
    if not steps and not working:
        return ""
    out = [f'<div class="nv-step"><span class="label">In the collection</span>{_bar(total, None, total)}'
           f'<span class="count">{total:,}</span></div>']
    prev = total
    for i, st in enumerate(steps):
        new = " new" if i == len(steps) - 1 else ""
        node, count = st["node"], st.get("count")
        if node in STEP_NOTE:
            if node == "analyze_query" and i != len(steps) - 1:
                continue  # only show "reading your question" while it is the current step
            out.append(f'<div class="nv-step note{new}"><span class="label">{STEP_NOTE[node]}</span></div>')
            continue
        if node == "live_search":
            out.append(f'<div class="nv-step{new}"><span class="label">{STEP_LABEL[node]}</span>'
                       f'{_bar(count, None, total)}<span class="count">+{count}</span></div>')
            prev = count
            continue
        out.append(f'<div class="nv-step{new}"><span class="label">{STEP_LABEL.get(node, node)}</span>'
                   f'{_bar(count or 0, prev, total)}<span class="count">{count or 0:,}</span></div>')
        prev = count or 0
    legend = ('<p class="nv-legend"><i class="k"></i>kept<i class="d"></i>filtered out at this step</p>'
              if len(steps) > 1 else "")
    detail = ""
    if trace:
        lines = "".join(f"<li>{re.sub(r'[*_]{1,2}([^*_]+)[*_]{1,2}', r'<strong>\1</strong>', html.escape(t))}</li>"
                        for t in trace)
        detail = f"<details><summary>How this answer was found</summary><ol>{lines}</ol></details>"
    return f'<section class="nv-sieve" aria-label="How the collection was narrowed">{"".join(out)}{legend}{detail}</section>'


def answer_md(question: str, body: str, earlier: list[tuple[str, str]]) -> str:
    md = f"<div class='nv-prose'>\n\n### {html.escape(question)}\n\n{body}\n\n</div>"
    if earlier:
        items = "\n\n".join(f"**{html.escape(q)}**\n\n{a}" for q, a in reversed(earlier))
        md += f"\n\n<details><summary>Earlier in this session ({len(earlier)})</summary>\n\n{items}\n\n</details>"
    return md


# ---------------------------------------------------------------- layout
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
        n = len(hits)
        line = "Showing the 50 best matches" if n == 50 else f"{n} result{'s' if n != 1 else ''}"
        if sort == "date" and n == 50:
            line = "Showing the 50 newest matches"
        return rows([h[0] for h in hits], query, line)

    def latest(sources: list[str], source_type: str, n: float) -> str:
        """The newest items in the collection, optionally filtered by source and type."""
        hits = keyword_search(store, "", sources or None, source_type, None, "date", int(n))
        return rows([h[0] for h in hits])

    def status() -> str:
        """Collection size, last update time and the health of each source."""
        s = store.stats()
        names = sorted(set(s["by_source"]) | set(s["source_status"]), key=lambda k: -s["by_source"].get(k, 0))
        trs = []
        for name in names:
            st = s["source_status"].get(name, "")
            ok = st.startswith("ok")
            note = "working" if ok else (html.escape(st.removeprefix("error: "))[:120] or "not fetched yet")
            trs.append(f'<tr><td><span class="nv-dot {"ok" if ok else "bad"}"></span>{html.escape(name)}</td>'
                       f'<td class="n">{s["by_source"].get(name, 0):,}</td><td>{note}</td></tr>')
        where = ("New items are collected every 12 hours by a GitHub Action and saved to a Hugging Face "
                 "dataset; this app downloads the latest copy every hour." if pull_mode else
                 "This app collects new items itself every few hours.")
        return (f'<div class="nv-sources"><p class="nv-hint">{s["total"]:,} items, updated {ago(s["last_refresh"])}. '
                f'{where}</p><table><thead><tr><th>Source</th><th style="text-align:right">Items</th>'
                f'<th>Last update</th></tr></thead><tbody>{"".join(trs)}</tbody></table></div>')

    def refresh_now() -> tuple[str, str, str]:
        """Update the collection now: fetch all sources (fetch mode, takes minutes) or download the latest
        copy from the Hugging Face dataset (pull mode)."""
        msg = refresh_fn()
        return f'<p class="nv-hint">{html.escape(msg)}</p>', status(), masthead_html(store)

    with gr.Blocks(title="AI Research Navigator") as demo:
        masthead = gr.HTML(masthead_html(store))
        with gr.Tabs():
            with gr.Tab("Ask"):
                msg = gr.Textbox(placeholder=ASK_PLACEHOLDER, show_label=False, submit_btn="Ask",
                                 lines=1, max_lines=4, elem_classes="nv-ask", autofocus=True)
                with gr.Row(elem_classes="nv-filters"):
                    a_sources = gr.Dropdown(source_names, multiselect=True, label="Sources", info="Empty means all")
                    a_type = gr.Radio(TYPE_CHOICES, value="any", label="Type")
                    a_window = gr.Dropdown(list(DATE_WINDOWS), value="Any time", label="Published")
                gr.Examples(EXAMPLES, inputs=msg, label="Or try one of these")
                sieve = gr.HTML()
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, min_width=320):
                        answer = gr.Markdown(f"<p class='nv-hint'>{ASK_EMPTY}</p>", elem_classes="nv-answer")
                    with gr.Column(scale=2, min_width=280):
                        refs = gr.HTML()
                with gr.Row():
                    start_over = gr.Button("Start a new conversation", size="sm", variant="secondary",
                                           scale=0, min_width=220, elem_classes="nv-reset")
                conv = gr.State([])
                # History lives in gr.State: it is hidden from the API/MCP schema, so callers only pass
                # (message, sources, source_type, window) and can't send a malformed chat history.
                msg.submit(assistant_fn, [msg, conv, a_sources, a_type, a_window], [answer, refs, sieve, msg, conv])
                start_over.click(lambda: (f"<p class='nv-hint'>{ASK_EMPTY}</p>", "", "", "", []), None,
                                 [answer, refs, sieve, msg, conv], api_visibility="private")

            with gr.Tab("Search"):
                q = gr.Textbox(placeholder='Keywords, e.g. diffusion "video editing"', show_label=False,
                               submit_btn="Search", elem_classes="nv-ask")
                with gr.Row(elem_classes="nv-filters"):
                    s_sources = gr.Dropdown(source_names, multiselect=True, label="Sources", info="Empty means all")
                    s_type = gr.Radio(TYPE_CHOICES, value="any", label="Type")
                    s_window = gr.Dropdown(list(DATE_WINDOWS), value="Any time", label="Published")
                    s_sort = gr.Radio([("Best match", "relevance"), ("Newest", "date"), ("Most upvoted", "upvotes")],
                                      value="relevance", label="Sort by")
                results = gr.HTML('<p class="nv-hint">Type keywords and press Enter. Put a phrase in quotes '
                                  'to match it exactly.</p>')
                inputs = [q, s_sources, s_type, s_window, s_sort]
                q.submit(search, inputs, results)
                for comp in (s_sources, s_type, s_window, s_sort):
                    comp.change(search, inputs, results, api_visibility="private")

            with gr.Tab("Latest"):
                with gr.Row(elem_classes="nv-filters"):
                    l_sources = gr.Dropdown(source_names, multiselect=True, label="Sources", info="Empty means all")
                    l_type = gr.Radio(TYPE_CHOICES, value="any", label="Type")
                    l_n = gr.Slider(10, 200, value=50, step=10, label="How many")
                feed = gr.HTML()
                for comp in (l_sources, l_type, l_n):
                    comp.change(latest, [l_sources, l_type, l_n], feed, api_visibility="private")

            with gr.Tab("Sources"):
                stat = gr.HTML()
                btn = gr.Button("Download the latest copy" if pull_mode else "Collect new items now",
                                variant="secondary", size="sm")
                out = gr.HTML()
                btn.click(refresh_now, None, [out, stat, masthead])

        demo.load(latest, [l_sources, l_type, l_n], feed, api_name="latest_feed")
        demo.load(status, None, stat, api_name="status")
        demo.load(lambda: masthead_html(store), None, masthead, api_visibility="private")
    return demo
