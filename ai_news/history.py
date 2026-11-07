"""The Ask tab's question history: kept in the visitor's own browser (gr.BrowserState), grouped by day, week
and month in their local time. Each entry stores the answer and a compact copy of its references, so a past
question comes back instantly without running the model again."""

from __future__ import annotations

import html
import time
import uuid
from datetime import date, datetime, timedelta, timezone

MAX_ENTRIES = 100       # oldest entries drop off after this many
MAX_REFS = 30           # references kept per entry
REF_SUMMARY_CHARS = 300  # these limits keep the stored history well under the browser's ~5 MB per site
REF_KEYS = ("id", "title", "url", "source", "source_type", "published", "upvotes", "relevance", "score", "digest",
            "weak")


def make_entry(question: str, answer: str, results: dict, now: float | None = None) -> dict:
    refs = (results or {}).get("refs", [])
    keep = []
    for c in refs[:MAX_REFS]:
        r = {k: c[k] for k in REF_KEYS if k in c}
        r["summary"] = (c.get("summary") or "")[:REF_SUMMARY_CHARS]
        keep.append(r)
    return {"id": uuid.uuid4().hex[:12], "ts": int(now if now is not None else time.time()),
            "q": " ".join(question.split()), "answer": answer, "summarized": (results or {}).get("summarized", 0),
            "total": len(refs), "refs": keep}


def add_entry(history: list | None, entry: dict) -> list:
    """Newest first; asking the same question again replaces its older entry."""
    key = entry["q"].lower()
    rest = [e for e in (history or []) if isinstance(e, dict) and e.get("q", "").lower() != key]
    return [entry] + rest[: MAX_ENTRIES - 1]


def local_dt(ts: float, tz_offset_min: int) -> datetime:
    """`tz_offset_min` is the browser's getTimezoneOffset(): minutes to add to local time to get UTC."""
    return datetime.fromtimestamp(ts, timezone.utc) - timedelta(minutes=tz_offset_min or 0)


def group_label(d: date, today: date) -> tuple[str, bool]:
    """The group a local date falls in, and whether that group starts open."""
    if d >= today:
        return "Today", True
    if d == today - timedelta(days=1):
        return "Yesterday", True
    monday = today - timedelta(days=today.weekday())
    if d >= monday:
        return d.strftime("%A"), True
    if d >= monday - timedelta(days=7):
        return "Last week", False
    if (d.year, d.month) == (today.year, today.month):
        return f"Earlier in {d.strftime('%B')}", False
    if d.year >= today.year - 1:
        return d.strftime("%B %Y"), False
    return str(d.year), False


def group_entries(history: list | None, now: float | None = None, tz_offset_min: int = 0) -> list[tuple]:
    """[(label, open, entries)] newest group first, entries newest first."""
    now = now if now is not None else time.time()
    today = local_dt(now, tz_offset_min).date()
    groups: dict[str, tuple[bool, list]] = {}
    for e in sorted((e for e in history or [] if isinstance(e, dict) and "ts" in e), key=lambda e: -e["ts"]):
        label, is_open = group_label(local_dt(e["ts"], tz_offset_min).date(), today)
        groups.setdefault(label, (is_open, []))[1].append(e)
    return [(label, is_open, entries) for label, (is_open, entries) in groups.items()]


def when(e: dict, label: str, tz_offset_min: int, this_year: int) -> str:
    """Time of day for today and yesterday, else the date (with the year when it is not this year)."""
    dt = local_dt(e["ts"], tz_offset_min)
    if label in ("Today", "Yesterday"):
        return dt.strftime("%H:%M")
    return dt.strftime("%-d %b") if dt.year == this_year else dt.strftime("%-d %b %Y")


HISTORY_EMPTY = '<p class="nv-hist-empty">Questions you ask appear here, saved in this browser only.</p>'


def history_html(history: list | None, now: float | None = None, tz_offset_min: int = 0,
                 active: str | None = None) -> str:
    now = now if now is not None else time.time()
    groups = group_entries(history, now, tz_offset_min)
    this_year = local_dt(now, tz_offset_min).year
    if not groups:
        return f'<nav class="nv-hist" aria-label="Your questions">{HISTORY_EMPTY}</nav>'
    parts = []
    for label, is_open, entries in groups:
        items = "".join(
            f'<li><button type="button" class="nv-hist-q{" on" if e.get("id") == active else ""}" '
            f'data-nv-entry="{html.escape(str(e.get("id", "")))}"'
            f'{" aria-current=true" if e.get("id") == active else ""}>'
            f'<span class="q">{html.escape(e.get("q", ""))}</span>'
            f'<span class="t">{when(e, label, tz_offset_min, this_year)}</span></button></li>' for e in entries)
        count = "" if is_open else f' <span class="n">{len(entries)}</span>'
        parts.append(f'<details{" open" if is_open else ""}><summary>{html.escape(label)}{count}</summary>'
                     f'<ul>{items}</ul></details>')
    return f'<nav class="nv-hist" aria-label="Your questions">{"".join(parts)}</nav>'


def find(history: list | None, entry_id: str) -> dict | None:
    return next((e for e in history or [] if isinstance(e, dict) and e.get("id") == entry_id), None)


def restore(entry: dict, tz_offset_min: int = 0) -> tuple[str, dict, list, str]:
    """(answer body, results state, conversation, note) for a saved entry."""
    refs = entry.get("refs", [])
    results = {"refs": refs, "summarized": min(entry.get("summarized", 0), len(refs))}
    conv = [{"role": "user", "content": entry.get("q", "")}, {"role": "assistant", "content": entry.get("answer", "")}]
    saved = local_dt(entry.get("ts", 0), tz_offset_min).strftime("%a %-d %b, %H:%M")
    total = entry.get("total", len(refs))
    shown = (f" Showing the top {len(refs)} of {total:,} references." if total > len(refs) else "")
    note = f"Saved answer from {saved}.{shown} Press Ask to run it again for fresh results."
    return entry.get("answer", ""), results, conv, note
