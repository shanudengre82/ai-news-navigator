"""Compact prompts tuned for small open models (low token overhead, strict JSON shapes)."""

ANALYZE = """Analyze this AI research query.
{history}
Query: {question}
Return ONLY JSON:
{{"intent": "<1 short sentence>",
  "search_queries": ["<1-3 concise technical keyword queries>"],
  "date_window_days": <int days if time-bounded else null>,
  "source_type": "paper" | "blog" | "any"}}"""

GRADE = """Judge each result for the query. Keep only genuinely on-topic items; write a <=25-word digest for kept items.
Query: {question}

{docs}

Return ONLY a JSON list in order:
[{{"doc": 1, "relevant": true, "digest": "<=25 words"}}, {{"doc": 2, "relevant": false, "digest": ""}}]"""

REWRITE = """Too few relevant results. Write 2 alternative technical search queries (synonyms or broader terms).
Query: {question}
Previous: {queries}
Return ONLY JSON: {{"search_queries": ["...", "..."]}}"""

GENERATE = """Answer the AI research query using ONLY these sources.
Query: {question}

{docs}

Write 1-2 short paragraphs of plain prose (no headings/lists) summarizing key approaches, differences, and findings. Cite inline as [1], [2]. End with one sentence starting with "**Start with** [n]" explaining why. Never invent facts, numbers, or links."""

POLISH = """Rewrite this draft LinkedIn post on recent AI research clearly for practitioners.
Rules: keep every listed item's title, link, and order verbatim; use only facts in the draft; no hype, <=2 emoji, <=2,200 chars; keep hashtags on the last line; return only the post text.

Draft:
{draft}"""
