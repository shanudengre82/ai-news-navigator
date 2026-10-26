"""Prompts. Small open models follow JSON instructions well when shown the exact shape."""

ANALYZE = """You help researchers find AI papers and news. Analyze the user's request.
{history}
User request: {question}

Return ONLY a JSON object of this shape:
{{"intent": "<one sentence: what the user wants>",
  "search_queries": ["<1 to 3 short keyword queries, technical terms, no filler>"],
  "date_window_days": <integer days if the user asks for recent/latest/this week/month, else null>,
  "source_type": "paper" | "blog" | "any"}}"""

GRADE = """You are reviewing search results for a research request.

Request: {question}

Results:
{docs}

For EACH result decide if it is genuinely useful for the request (on-topic, not merely sharing a keyword).
For useful ones, write a digest: at most 30 words on what the work does and how it bears on the request.
Return ONLY a JSON list, one object per result, in order:
[{{"doc": 1, "relevant": true, "digest": "<max 30 words>"}}, {{"doc": 2, "relevant": false, "digest": ""}}, ...]"""

REWRITE = """The search for this request returned too few relevant results.

Request: {question}
Previous queries: {queries}

Write 2 different search queries using synonyms, broader terms or the specific technical names
researchers would use. Return ONLY JSON: {{"search_queries": ["...", "..."]}}"""

GENERATE = """You are an AI research assistant. Using ONLY the sources below, answer the request.

Request: {question}

Sources:
{docs}

Write 1 or 2 short paragraphs of plain prose (no headings, no lists) that summarize what these sources
say together about the request: the main approaches, how they differ, and any concrete models or findings.
Cite sources inline as [1], [2]. End with one final sentence that begins with "**Start with** [n]" and says
why that source is the best one to read first.
Do not invent sources, numbers, or links. If the sources only partly cover the request, say so."""
