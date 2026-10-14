"""Prompts. Small open models follow JSON instructions well when shown the exact shape."""

ANALYZE = """You help researchers find AI papers and news. Analyze the user's request.
{history}
User request: {question}

Return ONLY a JSON object of this shape:
{{"intent": "<one sentence: what the user wants>",
  "search_queries": ["<1 to 3 short keyword queries, technical terms, no filler>"],
  "date_window_days": <integer days if the user asks for recent/latest/this week/month, else null>,
  "source_type": "paper" | "blog" | "any"}}"""

GRADE = """You are grading search results for relevance to a research request.

Request: {question}

Results:
{docs}

For EACH result decide if it is genuinely useful for the request (on-topic, not merely sharing a keyword).
Return ONLY a JSON list, one object per result, in order:
[{{"doc": 1, "relevant": true, "reason": "<max 20 words: why it helps>"}}, ...]"""

REWRITE = """The search for this request returned too few relevant results.

Request: {question}
Previous queries: {queries}

Write 2 different search queries using synonyms, broader terms or the specific technical names
researchers would use. Return ONLY JSON: {{"search_queries": ["...", "..."]}}"""

GENERATE = """You are an AI research assistant. Using ONLY the sources below, answer the request.

Request: {question}

Sources:
{docs}

Write in markdown:
1. A concise synthesis (2-4 short paragraphs) of what these sources say about the request,
   citing sources inline as [1], [2]. Mention concrete methods, models, or findings.
2. A line starting with "**Start with:**" naming the single best source to read first and why.
Do not invent sources, numbers, or links. If the sources only partly cover the request, say so."""
