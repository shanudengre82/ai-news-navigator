"""Systematic guardrails and PII protection for inputs, retrieved documents, and LLM outputs.

Designed to be deterministic, fast (<1 ms), and dependency-free so it fits Hugging Face Spaces
ZeroGPU constraints without extra model downloads. Never logs raw PII values — only entity types
and counts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

MAX_QUERY_CHARS = 2000
MAX_DRAFT_CHARS = 5000

# ------------------------------------------------------------------ PII patterns
# Ordered from most specific (secrets/emails) to broader numeric patterns so tokens
# are not partially matched by looser rules.
_API_KEY_RE = re.compile(
    r"\b(?:"
    r"hf_[A-Za-z0-9]{20,}"
    r"|sk-(?:proj-|ant-|live-|test-)?[A-Za-z0-9_-]{20,}"
    r"|(?:pk|sk)-lf-[A-Za-z0-9-]{16,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}"
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
    r")\b"
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SSN_RE = re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_CC_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_PHONE_RE = re.compile(
    r"(?:(?<!\w)(?:\+|00)\d{1,3}[\s.-]?(?:\(\d{1,4}\)[\s.-]?|\d{1,4}[\s.-]))"
    r"\d{2,4}[\s.-]\d{3,4}(?:[\s.-]\d{2,4})?\b"
    r"|(?<!\w)\(\d{3}\)[\s.-]?\d{3}[\s.-]\d{4}\b"
    r"|(?<!\w)\d{3}[\s.-]\d{3}[\s.-]\d{4}\b"
)
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
)

# ------------------------------------------------------------------ Injection & safety patterns
# Imperative attack patterns targeting the assistant itself; designed to avoid false positives
# on academic AI-safety queries like "papers on prompt injection defenses and jailbreak red teaming".
_CONTROL_TOKENS_RE = re.compile(
    r"(?:<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|assistant\|>|<\|user\|>|<<SYS>>|<</SYS>>|\[/?INST\])",
    re.I,
)
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    _CONTROL_TOKENS_RE,
    re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+)?"
        r"(?:previous|prior|above|earlier|preceding|your)\s+"
        r"(?:instructions|prompts|rules|guidelines|directives|constraints)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:reveal|show|print|dump|output|repeat|leak|disclose)\s+(?:me\s+)?"
        r"(?:your|the)\s+(?:system\s+prompt|hidden\s+instructions|initial\s+prompt|developer\s+message)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:you\s+are\s+now\s+(?:DAN|an?\s+unrestricted|no\s+longer\s+bound)"
        r"|pretend\s+you\s+have\s+no\s+(?:rules|restrictions|guardrails)"
        r"|act\s+as\s+an?\s+unfiltered)\b",
        re.I,
    ),
    re.compile(r"(?m)^\s*(?:system|developer)\s*:\s*(?:you\s+are|ignore|override|disregard)\b", re.I),
)

_UNSAFE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("self_harm", re.compile(
        r"\b(?:how\s+to\s+(?:commit\s+suicide|kill\s+myself|harm\s+myself)"
        r"|instructions\s+for\s+(?:suicide|self[- ]harm))\b",
        re.I,
    )),
    ("cyber_exploit", re.compile(
        r"\b(?:write|generate|give\s+me|create)\s+(?:(?:a|an)\s+)?(?:working\s+)?"
        r"(?:ransomware|keylogger|trojan|zero[- ]day\s+exploit|malware\s+payload)\b",
        re.I,
    )),
    ("csam", re.compile(r"\b(?:child\s+sexual\s+abuse\s+material|csam\s+(?:links|download))\b", re.I)),
)

_CITATION_RE = re.compile(r"\[(\d+)\]")
_URL_RE = re.compile(r"https?://[^\s\)\]>\"']+")


# ------------------------------------------------------------------ Data structures
@dataclass(frozen=True)
class PiiFinding:
    entity_type: str
    count: int

    def to_dict(self) -> dict:
        return {"entity_type": self.entity_type, "count": self.count}


@dataclass
class PiiResult:
    text: str
    findings: list[PiiFinding] = field(default_factory=list)

    @property
    def had_pii(self) -> bool:
        return bool(self.findings)

    @property
    def total_count(self) -> int:
        return sum(f.count for f in self.findings)

    @property
    def entity_types(self) -> list[str]:
        return [f.entity_type for f in self.findings]


@dataclass
class InputGuardrailResult:
    allowed: bool
    sanitized_text: str
    sanitized_history: list[str]
    pii: PiiResult
    blocked_reason: str | None = None
    flags: list[str] = field(default_factory=list)


@dataclass
class OutputGuardrailResult:
    text: str
    pii: PiiResult
    valid_citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    hallucinated_urls: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ Helpers
def _luhn_ok(digits: str) -> bool:
    """Validate a candidate payment-card number with the Luhn algorithm."""
    if not (13 <= len(digits) <= 19) or not digits.isdigit():
        return False
    if len(set(digits)) == 1:  # reject trivial repeated digits like 0000...
        return False
    total = 0
    for idx, ch in enumerate(reversed(digits)):
        n = ord(ch) - 48
        if idx % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def redact_pii(text: str) -> PiiResult:
    """Detect and redact PII/secrets in `text`. Returns redacted text and entity counts only."""
    if not text:
        return PiiResult(text=text or "")

    counts: dict[str, int] = {}
    out = text

    def _sub(entity: str, pattern: re.Pattern[str], source: str) -> str:
        n = 0

        def _repl(_m: re.Match[str]) -> str:
            nonlocal n
            n += 1
            return f"[REDACTED_{entity}]"

        res = pattern.sub(_repl, source)
        if n:
            counts[entity] = counts.get(entity, 0) + n
        return res

    out = _sub("API_KEY", _API_KEY_RE, out)
    out = _sub("EMAIL", _EMAIL_RE, out)
    out = _sub("SSN", _SSN_RE, out)

    # Credit cards require Luhn validation so arXiv/DOI/numeric strings do not false-positive.
    cc_hits = 0

    def _cc_repl(m: re.Match[str]) -> str:
        nonlocal cc_hits
        raw = m.group(0).strip()
        digits = re.sub(r"\D", "", raw)
        if _luhn_ok(digits):
            cc_hits += 1
            return "[REDACTED_CREDIT_CARD]"
        return m.group(0)

    out = _CC_CANDIDATE_RE.sub(_cc_repl, out)
    if cc_hits:
        counts["CREDIT_CARD"] = cc_hits

    out = _sub("PHONE", _PHONE_RE, out)
    out = _sub("IPV4", _IPV4_RE, out)

    findings = [PiiFinding(entity_type=k, count=v) for k, v in counts.items()]
    return PiiResult(text=out, findings=findings)


def detect_prompt_injection(text: str) -> bool:
    """Return True if `text` contains imperative prompt-injection or control-token patterns."""
    if not text:
        return False
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def detect_unsafe_content(text: str) -> str | None:
    """Return the matched safety category if `text` requests severe harm, else None."""
    if not text:
        return None
    for category, pat in _UNSAFE_PATTERNS:
        if pat.search(text):
            return category
    return None


def sanitize_document(text: str) -> str:
    """Neutralize indirect prompt-injection strings and control tokens in scraped documents."""
    if not text:
        return text or ""
    out = _CONTROL_TOKENS_RE.sub(" ", text)
    for pat in _INJECTION_PATTERNS[1:]:
        out = pat.sub("[filtered instruction]", out)
    return out


def check_input(
    text: str,
    history: list[str] | None = None,
    *,
    enabled: bool = True,
    pii_mode: str = "redact",
    max_chars: int = MAX_QUERY_CHARS,
) -> InputGuardrailResult:
    """Run systematic input guardrails on user text and optional conversation history."""
    hist = list(history or [])
    if not enabled:
        return InputGuardrailResult(
            allowed=True,
            sanitized_text=text,
            sanitized_history=hist,
            pii=PiiResult(text=text),
        )

    flags: list[str] = []
    if not text or not text.strip():
        return InputGuardrailResult(
            allowed=False,
            sanitized_text="",
            sanitized_history=hist,
            pii=PiiResult(text=""),
            blocked_reason="Please enter a non-empty question.",
            flags=["empty_input"],
        )

    if len(text) > max_chars:
        return InputGuardrailResult(
            allowed=False,
            sanitized_text=text[:max_chars],
            sanitized_history=hist,
            pii=PiiResult(text=text[:max_chars]),
            blocked_reason=(
                f"Your input is too long ({len(text):,} characters; limit is {max_chars:,}). "
                "Please shorten your question and try again."
            ),
            flags=["max_length_exceeded"],
        )

    # Always redact PII in both the current text and conversation history before anything else.
    pii_main = redact_pii(text)
    merged_counts: dict[str, int] = {f.entity_type: f.count for f in pii_main.findings}
    sanitized_hist: list[str] = []
    for h in hist:
        hp = redact_pii(str(h))
        sanitized_hist.append(hp.text)
        for f in hp.findings:
            merged_counts[f.entity_type] = merged_counts.get(f.entity_type, 0) + f.count

    combined_pii = PiiResult(
        text=pii_main.text,
        findings=[PiiFinding(k, v) for k, v in merged_counts.items()],
    )
    if combined_pii.had_pii:
        flags.append("pii_detected")
        if pii_mode == "block":
            types = ", ".join(combined_pii.entity_types)
            return InputGuardrailResult(
                allowed=False,
                sanitized_text=pii_main.text,
                sanitized_history=sanitized_hist,
                pii=combined_pii,
                blocked_reason=(
                    f"Personal or sensitive information ({types}) was detected in your request. "
                    "Please remove it and ask your AI research question again."
                ),
                flags=flags + ["pii_blocked"],
            )
        flags.append("pii_redacted")

    if detect_prompt_injection(pii_main.text):
        return InputGuardrailResult(
            allowed=False,
            sanitized_text=pii_main.text,
            sanitized_history=sanitized_hist,
            pii=combined_pii,
            blocked_reason=(
                "Your request looks like an attempt to override the assistant's instructions. "
                "Please ask a direct question about AI papers or news."
            ),
            flags=flags + ["prompt_injection"],
        )

    unsafe_cat = detect_unsafe_content(pii_main.text)
    if unsafe_cat:
        return InputGuardrailResult(
            allowed=False,
            sanitized_text=pii_main.text,
            sanitized_history=sanitized_hist,
            pii=combined_pii,
            blocked_reason=(
                "I cannot help with that request. I am an AI research assistant that searches and "
                "summarizes published AI papers and news."
            ),
            flags=flags + [f"unsafe:{unsafe_cat}"],
        )

    return InputGuardrailResult(
        allowed=True,
        sanitized_text=pii_main.text,
        sanitized_history=sanitized_hist,
        pii=combined_pii,
        flags=flags,
    )


def _norm_url(url: str) -> str:
    return url.rstrip(".,;:!?")


def check_output(
    text: str,
    *,
    max_citation: int | None = None,
    allowed_urls: Iterable[str] | None = None,
    enabled: bool = True,
) -> OutputGuardrailResult:
    """Validate citations, verify URLs against retrieved references, and redact any output PII."""
    if not enabled or not text:
        return OutputGuardrailResult(text=text or "", pii=PiiResult(text=text or ""))

    flags: list[str] = []
    out = text

    # 1. Citation grounding: strip out-of-range [n] markers
    valid_cites: list[int] = []
    invalid_cites: list[int] = []
    if max_citation is not None:
        def _cite_repl(m: re.Match[str]) -> str:
            n = int(m.group(1))
            if 1 <= n <= max_citation:
                valid_cites.append(n)
                return m.group(0)
            invalid_cites.append(n)
            return ""

        out = _CITATION_RE.sub(_cite_repl, out)
        if invalid_cites:
            out = re.sub(r" {2,}", " ", out)
            flags.append("invalid_citations_stripped")
    else:
        valid_cites = [int(m.group(1)) for m in _CITATION_RE.finditer(out)]

    # 2. Hallucinated URL check (only when allowed_urls is explicitly provided)
    hallucinated_urls: list[str] = []
    if allowed_urls is not None:
        allowed_set = {_norm_url(u) for u in allowed_urls if u}

        def _url_repl(m: re.Match[str]) -> str:
            raw = m.group(0)
            norm = _norm_url(raw)
            if norm in allowed_set:
                return raw
            hallucinated_urls.append(norm)
            return "[unverified link removed]"

        out = _URL_RE.sub(_url_repl, out)
        if hallucinated_urls:
            flags.append("hallucinated_urls_removed")

    # 3. Output PII redaction
    pii = redact_pii(out)
    out = pii.text
    if pii.had_pii:
        flags.append("output_pii_redacted")

    return OutputGuardrailResult(
        text=out,
        pii=pii,
        valid_citations=valid_cites,
        invalid_citations=invalid_cites,
        hallucinated_urls=hallucinated_urls,
        flags=flags,
    )
