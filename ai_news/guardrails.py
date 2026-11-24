"""Custom Guardrails & PII Framework (inspired by Patronus AI's Technical, Ethical & Operational pillars).

Designed to be deterministic, fast (<2 ms), self-documenting, and dependency-free so it fits
Hugging Face Spaces ZeroGPU and local GPU constraints without extra model downloads.
Never logs raw PII values — only entity types, counts, and per-guardrail verdicts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from ai_news.text import tokenize

MAX_QUERY_CHARS = 2000
MAX_DRAFT_CHARS = 5000

# ------------------------------------------------------------------ Guardrail Registry & Descriptions
@dataclass(frozen=True)
class GuardrailSpec:
    """Declarative specification and human-readable description of a custom guardrail."""
    id: str
    name: str
    pillar: str          # "Technical" | "Ethical" | "Operational"
    stage: str           # "Input" | "Context" | "Output" | "Input & Output" | "Lifecycle"
    action: str          # "block" | "redact" | "sanitize" | "warn" | "audit"
    description: str
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "pillar": self.pillar,
            "stage": self.stage,
            "action": self.action,
            "description": self.description,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class GuardrailCheckResult:
    """Outcome of one guardrail rule on a specific prompt or output."""
    id: str
    name: str
    pillar: str
    stage: str
    status: str          # "passed" | "redacted" | "sanitized" | "warned" | "blocked" | "skipped"
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "pillar": self.pillar,
            "stage": self.stage,
            "status": self.status,
            "detail": self.detail,
        }


GUARDRAIL_CATALOG: tuple[GuardrailSpec, ...] = (
    # ---- Technical Pillar ----
    GuardrailSpec(
        id="input_length_format",
        name="Input Length & Format",
        pillar="Technical",
        stage="Input",
        action="block",
        description=(
            "Rejects empty prompts, enforces maximum character limits (2,000 chars for Ask, 5,000 for Post), "
            "and strips invisible zero-width or null-byte obfuscation characters before downstream checks."
        ),
    ),
    GuardrailSpec(
        id="pii_redaction",
        name="PII & Secrets Redaction",
        pillar="Technical",
        stage="Input & Output",
        action="redact",
        description=(
            "Detects and masks personally identifiable information and secrets — emails, phone numbers, "
            "SSNs, Luhn-validated payment cards, IPv4 addresses, JWTs, and API keys (OpenAI, Anthropic, "
            "Hugging Face, AWS, GitHub, Langfuse) — in prompts, chat history, outputs, and traces."
        ),
    ),
    GuardrailSpec(
        id="prompt_injection",
        name="Prompt Injection & Jailbreak",
        pillar="Technical",
        stage="Input",
        action="block",
        description=(
            "Blocks imperative instruction overrides, system-prompt extraction, DAN/unfiltered jailbreaks, "
            "role-spoofing prefixes, and LLM chat control tokens (<|im_start|>, <<SYS>>, [INST]) while "
            "allowing legitimate AI-safety and red-teaming research questions."
        ),
    ),
    GuardrailSpec(
        id="topic_scope",
        name="Topic & Domain Boundary",
        pillar="Technical",
        stage="Input",
        action="block",
        description=(
            "Keeps the assistant focused on AI/ML research, papers, engineering, and tech news. Politely "
            "declines clearly off-domain personal requests (recipes, horoscopes, sports bets, travel plans) "
            "without running search or LLM inference, while permitting interdisciplinary AI queries."
        ),
    ),
    GuardrailSpec(
        id="indirect_injection",
        name="Indirect Injection Sanitization",
        pillar="Technical",
        stage="Context",
        action="sanitize",
        description=(
            "Neutralizes chat control tokens and embedded instruction-override strings inside scraped "
            "arXiv abstracts and blog posts before they enter LLM grading or generation prompts, counting "
            "every sanitized document in the trace."
        ),
    ),
    GuardrailSpec(
        id="hallucination_grounding",
        name="Hallucination & Citation Grounding",
        pillar="Technical",
        stage="Output",
        action="sanitize",
        description=(
            "Strips out-of-range [n] citations, removes unverified/hallucinated URLs not present in the "
            "retrieved references, and measures lexical grounding overlap against cited sources to flag "
            "low-grounding answers."
        ),
    ),
    GuardrailSpec(
        id="output_format_leak",
        name="Output Format & Prompt Leak Check",
        pillar="Technical",
        stage="Output",
        action="sanitize",
        description=(
            "Ensures generated answers do not leak raw LLM control tokens, internal system-prompt "
            "markers, or empty/malformed responses before returning text to the user."
        ),
    ),
    # ---- Ethical Pillar ----
    GuardrailSpec(
        id="credential_solicitation",
        name="Credential & PII Solicitation",
        pillar="Ethical",
        stage="Input",
        action="block",
        description=(
            "Politely refuses attempts to obtain passwords, API/SSH/private keys, tokens, .env secrets, "
            "or personal data (SSNs, credit cards, home addresses, phone numbers, user emails, doxxing) "
            "immediately at input with zero search or LLM execution."
        ),
    ),
    GuardrailSpec(
        id="severe_harm",
        name="Severe Harm Prevention",
        pillar="Ethical",
        stage="Input",
        action="block",
        description=(
            "Blocks requests soliciting self-harm instructions, working cyber-exploit or malware payloads "
            "(ransomware, keyloggers, zero-day exploits), or CSAM."
        ),
    ),
    GuardrailSpec(
        id="toxicity_moderation",
        name="Toxicity & Hate Speech Moderation",
        pillar="Ethical",
        stage="Input & Output",
        action="block",
        description=(
            "Screens user prompts and generated responses for hate speech, slurs, harassment, and explicit "
            "violent threats while allowing academic questions about toxicity benchmarks and datasets."
        ),
    ),
    GuardrailSpec(
        id="bias_fairness",
        name="Bias & Stereotyping Check",
        pillar="Ethical",
        stage="Output",
        action="warn",
        description=(
            "Flags overt discriminatory generalizations or demographic stereotyping in generated responses "
            "to uphold ethical neutrality and fairness."
        ),
    ),
    # ---- Operational Pillar ----
    GuardrailSpec(
        id="regulatory_compliance",
        name="Regulated Advice Compliance",
        pillar="Operational",
        stage="Input & Output",
        action="block",
        description=(
            "Prevents the research assistant from providing personal medical diagnoses/prescriptions, "
            "legal representation, or personalized financial/investment advice while allowing AI research "
            "queries in healthcare, law, and finance."
        ),
    ),
    GuardrailSpec(
        id="audit_tracing",
        name="Per-Prompt Audit Tracing",
        pillar="Operational",
        stage="Lifecycle",
        action="audit",
        description=(
            "Records a PII-redacted JSONL audit trail (and optional Langfuse trace) for every Ask, "
            "cached Ask, and Post rewrite, capturing per-guardrail verdicts, latency, spans, and scores."
        ),
    ),
)

_SPEC_BY_ID: dict[str, GuardrailSpec] = {s.id: s for s in GUARDRAIL_CATALOG}
CustomInputRule = Callable[[str], tuple[bool, str | None]]
_CUSTOM_SPECS: dict[str, GuardrailSpec] = {}
_CUSTOM_INPUT_RULES: dict[str, CustomInputRule] = {}


def get_guardrail_catalog() -> list[GuardrailSpec]:
    """Return all built-in and registered custom guardrails with their descriptions."""
    return list(GUARDRAIL_CATALOG) + list(_CUSTOM_SPECS.values())


def register_custom_guardrail(spec: GuardrailSpec, input_fn: CustomInputRule | None = None) -> None:
    """Register a custom guardrail spec (and optional input check function `(text) -> (allowed, reason)`)."""
    _CUSTOM_SPECS[spec.id] = spec
    if input_fn is not None:
        _CUSTOM_INPUT_RULES[spec.id] = input_fn


def unregister_custom_guardrail(rule_id: str) -> None:
    _CUSTOM_SPECS.pop(rule_id, None)
    _CUSTOM_INPUT_RULES.pop(rule_id, None)


def _check(rule_id: str, status: str, detail: str = "") -> GuardrailCheckResult:
    spec = _SPEC_BY_ID.get(rule_id) or _CUSTOM_SPECS.get(rule_id)
    if spec is None:
        return GuardrailCheckResult(id=rule_id, name=rule_id, pillar="Technical", stage="Input",
                                    status=status, detail=detail)
    return GuardrailCheckResult(
        id=spec.id, name=spec.name, pillar=spec.pillar, stage=spec.stage, status=status, detail=detail
    )


# ------------------------------------------------------------------ Obfuscation & PII patterns
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\x00]")

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

# ------------------------------------------------------------------ Injection, safety & policy patterns
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

_EXTRACT_VERB = (
    r"(?:give|show|tell|list|dump|print|output|reveal|leak|share|send|extract|expose|export|"
    r"fetch|retrieve|provide|hand\s+over|get|find|look\s*up|steal|crack|obtain)"
    r"(?:\s+(?:me|us))?"
    r"|(?:what|where)\s+(?:is|are)"
    r"|can\s+i\s+(?:have|get|see)"
)
_POSSESSIVE_SCOPE = (
    r"(?:(?:all|any|every)\s+)?(?:(?:of\s+)?(?:your|you|the|my|our|their|user'?s?|users'?|"
    r"admin'?s?|root|server|system|app|saved|stored|private|secret|hidden|internal|env|environment)\s+){0,3}"
)
_CREDENTIAL_OR_PII_REQUEST_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("credential_or_pii_request", re.compile(
        rf"\b(?:{_EXTRACT_VERB})\s+{_POSSESSIVE_SCOPE}"
        r"(?:passwords?|passcodes?|passphrases?|passwd|credentials?|logins?|"
        r"(?:api|access|private|secret|ssh|encryption|signing|master|admin|root|auth)\s*[-_]?\s*keys?|"
        r"(?:auth|access|bearer|api|secret|refresh|session)\s*[-_]?\s*tokens?|"
        r"hf_token|openai_api_key|langfuse_secret_key|aws_secret_access_key|"
        r"env(?:ironment)?\s+var(?:iable)?s?|\.env\b)",
        re.I,
    )),
    ("credential_or_pii_request", re.compile(
        rf"\b(?:{_EXTRACT_VERB})\s+(?:all\s+)?(?:of\s+)?"
        r"(?:your|you|the|server|system|app|saved|stored|private|secret|admin|root)\s+"
        r"(?:keys|secrets)\b(?!\s*(?:sharing|findings?|takeaways?|ideas?|papers?|concepts?|value|values))",
        re.I,
    )),
    ("credential_or_pii_request", re.compile(
        rf"\b(?:{_EXTRACT_VERB})\s+(?:all\s+)?(?:of\s+)?"
        r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?'s\s+|someone'?s\s+|any(?:one|body)'?s\s+|"
        r"users?'?\s+|their\s+|your\s+|you\s+|private\s+|personal\s+|the\s+)?"
        r"(?:social\s+security\s+numbers?|ssns?|credit\s+card\s+numbers?|cvvs?|"
        r"bank\s+accounts?|home\s+address(?:es)?|phone\s+numbers?|email\s+address(?:es)?|"
        r"user\s+emails?|personal\s+(?:data|info(?:rmation)?))\b",
        re.I,
    )),
    ("credential_or_pii_request", re.compile(r"\b(?:doxx?(?:ing)?\s+(?:him|her|them|someone|[A-Z][a-z]+))\b", re.I)),
)

# Academic/AI domain cues: if present, topic_scope and regulated_advice will not block research questions.
_AI_RESEARCH_CUE_RE = re.compile(
    r"\b(?:ai|ml|llm|llms|model|models|neural|paper|papers|arxiv|transformer|transformers|rag|"
    r"agent|agents|benchmark|benchmarks|dataset|datasets|algorithm|algorithms|diffusion|lora|"
    r"fine[- ]tun\w*|embedding|embeddings|rerank\w*|inference|training|pretrain\w*|multimodal|"
    r"vision|speech|robot\w*|reinforcement|alignment|interpretability|hallucination|guardrail\w*|"
    r"deep\s+learning|machine\s+learning|natural\s+language|computer\s+vision|research|study|survey)\b",
    re.I,
)

# Toxicity / hate speech / explicit threats (avoids matching academic discussions of toxicity datasets)
_TOXICITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:go\s+kill\s+yourself|kys|i\s+will\s+(?:kill|murder|hurt|shoot)\s+you)\b", re.I),
    re.compile(
        r"\b(?:how\s+to\s+(?:make|build)\s+(?:a\s+)?(?:bomb|explosive|poison\s+to\s+kill))\b",
        re.I,
    ),
    re.compile(
        r"\b(?:all\s+(?:immigrants|women|men|muslims|jews|christians|black\s+people|white\s+people|asians)\s+"
        r"(?:are\s+(?:inferior|evil|criminals|trash|vermin|subhuman)|should\s+(?:die|be\s+killed|be\s+banned)))\b",
        re.I,
    ),
    re.compile(r"\b(?:you\s+are\s+(?:a\s+)?(?:worthless|pathetic|brainless)\s+(?:idiot|moron|trash|piece\s+of\s+shit))\b", re.I),
)

# Clearly off-domain personal requests (only triggers when no AI/ML research cues are present)
_OFF_TOPIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:give\s+me|write|share|what\s+is)\s+(?:a\s+)?(?:\w+\s+){0,2}(?:recipe|cooking\s+instructions)\b", re.I),
    re.compile(r"\b(?:my\s+horoscope|zodiac\s+sign\s+prediction|astrology\s+reading\s+for\s+today)\b", re.I),
    re.compile(r"\b(?:plan\s+(?:my|a)\s+(?:vacation|holiday|travel\s+itinerary|trip\s+to))\b", re.I),
    re.compile(r"\b(?:who\s+won\s+(?:last\s+night'?s|the)\s+(?:game|match|super\s+bowl|world\s+cup)|sports\s+betting\s+picks)\b", re.I),
)

# Regulated high-stakes personal advice (medical diagnosis/prescription, legal counsel, investment/tax picks)
_REGULATED_ADVICE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("medical_advice", re.compile(
        r"\b(?:diagnose\s+(?:my|this)|prescribe\s+(?:me|medication|drugs)|"
        r"what\s+(?:medication|dosage|pills?)\s+should\s+i\s+take\s+for\s+my|"
        r"should\s+i\s+stop\s+taking\s+my\s+(?:medication|prescription))\b",
        re.I,
    )),
    ("legal_advice", re.compile(
        r"\b(?:give\s+me\s+legal\s+advice\s+for\s+my|act\s+as\s+my\s+lawyer|"
        r"draft\s+(?:a\s+)?(?:lawsuit|legal\s+contract|will)\s+for\s+me)\b",
        re.I,
    )),
    ("financial_advice", re.compile(
        r"\b(?:which\s+(?:stocks?|crypto(?:currency)?|etfs?)\s+should\s+i\s+(?:buy|sell|invest\s+in)|"
        r"give\s+me\s+(?:personalized\s+)?(?:investment|trading|tax)\s+advice)\b",
        re.I,
    )),
)

# Output bias / stereotyping warning patterns
_BIAS_OUTPUT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:(?:women|men|immigrants|older\s+people|young\s+people)\s+are\s+"
        r"(?:naturally|inherently|always)\s+(?:worse|inferior|incapable|less\s+intelligent|unfit))\b",
        re.I,
    ),
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
    checks: list[GuardrailCheckResult] = field(default_factory=list)


@dataclass
class OutputGuardrailResult:
    text: str
    pii: PiiResult
    valid_citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    hallucinated_urls: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    grounding_score: float | None = None
    checks: list[GuardrailCheckResult] = field(default_factory=list)


# ------------------------------------------------------------------ Helpers & Detectors
def strip_obfuscation(text: str) -> tuple[str, int]:
    """Strip zero-width and null-byte obfuscation characters; return (cleaned, removed_count)."""
    if not text:
        return text or "", 0
    return _ZERO_WIDTH_RE.subn("", text)


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


def detect_credential_or_pii_request(text: str) -> str | None:
    """Return a flag name if `text` asks to reveal/extract passwords, keys, secrets, or personal data."""
    if not text:
        return None
    for category, pat in _CREDENTIAL_OR_PII_REQUEST_PATTERNS:
        if pat.search(text):
            return category
    return None


def detect_toxicity(text: str) -> bool:
    """Return True if `text` contains overt hate speech, harassment, or violent threats."""
    if not text:
        return False
    return any(p.search(text) for p in _TOXICITY_PATTERNS)


def detect_off_topic(text: str) -> bool:
    """Return True if `text` is clearly an off-domain personal request unrelated to AI/ML research."""
    if not text or _AI_RESEARCH_CUE_RE.search(text):
        return False
    return any(p.search(text) for p in _OFF_TOPIC_PATTERNS)


def detect_regulated_advice(text: str) -> str | None:
    """Return regulated domain category if `text` asks for personal medical/legal/financial advice."""
    if not text or _AI_RESEARCH_CUE_RE.search(text):
        return None
    for category, pat in _REGULATED_ADVICE_PATTERNS:
        if pat.search(text):
            return category
    return None


def detect_bias_output(text: str) -> bool:
    """Return True if generated output contains overt discriminatory generalizations."""
    if not text:
        return False
    return any(p.search(text) for p in _BIAS_OUTPUT_PATTERNS)


def sanitize_document_with_stats(text: str) -> tuple[str, int]:
    """Neutralize indirect prompt-injection strings and control tokens; return (cleaned, hit_count)."""
    if not text:
        return text or "", 0
    out, hits = _CONTROL_TOKENS_RE.subn(" ", text)
    for pat in _INJECTION_PATTERNS[1:]:
        out, n = pat.subn("[filtered instruction]", out)
        hits += n
    return out, hits


def sanitize_document(text: str) -> str:
    """Neutralize indirect prompt-injection strings and control tokens in scraped documents."""
    return sanitize_document_with_stats(text)[0]


# ------------------------------------------------------------------ Input Guardrails
def check_input(
    text: str,
    history: list[str] | None = None,
    *,
    enabled: bool = True,
    pii_mode: str = "redact",
    max_chars: int = MAX_QUERY_CHARS,
    check_topic: bool = True,
) -> InputGuardrailResult:
    """Run all Technical, Ethical, and Operational input guardrails on user text and history."""
    hist = list(history or [])
    if not enabled:
        return InputGuardrailResult(
            allowed=True,
            sanitized_text=text,
            sanitized_history=hist,
            pii=PiiResult(text=text),
        )

    flags: list[str] = []
    checks: list[GuardrailCheckResult] = []

    # 1. Input Length & Format (Technical)
    clean_text, obf_removed = strip_obfuscation(text or "")
    if obf_removed:
        flags.append("obfuscation_stripped")
    if not clean_text or not clean_text.strip():
        checks.append(_check("input_length_format", "blocked", "empty input"))
        return InputGuardrailResult(
            allowed=False,
            sanitized_text="",
            sanitized_history=hist,
            pii=PiiResult(text=""),
            blocked_reason="Please enter a non-empty question.",
            flags=flags + ["empty_input"],
            checks=checks,
        )

    if len(clean_text) > max_chars:
        checks.append(_check("input_length_format", "blocked", f"{len(clean_text):,} > {max_chars:,} chars"))
        return InputGuardrailResult(
            allowed=False,
            sanitized_text=clean_text[:max_chars],
            sanitized_history=hist,
            pii=PiiResult(text=clean_text[:max_chars]),
            blocked_reason=(
                f"Your input is too long ({len(clean_text):,} characters; limit is {max_chars:,}). "
                "Please shorten your question and try again."
            ),
            flags=flags + ["max_length_exceeded"],
            checks=checks,
        )
    checks.append(_check(
        "input_length_format",
        "sanitized" if obf_removed else "passed",
        f"stripped {obf_removed} zero-width chars" if obf_removed else "ok",
    ))

    # 2. PII & Secrets Redaction (Technical)
    pii_main = redact_pii(clean_text)
    merged_counts: dict[str, int] = {f.entity_type: f.count for f in pii_main.findings}
    sanitized_hist: list[str] = []
    for h in hist:
        hp = redact_pii(strip_obfuscation(str(h))[0])
        sanitized_hist.append(hp.text)
        for f in hp.findings:
            merged_counts[f.entity_type] = merged_counts.get(f.entity_type, 0) + f.count

    combined_pii = PiiResult(
        text=pii_main.text,
        findings=[PiiFinding(k, v) for k, v in merged_counts.items()],
    )
    if combined_pii.had_pii:
        flags.append("pii_detected")
        types = ", ".join(combined_pii.entity_types)
        if pii_mode == "block":
            checks.append(_check("pii_redaction", "blocked", f"blocked PII: {types}"))
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
                checks=checks,
            )
        flags.append("pii_redacted")
        checks.append(_check("pii_redaction", "redacted", f"masked: {types}"))
    else:
        checks.append(_check("pii_redaction", "passed", "no PII detected"))

    # 3. Prompt Injection & Jailbreak (Technical)
    if detect_prompt_injection(pii_main.text):
        checks.append(_check("prompt_injection", "blocked", "instruction override or control token"))
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
            checks=checks,
        )
    checks.append(_check("prompt_injection", "passed", "ok"))

    # 4. Credential & PII Solicitation (Ethical)
    solicitation = detect_credential_or_pii_request(pii_main.text)
    if solicitation:
        checks.append(_check("credential_solicitation", "blocked", "credential or personal data solicitation"))
        return InputGuardrailResult(
            allowed=False,
            sanitized_text=pii_main.text,
            sanitized_history=sanitized_hist,
            pii=combined_pii,
            blocked_reason=(
                "I'm sorry, I can't help with requests for passwords, API keys, secrets, "
                "or personal/private information. I'm an AI research assistant — feel free to "
                "ask me about published AI papers, benchmarks, or news instead."
            ),
            flags=flags + [solicitation],
            checks=checks,
        )
    checks.append(_check("credential_solicitation", "passed", "ok"))

    # 5. Severe Harm Prevention (Ethical)
    unsafe_cat = detect_unsafe_content(pii_main.text)
    if unsafe_cat:
        checks.append(_check("severe_harm", "blocked", unsafe_cat))
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
            checks=checks,
        )
    checks.append(_check("severe_harm", "passed", "ok"))

    # 6. Toxicity & Hate Speech Moderation (Ethical)
    if detect_toxicity(pii_main.text):
        checks.append(_check("toxicity_moderation", "blocked", "abusive or threatening content"))
        return InputGuardrailResult(
            allowed=False,
            sanitized_text=pii_main.text,
            sanitized_history=sanitized_hist,
            pii=combined_pii,
            blocked_reason=(
                "I'm sorry, I can't respond to abusive, hateful, or threatening requests. "
                "Please rephrase your question respectfully around AI papers or news."
            ),
            flags=flags + ["toxicity_blocked"],
            checks=checks,
        )
    checks.append(_check("toxicity_moderation", "passed", "ok"))

    # 7. Regulated Advice Compliance (Operational)
    reg_cat = detect_regulated_advice(pii_main.text)
    if reg_cat:
        checks.append(_check("regulatory_compliance", "blocked", reg_cat))
        return InputGuardrailResult(
            allowed=False,
            sanitized_text=pii_main.text,
            sanitized_history=sanitized_hist,
            pii=combined_pii,
            blocked_reason=(
                "I'm sorry, I can't provide personal medical, legal, or financial advice. "
                "I'm an AI research assistant — I can help you find published AI research papers on "
                "healthcare, law, or finance instead."
            ),
            flags=flags + [f"regulated:{reg_cat}"],
            checks=checks,
        )
    checks.append(_check("regulatory_compliance", "passed", "ok"))

    # 8. Topic & Domain Boundary (Technical)
    if check_topic and detect_off_topic(pii_main.text):
        checks.append(_check("topic_scope", "blocked", "outside AI/ML research domain"))
        return InputGuardrailResult(
            allowed=False,
            sanitized_text=pii_main.text,
            sanitized_history=sanitized_hist,
            pii=combined_pii,
            blocked_reason=(
                "I'm sorry, that falls outside my scope. I'm an AI research assistant focused on "
                "AI/ML papers, models, benchmarks, and tech news — ask me about a recent AI topic instead."
            ),
            flags=flags + ["off_topic"],
            checks=checks,
        )
    checks.append(_check("topic_scope", "passed" if check_topic else "skipped", "in domain" if check_topic else "skipped"))

    # 9. Custom registered input rules (extensible policy hooks)
    for rule_id, fn in list(_CUSTOM_INPUT_RULES.items()):
        spec = _CUSTOM_SPECS.get(rule_id)
        if spec and not spec.enabled:
            continue
        ok, reason = fn(pii_main.text)
        if not ok:
            checks.append(_check(rule_id, "blocked", reason or "custom policy"))
            return InputGuardrailResult(
                allowed=False,
                sanitized_text=pii_main.text,
                sanitized_history=sanitized_hist,
                pii=combined_pii,
                blocked_reason=reason or "Request blocked by custom guardrail policy.",
                flags=flags + [f"custom:{rule_id}"],
                checks=checks,
            )
        checks.append(_check(rule_id, "passed", "ok"))

    return InputGuardrailResult(
        allowed=True,
        sanitized_text=pii_main.text,
        sanitized_history=sanitized_hist,
        pii=combined_pii,
        flags=flags,
        checks=checks,
    )


def _norm_url(url: str) -> str:
    return url.rstrip(".,;:!?")


def _grounding_overlap(answer: str, source_texts: Iterable[str]) -> float | None:
    ans_tokens = set(tokenize(answer))
    if not ans_tokens:
        return None
    src_tokens: set[str] = set()
    for s in source_texts:
        src_tokens.update(tokenize(s))
    if not src_tokens:
        return None
    return round(len(ans_tokens & src_tokens) / len(ans_tokens), 4)


# ------------------------------------------------------------------ Output Guardrails
def check_output(
    text: str,
    *,
    max_citation: int | None = None,
    allowed_urls: Iterable[str] | None = None,
    source_texts: Iterable[str] | None = None,
    enabled: bool = True,
) -> OutputGuardrailResult:
    """Validate citations, URLs, grounding, format/leaks, toxicity, bias, and output PII."""
    if not enabled or not text:
        return OutputGuardrailResult(text=text or "", pii=PiiResult(text=text or ""))

    flags: list[str] = []
    checks: list[GuardrailCheckResult] = []
    out = text

    # 1. Output Format & Prompt Leak Check (Technical)
    out, leaked_tokens = _CONTROL_TOKENS_RE.subn("", out)
    if leaked_tokens:
        flags.append("control_tokens_stripped")
        checks.append(_check("output_format_leak", "sanitized", f"stripped {leaked_tokens} control tokens"))
    else:
        checks.append(_check("output_format_leak", "passed", "ok"))

    # 2. Hallucination & Citation Grounding (Technical)
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

    grounding = _grounding_overlap(out, source_texts) if source_texts is not None else None
    if grounding is not None and grounding < 0.2:
        flags.append("low_grounding_warning")

    if invalid_cites or hallucinated_urls:
        checks.append(_check(
            "hallucination_grounding",
            "sanitized",
            f"stripped {len(invalid_cites)} cites, {len(hallucinated_urls)} URLs",
        ))
    elif grounding is not None and grounding < 0.2:
        checks.append(_check("hallucination_grounding", "warned", f"low lexical overlap ({grounding:.0%})"))
    else:
        g_note = f"{len(valid_cites)} cites" + (f", {grounding:.0%} overlap" if grounding is not None else "")
        checks.append(_check("hallucination_grounding", "passed", g_note))

    # 3. Output PII Redaction (Technical)
    pii = redact_pii(out)
    out = pii.text
    if pii.had_pii:
        flags.append("output_pii_redacted")
        checks.append(_check("pii_redaction", "redacted", f"output masked: {', '.join(pii.entity_types)}"))
    else:
        checks.append(_check("pii_redaction", "passed", "no output PII"))

    # 4. Output Toxicity & Bias Checks (Ethical)
    if detect_toxicity(out):
        flags.append("output_toxicity_warning")
        checks.append(_check("toxicity_moderation", "warned", "toxic phrasing detected in output"))
    else:
        checks.append(_check("toxicity_moderation", "passed", "ok"))

    if detect_bias_output(out):
        flags.append("output_bias_warning")
        checks.append(_check("bias_fairness", "warned", "potential demographic generalization"))
    else:
        checks.append(_check("bias_fairness", "passed", "ok"))

    return OutputGuardrailResult(
        text=out,
        pii=pii,
        valid_citations=valid_cites,
        invalid_citations=invalid_cites,
        hallucinated_urls=hallucinated_urls,
        flags=flags,
        grounding_score=grounding,
        checks=checks,
    )
