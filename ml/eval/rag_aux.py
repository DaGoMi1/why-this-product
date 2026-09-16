"""RAG 설명 보조 지표: 후보 밖 ASIN·가격·재고 표면 검출. OpenAI 호출 없음."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

EVAL_K = 5
EVAL_SELECT_K = 0

EVAL_QUERIES = [
    "hydrating serum",
    "gentle cleanser",
    "moisturizer for dry skin",
    "mineral sunscreen",
    "retinol night cream",
    "niacinamide serum",
    "eye cream dark circles",
    "hydrating lip balm",
    "hydrating toner",
    "vitamin c serum",
    "hyaluronic acid serum",
    "sheet face mask",
    "mascara",
    "foundation for oily skin",
    "perfume roll on oil",
    "electric shaver",
    "mouthwash",
    "body lotion",
    "nail polish",
    "exfoliating scrub",
]

EVAL_USER_IDS = [
    "A171X499VNL9QL",
    "A1E5TS4G2REL77",
    "A1EV4A8OSRC5IO",
    "A1U8L4X1O2LXXF",
    "A1QBOC76MIOJYP",
]

_ASIN = re.compile(r"\bB0[0-9A-Z]{8}\b", re.I)
_PRICE = re.compile(
    r"\$\s*\d+(?:[.,]\d+)?|"
    r"₩\s*\d+(?:[.,]\d+)?|"
    r"\d+(?:[.,]\d+)?\s*(?:달러|불|만원|원)",
    re.I,
)
_STOCK = re.compile(
    r"재고|품절|재입고|in\s*stock|out\s*of\s*stock|sold\s*out",
    re.I,
)


@dataclass
class ReasonFlags:
    asin_outside: list[str] = field(default_factory=list)
    price_spans: list[str] = field(default_factory=list)
    price_ungrounded: list[str] = field(default_factory=list)
    stock_spans: list[str] = field(default_factory=list)
    stock_ungrounded: list[str] = field(default_factory=list)
    empty: bool = False

    def has_issue(self) -> bool:
        return bool(
            self.asin_outside or self.price_ungrounded or self.stock_ungrounded
        )


def snippet_blob(snippets: list[str] | list[dict]) -> str:
    parts: list[str] = []
    for row in snippets:
        if isinstance(row, dict):
            parts.append(str(row.get("text") or ""))
        else:
            parts.append(str(row))
    return " ".join(parts)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _span_in_snippets(span: str, blob: str) -> bool:
    if not span or not blob:
        return False
    compact_blob = _compact(blob)
    if _compact(span) in compact_blob:
        return True
    blob_l = blob.lower()
    span_l = span.lower()
    if span_l in blob_l:
        return True
    for num in re.findall(r"\d+(?:[.,]\d+)?", span):
        n = num.replace(",", "")
        if f"${n}" in compact_blob or f"₩{n}" in compact_blob:
            return True
        if f"${num.lower()}" in blob_l or f"₩{num.lower()}" in blob_l:
            return True
    return False


def outside_asins(text: str, allowed: set[str]) -> list[str]:
    allow = {a.upper() for a in allowed}
    found = [m.upper() for m in _ASIN.findall(text or "")]
    extra: list[str] = []
    seen: set[str] = set()
    for iid in found:
        if iid not in allow and iid not in seen:
            extra.append(iid)
            seen.add(iid)
    return extra


def flag_reason(
    reason: str,
    allowed: set[str],
    snippets: list[str] | list[dict],
) -> ReasonFlags:
    text = (reason or "").strip()
    if not text:
        return ReasonFlags(empty=True)
    blob = snippet_blob(snippets)
    flags = ReasonFlags(asin_outside=outside_asins(text, allowed))
    for m in _PRICE.finditer(text):
        span = m.group(0).strip()
        flags.price_spans.append(span)
        if not _span_in_snippets(span, blob):
            flags.price_ungrounded.append(span)
    for m in _STOCK.finditer(text):
        span = m.group(0).strip()
        flags.stock_spans.append(span)
        if not _span_in_snippets(span, blob):
            flags.stock_ungrounded.append(span)
    return flags
