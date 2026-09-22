"""Lexical query-gold rules. Not human relevance judgments."""

from __future__ import annotations

from dataclasses import dataclass

from ml.eval.rag_aux import EVAL_QUERIES

MAX_POSITIVES = 40
GOLD_JSONL = "data/eval/query_gold.jsonl"
GOLD_CSV = "data/eval/query_gold.csv"


@dataclass(frozen=True)
class QuerySpec:
    query: str
    intent: str
    must: tuple[tuple[str, ...], ...]


# intent: ingredient | concern | product | brand
# must: AND of groups, OR inside a group. Lowercase substrings.
QUERY_SPECS: tuple[QuerySpec, ...] = (
    QuerySpec("hydrating serum", "concern", (("hydrat", "moistur"), ("serum",))),
    QuerySpec("gentle cleanser", "concern", (("gentle", "mild", "sensitive"), ("cleanser", "face wash", "facial wash"))),
    QuerySpec("moisturizer for dry skin", "concern", (("moistur", "hydrat"), ("dry",))),
    QuerySpec("mineral sunscreen", "ingredient", (("mineral", "zinc", "titanium"), ("sunscreen", "spf", "sunblock"))),
    QuerySpec("retinol night cream", "ingredient", (("retinol", "retinoid"), ("cream", "night"))),
    QuerySpec("niacinamide serum", "ingredient", (("niacinamide",), ("serum",))),
    QuerySpec("eye cream dark circles", "concern", (("eye cream", "eye gel", "under eye"), ("dark circle", "puff", "circle"))),
    QuerySpec("hydrating lip balm", "concern", (("lip",), ("balm", "chap"))),
    QuerySpec("hydrating toner", "product", (("toner",), ("hydrat", "moistur"))),
    QuerySpec("vitamin c serum", "ingredient", (("vitamin c", "ascorbic"), ("serum",))),
    QuerySpec("hyaluronic acid serum", "ingredient", (("hyaluronic",), ("serum",))),
    QuerySpec("sheet face mask", "product", (("sheet mask", "face mask"),)),
    QuerySpec("mascara", "product", (("mascara",),)),
    QuerySpec("foundation for oily skin", "concern", (("foundation",), ("oil", "shine", "matte"))),
    QuerySpec("perfume roll on oil", "product", (("perfume", "fragrance"), ("roll",))),
    QuerySpec("electric shaver", "product", (("shaver", "razor"), ("electric", "rotary", "foil"))),
    QuerySpec("mouthwash", "product", (("mouthwash", "mouth wash", "oral rinse"),)),
    QuerySpec("body lotion", "product", (("body",), ("lotion", "cream", "butter"))),
    QuerySpec("nail polish", "product", (("nail polish", "nail lacquer", "nail enamel"),)),
    QuerySpec("exfoliating scrub", "product", (("exfoliat", "scrub"),)),
)


def _spec_queries() -> list[str]:
    return [s.query for s in QUERY_SPECS]


def assert_specs_match_eval_queries() -> None:
    listed = _spec_queries()
    if listed != list(EVAL_QUERIES):
        raise ValueError("QUERY_SPECS order/text must match EVAL_QUERIES")


def blob_for_item(title: str | None, brand: str | None, doc_text: str | None) -> str:
    parts = [title or "", brand or "", doc_text or ""]
    return " ".join(parts).casefold()


def matched_tokens(blob: str, spec: QuerySpec) -> list[str] | None:
    hits: list[str] = []
    for group in spec.must:
        found = next((tok for tok in group if tok in blob), None)
        if found is None:
            return None
        hits.append(found)
    return hits


def is_positive(blob: str, spec: QuerySpec) -> bool:
    return matched_tokens(blob, spec) is not None


def find_spec(query: str) -> QuerySpec | None:
    q = (query or "").strip()
    for spec in QUERY_SPECS:
        if spec.query == q:
            return spec
    return None


def filter_ids_by_query_spec(
    item_ids: list[str],
    query: str,
    blobs: dict[str, str],
) -> list[str]:
    """Keep ids whose blob passes QuerySpec. Unknown queries: no filter."""
    spec = find_spec(query)
    if spec is None:
        return list(item_ids)
    return [iid for iid in item_ids if is_positive(blobs.get(iid, ""), spec)]
