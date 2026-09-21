"""Lexical gold matching. No parquet / OpenAI."""

from __future__ import annotations

from ml.eval.query_label import (
    QUERY_SPECS,
    assert_specs_match_eval_queries,
    blob_for_item,
    is_positive,
    matched_tokens,
)


def test_specs_align_with_eval_queries() -> None:
    assert_specs_match_eval_queries()


def test_niacinamide_serum_requires_both_groups() -> None:
    spec = next(s for s in QUERY_SPECS if s.query == "niacinamide serum")
    yes = blob_for_item("Niacinamide 10% Serum", "The Ordinary", "")
    no_serum = blob_for_item("Niacinamide moisturizer", None, "")
    no_ing = blob_for_item("Hydrating serum", None, "")
    assert is_positive(yes, spec)
    assert matched_tokens(yes, spec) == ["niacinamide", "serum"]
    assert not is_positive(no_serum, spec)
    assert not is_positive(no_ing, spec)


def test_sheet_mask_does_not_take_hair_mask() -> None:
    spec = next(s for s in QUERY_SPECS if s.query == "sheet face mask")
    assert is_positive(blob_for_item("Korean sheet mask", None, ""), spec)
    assert not is_positive(blob_for_item("Hair mask treatment", None, ""), spec)
