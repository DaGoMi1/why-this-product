"""RAG 보조 지표 검출기. OpenAI 호출 없음."""

from __future__ import annotations

from ml.eval.rag_aux import flag_reason, outside_asins


def test_outside_asin_not_in_candidates() -> None:
    extra = outside_asins("다른 상품 B00ABCDEF1 도 좋아요", {"B000111111"})
    assert extra == ["B00ABCDEF1"]


def test_own_and_peer_asin_allowed() -> None:
    allowed = {"B000111111", "B000222222"}
    extra = outside_asins("B000111111 대신 B000222222", allowed)
    assert extra == []


def test_price_grounded_when_in_snippet() -> None:
    flags = flag_reason(
        "이 세럼은 $12 에 보습이 좋습니다.",
        {"B000111111"},
        ["A hydrating serum for $12 that lasts."],
    )
    assert flags.price_spans == ["$12"]
    assert flags.price_ungrounded == []
    assert not flags.has_issue()


def test_price_ungrounded_when_missing_from_snippet() -> None:
    flags = flag_reason(
        "가격은 12000원 입니다.",
        {"B000111111"},
        ["Very moisturizing serum, no mention of cost."],
    )
    assert flags.price_spans == ["12000원"]
    assert flags.price_ungrounded == ["12000원"]
    assert flags.has_issue()


def test_stock_ungrounded() -> None:
    flags = flag_reason(
        "지금 품절이라 빨리 사세요.",
        {"B000111111"},
        ["Gentle cleanser for sensitive skin."],
    )
    assert flags.stock_spans == ["품절"]
    assert flags.stock_ungrounded == ["품절"]


def test_empty_reason() -> None:
    flags = flag_reason("  ", {"B000111111"}, ["meta text"])
    assert flags.empty
    assert not flags.has_issue()


def test_won_suffix_not_in_원하는() -> None:
    flags = flag_reason(
        "수분을 원하는 피부에 맞습니다.",
        {"B000111111"},
        ["Good for dry skin."],
    )
    assert flags.price_spans == []
