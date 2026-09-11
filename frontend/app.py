"""Streamlit UI placeholder — MVP recommend/explain demo comes in Phase 1+."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Why This Product", page_icon=None, layout="centered")

st.title("Why This Product?")
st.caption("이커머스 RecSys + RAG 쇼핑 어시스턴트")

st.info(
    "UI는 스캐폴딩 placeholder입니다. "
    "추천·설명 데모는 ROADMAP Phase 1 이후 API와 함께 연결합니다."
)

st.markdown(
    """
### Coming: MVP UI
- 유저 / seed item / 쿼리 입력
- top-K 추천 리스트
- “왜 이 상품인가” 설명 패널

문서: `docs/ROADMAP.md` · API: `uvicorn backend.main:app`
"""
)
