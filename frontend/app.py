"""추천 리스트와 상품 설명."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000")

DEMO_USERS = [
    ("A171X499VNL9QL", "전동 면도기·프리셰이브 (Norelco 등)"),
    ("A1E5TS4G2REL77", "세럼·모이스처라이저·잡티 케어"),
    ("A1EV4A8OSRC5IO", "향수 롤온 오일"),
    ("A1U8L4X1O2LXXF", "메이크업 (파운데이션·마스카라)"),
    ("A1QBOC76MIOJYP", "구강 케어 (워터픽·가글)"),
]

st.set_page_config(page_title="Why This Product", layout="centered")
st.title("Why This Product?")
st.caption(
    "쿼리는 영어 상품 검색(content FAISS + MMR), user_id는 인기 추천 + MMR입니다. "
    "후보 안에서만 이유를 설명합니다."
)

if "input_mode" not in st.session_state:
    st.session_state.input_mode = "query"
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "query" not in st.session_state:
    st.session_state.query = ""
if "recs" not in st.session_state:
    st.session_state.recs = []
if "explain" not in st.session_state:
    st.session_state.explain = None
if "explain_error" not in st.session_state:
    st.session_state.explain_error = None


def _use_query() -> None:
    st.session_state.input_mode = "query"
    st.session_state.user_id = ""


def _use_user_id() -> None:
    st.session_state.input_mode = "user_id"
    st.session_state.query = ""


def _fill_user(uid: str) -> None:
    st.session_state.user_id = uid


if st.session_state.input_mode == "user_id":
    st.text_input("user_id", key="user_id")
    st.caption("이 모드는 인기 상품을 MMR로 보여 주고, 이미 본 상품은 빼니다.")
    for uid, desc in DEMO_USERS:
        st.button(
            f"{uid} — {desc}",
            key=f"demo_{uid}",
            on_click=_fill_user,
            args=(uid,),
        )
    st.button("query 사용", on_click=_use_query)
else:
    st.text_input("query", key="query")
    st.caption(
        "카탈로그와 임베딩이 영어라 영어 문장으로 써야 알맞은 추천이 나옵니다. "
        "예: I need a hydrating serum / gentle cleanser for sensitive skin"
    )
    st.button("user_id 사용", on_click=_use_user_id)

k = st.slider("k", 1, 5, 5)

if st.button("Recommend"):
    payload: dict = {"k": k, "use_mmr": True}
    query_value = ""
    if st.session_state.input_mode == "user_id":
        user_id = (st.session_state.user_id or "").strip()
        if not user_id:
            st.error("user_id가 필요합니다.")
            st.stop()
        payload["user_id"] = user_id
    else:
        query_value = (st.session_state.query or "").strip()
        if not query_value:
            st.error("query가 필요합니다.")
            st.stop()
        payload["query"] = query_value
    try:
        resp = httpx.post(f"{API}/api/recommend", json=payload, timeout=60.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        st.error(str(exc))
        st.session_state.recs = []
        st.session_state.explain = None
        st.session_state.explain_error = None
    else:
        data = resp.json()
        items = data.get("items", [])
        st.session_state.recs = items
        st.session_state.explain = None
        st.session_state.explain_error = None
        ids = [it["item_id"] for it in items]
        if ids:
            try:
                exp = httpx.post(
                    f"{API}/api/explain",
                    json={
                        "item_ids": ids,
                        "query": query_value or None,
                        "select_k": 0,
                    },
                    timeout=120.0,
                )
                if exp.status_code == 503:
                    st.session_state.explain_error = exp.json().get(
                        "detail", "설명을 가져올 수 없습니다."
                    )
                else:
                    exp.raise_for_status()
                    st.session_state.explain = exp.json()
            except httpx.HTTPError as exc:
                st.session_state.explain_error = str(exc)

items = st.session_state.recs
if items:
    by_id = {}
    explained = st.session_state.explain
    if explained:
        by_id = {row["item_id"]: row for row in explained.get("items", [])}
    fallback = st.session_state.explain_error or "설명을 만들지 못했습니다."
    for i, it in enumerate(items, start=1):
        iid = it["item_id"]
        title = it.get("title") or iid
        row = by_id.get(iid) or {}
        reason = (row.get("reason") or "").strip() or fallback
        st.write(f"추천 상품 {i}: {title}")
        st.write(f"추천 사유 {i}: {reason}")
        for snip in (row.get("snippets") or [])[:3]:
            src = (snip.get("source") or "snippet").strip() or "snippet"
            text = " ".join(str(snip.get("text") or "").split())
            if len(text) > 220:
                text = text[:217].rstrip() + "..."
            if text:
                st.caption(f"[{src}] {text}")
        if i < len(items):
            st.text("----")
