"""OpenAI 설명. 후보 item_id 밖은 버린다. 키 없으면 실패."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from openai import OpenAI

from ml.rag.index import RagChunk

# 상품별 호출은 유지하되 I/O 대기는 병렬로 겹친다.
_DEFAULT_PARALLEL_WORKERS = 8


class MissingOpenAIKeyError(RuntimeError):
    pass


@dataclass
class ExplainItem:
    item_id: str
    reason: str
    snippets: list[dict[str, str]]


@dataclass
class ExplainResult:
    items: list[ExplainItem]
    selected_ids: list[str]
    prompt_tokens: int
    completion_tokens: int
    model: str
    timings_ms: dict[str, float] = field(default_factory=dict)


def require_api_key() -> str:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise MissingOpenAIKeyError("OPENAI_API_KEY is required for /api/explain")
    return key


def _usage(resp) -> tuple[int, int]:
    usage = resp.usage
    if not usage:
        return 0, 0
    return int(usage.prompt_tokens or 0), int(usage.completion_tokens or 0)


def _parse_json(raw: str) -> dict:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("LLM did not return JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("LLM did not return a JSON object")
    return data


def _chat(client: OpenAI, model: str, system: str, user: dict) -> tuple[dict, int, int]:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(user, ensure_ascii=False),
            },
        ],
    )
    prompt_tokens, completion_tokens = _usage(resp)
    data = _parse_json(resp.choices[0].message.content or "{}")
    return data, prompt_tokens, completion_tokens


def _reason_for_one(
    client: OpenAI,
    model: str,
    item_id: str,
    title: str,
    snippets: list[dict[str, str]],
    query: str,
) -> tuple[str, int, int]:
    system = (
        "You write a short product reason for an e-commerce recommender. "
        "The reason field MUST be Korean Hangul sentences. "
        "Do not write the reason in English even if the query, title, and snippets are English. "
        "Product names and brand names may stay in English. "
        "Use only the provided snippets. Do not invent price, stock, or facts. "
        "Do not claim ingredients or effects that are not in the snippets. "
        "If the query asks for something the snippets do not support, say only what the snippets support. "
        "Write about this one product only. One or two sentences. "
        'Reply JSON only: {"item_id":"...","reason":"..."}'
    )
    data, pin, pout = _chat(
        client,
        model,
        system,
        {
            "query": query,
            "item_id": item_id,
            "title": title,
            "snippets": snippets,
        },
    )
    got_id = str(data.get("item_id") or "")
    reason = str(data.get("reason") or "").strip()
    if got_id and got_id != item_id:
        return "", pin, pout
    return reason, pin, pout


def _select_ids(
    client: OpenAI,
    model: str,
    query: str,
    rows: list[dict[str, str]],
    select_k: int,
    allowed: set[str],
) -> tuple[list[str], int, int]:
    system = (
        "You pick recommended products from the given list only. "
        f"Return at most {select_k} item_id values. "
        "Do not invent ids. "
        'Reply JSON only: {"selected_ids":["..."]}'
    )
    data, pin, pout = _chat(
        client,
        model,
        system,
        {"query": query, "candidates": rows, "select_k": select_k},
    )
    selected: list[str] = []
    for raw_id in data.get("selected_ids") or []:
        iid = str(raw_id)
        if iid in allowed and iid not in selected:
            selected.append(iid)
        if len(selected) >= select_k:
            break
    return selected, pin, pout


def explain_items(
    item_ids: list[str],
    titles: dict[str, str | None],
    snippets: dict[str, list[RagChunk]],
    query: str | None,
    model: str,
    select_k: int = 0,
    max_workers: int | None = None,
) -> ExplainResult:
    allowed = [iid for iid in item_ids if iid in snippets]
    if not allowed:
        raise ValueError("no snippets for requested item_ids")
    key = require_api_key()
    client = OpenAI(api_key=key)
    user_q = (query or "").strip() or "이 상품을 추천하는 이유를 짧게 설명하세요."
    workers = max(1, min(max_workers or _DEFAULT_PARALLEL_WORKERS, len(allowed)))

    def _one(iid: str) -> tuple[ExplainItem, int, int]:
        snips = [{"text": c.text, "source": c.source} for c in snippets[iid]]
        reason, pin, pout = _reason_for_one(
            client,
            model,
            iid,
            titles.get(iid) or "",
            snips,
            user_q,
        )
        return ExplainItem(item_id=iid, reason=reason, snippets=snips), pin, pout

    prompt_tokens = 0
    completion_tokens = 0
    items: list[ExplainItem] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for item, pin, pout in pool.map(_one, allowed):
            items.append(item)
            prompt_tokens += pin
            completion_tokens += pout
    selected: list[str] = []
    if select_k > 0:
        rows = [{"item_id": it.item_id, "reason": it.reason} for it in items]
        selected, pin, pout = _select_ids(
            client,
            model,
            user_q,
            rows,
            select_k,
            set(allowed),
        )
        prompt_tokens += pin
        completion_tokens += pout
    return ExplainResult(
        items=items,
        selected_ids=selected,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
    )
