"""RAG 설명 보조 지표: 환각·길이·latency·대략 비용. 추천 Recall과 섞지 않음."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

from backend.services.explain import ExplainService
from backend.services.recommend import RecommendService
from ml.config import load_mvp_config, resolve_path
from ml.eval.rag_aux import (
    EVAL_K,
    EVAL_QUERIES,
    EVAL_SELECT_K,
    EVAL_USER_IDS,
    flag_reason,
)
from ml.rag.llm import MissingOpenAIKeyError

# gpt-4o-mini list prices (USD / 1M tokens)
_IN = 0.15
_OUT = 0.60


def _dump_path(cfg: dict) -> Path:
    processed = resolve_path(cfg["data"]["processed_dir"])
    return processed / "eval" / "rag_aux.json"


def main() -> None:
    cfg = load_mvp_config()
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise MissingOpenAIKeyError("OPENAI_API_KEY is required for eval_rag")
    rec = RecommendService.from_processed()
    explain = ExplainService.from_processed()

    jobs: list[tuple[str, str]] = [("query", q) for q in EVAL_QUERIES]
    jobs.extend(("user", uid) for uid in EVAL_USER_IDS)

    latencies: list[float] = []
    lengths: list[int] = []
    n_reasons = 0
    n_empty = 0
    asin_reason = 0
    asin_req = 0
    price_mentions = 0
    price_ungrounded = 0
    stock_mentions = 0
    stock_ungrounded = 0
    prompt_tok = 0
    completion_tok = 0
    flagged: list[dict] = []

    for kind, key in jobs:
        if kind == "query":
            result = rec.recommend_from_query(str(key), k=EVAL_K, use_mmr=True)
            query = str(key)
        else:
            result = rec.recommend_for_user(str(key), k=EVAL_K, use_mmr=True)
            query = None
        ids = [it["item_id"] for it in result.items]
        allowed = set(ids)
        t0 = time.perf_counter()
        out = explain.explain(ids, query=query, select_k=EVAL_SELECT_K)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        prompt_tok += out.prompt_tokens
        completion_tok += out.completion_tokens
        req_asin = False
        for it in out.items:
            flags = flag_reason(it.reason, allowed, it.snippets)
            if flags.empty:
                n_empty += 1
                continue
            n_reasons += 1
            lengths.append(len(it.reason))
            if flags.asin_outside:
                asin_reason += 1
                req_asin = True
            if flags.price_spans:
                price_mentions += 1
            if flags.price_ungrounded:
                price_ungrounded += 1
            if flags.stock_spans:
                stock_mentions += 1
            if flags.stock_ungrounded:
                stock_ungrounded += 1
            if flags.has_issue():
                flagged.append(
                    {
                        "kind": kind,
                        "key": key,
                        "item_id": it.item_id,
                        "reason": it.reason,
                        "asin_outside": flags.asin_outside,
                        "price_ungrounded": flags.price_ungrounded,
                        "stock_ungrounded": flags.stock_ungrounded,
                    }
                )
        if req_asin:
            asin_req += 1
        print(
            f"[{kind}] {key!r} n={len(ids)} latency={elapsed:.2f}s "
            f"tokens={out.prompt_tokens}+{out.completion_tokens}"
        )

    n_req = len(jobs)
    p50 = statistics.median(latencies) if latencies else 0.0
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0
    cost = (prompt_tok / 1_000_000) * _IN + (completion_tok / 1_000_000) * _OUT
    summary = {
        "k": EVAL_K,
        "select_k": EVAL_SELECT_K,
        "n_requests": n_req,
        "n_query": len(EVAL_QUERIES),
        "n_user": len(EVAL_USER_IDS),
        "n_reasons": n_reasons,
        "n_empty": n_empty,
        "asin_outside_reasons": asin_reason,
        "asin_outside_requests": asin_req,
        "price_mentions": price_mentions,
        "price_ungrounded": price_ungrounded,
        "stock_mentions": stock_mentions,
        "stock_ungrounded": stock_ungrounded,
        "mean_reason_chars": round(mean_len, 1),
        "p50_latency_s": round(p50, 3),
        "tokens_in": prompt_tok,
        "tokens_out": completion_tok,
        "cost_usd": round(cost, 6),
        "cost_per_request_usd": round(cost / n_req, 6) if n_req else 0.0,
        "flagged": flagged,
    }
    dump = _dump_path(cfg)
    dump.parent.mkdir(parents=True, exist_ok=True)
    dump.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("--- rag aux ---")
    print(f"requests={n_req} reasons={n_reasons} empty={n_empty} k={EVAL_K} select_k={EVAL_SELECT_K}")
    print(f"asin_outside={asin_reason}/{n_reasons} reasons, {asin_req}/{n_req} requests")
    print(f"price_ungrounded={price_ungrounded}/{n_reasons} (mentions={price_mentions})")
    print(f"stock_ungrounded={stock_ungrounded}/{n_reasons} (mentions={stock_mentions})")
    print(f"mean_reason_chars={mean_len:.1f}")
    print(f"p50_latency_s={p50:.3f}")
    print(f"tokens_in={prompt_tok} tokens_out={completion_tok} cost_usd~={cost:.6f}")
    print(f"wrote {dump}")
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
