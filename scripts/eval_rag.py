"""RAG 설명 보조 지표: 환각·길이·latency·대략 비용. 추천 Recall과 섞지 않음."""

from __future__ import annotations

import os
import re
import statistics
import sys
import time

from backend.services.explain import ExplainService
from backend.services.recommend import RecommendService
from ml.config import load_mvp_config
from ml.rag.llm import MissingOpenAIKeyError

_ASIN = re.compile(r"\bB0[0-9A-Z]{8}\b", re.I)
# gpt-4o-mini list prices (USD / 1M tokens)
_IN = 0.15
_OUT = 0.60


def _outside_ids(text: str, allowed: set[str]) -> list[str]:
    found = [m.upper() for m in _ASIN.findall(text)]
    return [iid for iid in found if iid not in allowed]


def main() -> None:
    cfg = load_mvp_config()
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise MissingOpenAIKeyError("OPENAI_API_KEY is required for eval_rag")
    rec = RecommendService.from_processed()
    explain = ExplainService.from_processed()
    queries = ["hydrating serum", "gentle cleanser"]
    latencies: list[float] = []
    lengths: list[int] = []
    hallu = 0
    n_reasons = 0
    prompt_tok = 0
    completion_tok = 0
    allowed_all: list[set[str]] = []

    for q in queries:
        result = rec.recommend_from_query(q, k=10)
        ids = [it["item_id"] for it in result.items]
        allowed = set(ids)
        allowed_all.append(allowed)
        t0 = time.perf_counter()
        out = explain.explain(ids, query=q, select_k=3)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        prompt_tok += out.prompt_tokens
        completion_tok += out.completion_tokens
        blob = " ".join(it.reason for it in out.items) + " " + " ".join(out.selected_ids)
        extra = _outside_ids(blob, {a.upper() for a in allowed} | {a.upper() for a in ids})
        # also treat selected_ids already filtered; leftover asins in prose
        if extra:
            hallu += 1
        for it in out.items:
            if it.reason:
                lengths.append(len(it.reason))
                n_reasons += 1
        print(
            f"[query] {q!r} n={len(ids)} selected={out.selected_ids} "
            f"latency={elapsed:.2f}s tokens={out.prompt_tokens}+{out.completion_tokens}"
        )

    n_req = len(queries)
    p50 = statistics.median(latencies) if latencies else 0.0
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0
    cost = (prompt_tok / 1_000_000) * _IN + (completion_tok / 1_000_000) * _OUT
    rate = hallu / n_req if n_req else 0.0
    print("--- rag aux ---")
    print(f"requests={n_req} reasons={n_reasons}")
    print(f"hallucination_rate={rate:.4f} ({hallu}/{n_req})")
    print(f"mean_reason_chars={mean_len:.1f}")
    print(f"p50_latency_s={p50:.3f}")
    print(f"tokens_in={prompt_tok} tokens_out={completion_tok} cost_usd~={cost:.6f}")
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
