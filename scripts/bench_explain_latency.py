"""Phase 6 style latency: recommend k=5 then explain (parallel LLM)."""

from __future__ import annotations

import statistics
import sys
import time

from backend.services.explain import ExplainService
from backend.services.recommend import RecommendService
from ml.rag.llm import MissingOpenAIKeyError


def main() -> None:
    query = "hydrating serum"
    k = 5
    n_runs = 3
    rec = RecommendService.from_processed()
    explain = ExplainService.from_processed()

    result = rec.recommend_from_query(query, k=k, use_mmr=True)
    ids = [it["item_id"] for it in result.items]
    print(f"recommend strategy={result.strategy} timings_ms={result.timings_ms}")
    print(f"ids={ids}")

    rag_ms: list[float] = []
    costs = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        out = explain.explain(ids, query=query, select_k=0)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        rag_ms.append(out.timings_ms.get("rag", elapsed_ms))
        cost = (out.prompt_tokens / 1_000_000) * 0.15 + (out.completion_tokens / 1_000_000) * 0.60
        costs.append(cost)
        empty = sum(1 for it in out.items if not (it.reason or "").strip())
        print(
            f"run={i+1} rag_ms={rag_ms[-1]:.1f} "
            f"tokens={out.prompt_tokens}+{out.completion_tokens} "
            f"empty={empty}/{len(out.items)} cost~=${cost:.6f}"
        )

    print("--- summary ---")
    print(f"recommend_total_ms={result.timings_ms.get('total')}")
    print(f"rag_p50_ms={statistics.median(rag_ms):.1f}")
    print(f"rag_mean_ms={statistics.mean(rag_ms):.1f}")
    print(f"cost_mean_usd={statistics.mean(costs):.6f}")
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except MissingOpenAIKeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
