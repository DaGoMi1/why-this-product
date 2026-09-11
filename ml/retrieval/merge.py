"""인기도와 콘텐츠 기반 추천 결과를 혼합하여 추천 결과 생성"""

from __future__ import annotations


def merge_candidates(
    popularity: list[tuple[str, float]],                        # 인기도 기준 추천 결과
    content: list[tuple[str, float]],                           # 콘텐츠 기반 추천 결과
    merge_k: int,                                               # 추천할 상품 개수
    pop_weight: float = 0.4,                                    # 인기도 기준 가중치
    content_weight: float = 0.6,                                # 콘텐츠 기반 가중치
) -> list[tuple[str, float]]:

    # 인기도 기준 추천 결과를 정규화 (Min-Max 정규화)
    def _norm(pairs: list[tuple[str, float]]) -> dict[str, float]:
        if not pairs:
            return {}

        vals = [s for _, s in pairs]                            # 점수 리스트 생성
        lo, hi = min(vals), max(vals)                           # 최소값과 최대값 계산
        if hi <= lo:
            return {i: 1.0 for i, _ in pairs}                   # 최소값과 최대값이 같으면 모든 점수를 1.0으로 설정
        return {i: (s - lo) / (hi - lo) for i, s in pairs}      # 점수를 정규화

    pop_n = _norm(popularity)                                   # 인기도 기준 추천 결과를 정규화
    content_n = _norm(content)                                  # 콘텐츠 기반 추천 결과를 정규화
    ids = set(pop_n) | set(content_n)                 # 인기도 기준 추천 결과와 콘텐츠 기반 추천 결과의 아이템 ID 집합

    scored = [                                                  # 아이템 ID와 점수 리스트 생성                    
        (
            iid,                                                                            # 아이템 ID 
            pop_weight * pop_n.get(iid, 0.0) + content_weight * content_n.get(iid, 0.0),    # 인기도 기준 점수와 콘텐츠 기반 점수의 가중 합                 
        )
        for iid in ids
    ]

    scored.sort(key=lambda x: x[1], reverse=True)               # 점수를 내림차순으로 정렬
    return scored[:merge_k]                                     # 추천할 상품 개수만큼 반환
