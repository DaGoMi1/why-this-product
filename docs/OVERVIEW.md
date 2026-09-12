# Overview

## 무엇을 만드나

**Why This Product?** 는 이커머스 카탈로그에서

1. multi-stage RecSys로 상품 후보를 좁히고
2. RAG + OpenAI LLM으로 **왜 이 상품을 골랐는지** 설명하는

쇼핑 어시스턴트입니다.

추천의 본체는 항상 RecSys funnel입니다. LLM은 이미 검색·랭킹된 후보 ASIN 안에서만 설명하거나 고릅니다.

## 왜 이 프로젝트인가

기존 포트폴리오([movie-recommendation](https://github.com/DaGoMi1/movie-recommendation), [why-song-serious](https://github.com/DaGoMi1/why-song-serious), [busan-trip-rag](https://github.com/DaGoMi1/busan-trip-rag))는 각각 대회 앙상블, 팀 two-stage 서빙, 여행 RAG에 강점이 있습니다.

이 레포가 채우는 갭:

| 갭 | 이 프로젝트에서의 답 |
|----|----------------------|
| 이커머스 도메인 | Amazon All_Beauty 카탈로그·리뷰 |
| 혼자 end-to-end funnel | retrieve → rank → re-rank → explain → serve |
| 평가 설계 | temporal split, Recall/NDCG, coverage, ablation |
| LLM을 RecSys 보조로 | 후보 밖 환각 금지, 비용·지연 trade-off |

깃허브 **메인(플래그십)** 프로젝트로 키웁니다.

## 타깃 사용자 / 시나리오

- 유저가 최근 본·산 상품(또는 쿼리)을 주면 top-K 추천
- 각 추천에 대해 “왜 이 상품인가” 한두 문장 설명
- (후속) 속성 Q&A: “민감성 피부용인가요?” → 메타/리뷰 근거로 답

## Goals

- 산업형 multi-stage 구조를 작은 공개 데이터로 재현
- 오프라인 지표와 서빙 latency를 README에 숫자로 남김
- RAG/LLM을 장식이 아니라 **제약 있는 보조 모듈**로 설계

## Non-goals

- LLM이 전체 카탈로그에서 추천을 생성·대체하는 것
- 대규모 실시간 feature store / 온라인 A/B 인프라
- 여러 Amazon 카테고리·멀티 데이터셋 동시 지원 (MVP 이후 확장만)
- 상용 결제·재고 연동

## 관련 문서

- [ARCHITECTURE.md](ARCHITECTURE.md) — 모듈 책임
- [ROADMAP.md](ROADMAP.md) — 단계별 체크리스트
- [DATA.md](DATA.md) — 데이터·split
- [FIELDS.md](FIELDS.md) — raw / processed 필드
- [EVAL.md](EVAL.md) — 지표
