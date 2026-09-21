# Overview

## 무엇을 만드나

**Why This Product?** 는 이커머스 카탈로그에서 영어 쿼리로 상품을 찾고, RAG + OpenAI로 **왜 이 상품인가**를 스니펫 근거와 함께 설명하는 쇼핑 어시스턴트입니다.

Phase 7의 운영 축은 검색 적합성 **라벨 기준**, 카탈로그 **속성 품질**, 설명 **근거 검수**입니다.  
Phase 0–6 RecSys funnel(pop / iALS / 랭커)은 희소 데이터에서 popularity가 이긴 **실험 유산**으로 [EVAL.md](EVAL.md)에 남깁니다. LLM은 이미 고른 후보 ASIN 안에서만 설명합니다.

## 왜 이 프로젝트인가

기존 포트폴리오([movie-recommendation](https://github.com/DaGoMi1/movie-recommendation), [why-song-serious](https://github.com/DaGoMi1/why-song-serious), [busan-trip-rag](https://github.com/DaGoMi1/busan-trip-rag))는 각각 대회 앙상블, 팀 two-stage 서빙, 여행 RAG에 강점이 있습니다.

이 레포가 채우는 갭:

| 갭 | 이 프로젝트에서의 답 |
|----|----------------------|
| 이커머스 도메인 | Amazon All_Beauty 카탈로그·리뷰 |
| 혼자 end-to-end funnel | retrieve → rank → re-rank → explain → serve |
| 평가 설계 | temporal split, Recall/NDCG, 쿼리 라벨·미탐/오탐 |
| LLM을 RecSys 보조로 | 후보 밖 환각 금지, 스니펫 인용, 비용·지연 trade-off |

## 타깃 사용자 / 시나리오

- 기본: 영어 쿼리로 top-K를 찾고, 각 상품에 한국어 한두 문장 + 영어 스니펫 인용
- 데모에 `user_id`(인기+MMR)는 남아 있다. 서빙에서 빼는 것은 Phase 8

## Goals

- 검색 적합성 라벨을 재현 가능한 규칙으로 남김 ([LABELING.md](LABELING.md))
- 카탈로그 속성 공백·쿼리 미탐/오탐을 숫자로 남김
- RAG/LLM을 장식이 아니라 **스니펫에 묶인 보조 모듈**로 설계
- (유산) 산업형 multi-stage를 작은 공개 데이터로 재현하고, 탈락 이유를 EVAL에 둠

## 관련 문서

- [ARCHITECTURE.md](ARCHITECTURE.md) — 모듈 책임
- [ROADMAP.md](ROADMAP.md) — 단계별 체크리스트
- [DATA.md](DATA.md) — 데이터·split
- [FIELDS.md](FIELDS.md) — raw / processed 필드
- [EVAL.md](EVAL.md) — 지표
- [LABELING.md](LABELING.md) — 쿼리 적합성 라벨 규칙
