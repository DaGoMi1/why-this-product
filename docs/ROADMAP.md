# Roadmap

플래그십을 **MVP부터** 키웁니다. 체크박스는 완료 시 채웁니다.

## Phase 0 — Scaffold

- [x] README, LICENSE, .gitignore, .env, requirements
- [x] docs (OVERVIEW / ARCHITECTURE / ROADMAP / DATA / EVAL)
- [x] 패키지 트리 + FastAPI `/health` + Streamlit placeholder

## Phase 1 — Data & baselines

- [x] Amazon Reviews 2018 All_Beauty 다운로드 스크립트
- [x] 상호작용·상품 메타 정규화 (`docs/DATA.md` 스키마)
- [x] **temporal split** (누수 방지)
- [x] popularity baseline
- [x] content FAISS 인덱스 빌드
- [x] `scripts/eval_smoke.py` — Recall@K 스모크
- [x] `POST /api/recommend` → popularity + content 병합

## Phase 2 — Hybrid retrieve

- [x] iALS retrieve + implicit 라벨 ablation (`all` / `>=3` / `>=4` / `>=5`) — 승자 `rating_ge_5`
- [x] two-tower (히스토리 mean-pool) — 짧게 실험 후 탈락 (학습 페어 6.6%, Recall@10 0.0000)
- [x] content 시드 쿼리 `mean` vs `per_seed` — 승자 `per_seed` (multi-seed 0.0204 → 0.0510)
- [x] content + CF hybrid / RRF fusion — 최고 RRF(pop, iALS)=0.1114, pop(0.1689) 미달 → 서빙 기본은 popularity
- [x] retrieve ablation 표 (iALS 라벨, two-tower 탈락, content 시드, RRF, `docs/EVAL.md`)



## Phase 3 — Ranking

- [x] 피처 + hard negative 정의 (train 마지막 아이템 양성, pop 200 위 hard neg, train 통계만)
- [x] 첫 모델 LightGBM (pop 후보 재정렬). Recall@10 0.0813 < pop 0.1689 → 서빙 기본은 popularity, `use_ranker` 플래그
- [x] 부스팅 후보 LightGBM / XGBoost / CatBoost. 최고 XGB 0.0938 < pop 0.1689 → 서빙 기본은 popularity
- [x] retrieve@200 vs 펀넬@10 진단. 풀 승자 pop(0.3468), 기존 랭커 최고 0.0938 < pop 단일 0.1689 → 서빙은 단일 단계
- [x] 오프라인 지표 확정: 관련 Recall(`rating>=4`) + 등급 NDCG@10. pop 카운트도 >=4. 승자 여전히 pop (R@10 0.2023 / NDCG 0.0699)
- [x] 긍정 `rating>=5` 통일. retrieve×랭커 격자 + 카탈로그 단독. 최종 @10 승자 pop 0.1900 / 0.0698. @200은 RRF(pop, content) 0.3500이지만 top-10 미달 → 서빙은 popularity 단일
- [x] DeepFM / LambdaMART 스킵. ge_5 격자에서 재정렬·카탈로그 단독이 전부 pop 미달이고 학습 라벨이 희소(양성-in-pop 8,894)
- [x] ranker ablation 종료. 선정은 **랭커 없음 = popularity 단일** (`use_ranker` 기본 off)
- [x] 선정·탈락·미실험 이유 `docs/EVAL.md`

## Phase 4 — Re-rank & cold-start

- [x] MMR (임베딩 ILD). λ=0.5 서빙 (R@10 0.1700, ILD 0.864). λ=0.7은 Recall 붕괴. unique category는 이 카탈로그에서 상수 1
- [x] 신규 상품: train 미등장 ASIN을 content `per_seed`로 풀에 최대 20. @10 히트는 0
- [x] cold-start 세그먼트: cold-user n=200 pop 0.0860 vs MMR0.5 0.0785. cold-item GT n=17 전부 0

## Phase 5 — RAG shopping assistant

- [x] 상품 문서 청크 + FAISS (설명용, train 메타+리뷰, retrieve 인덱스와 분리)
- [x] `POST /api/explain` — 근거 스니펫 + OpenAI로 한두 문장 이유 (`OPENAI_API_KEY` 필수)
- [x] LLM: 후보 안에서만 선택/문장 다듬기 (키 없으면 503)
- [x] latency·API 비용 표 (직렬 p50 5.45s → 병렬 p50 ~1.4s, ~$0.00030/req, k=5)

## Phase 6 — Production polish

- [x] Docker Compose (API + UI)
- [x] 요청당 latency breakdown (retrieve / rank / rerank / rag)
- [x] 데모 GIF, GitHub description·topics 정리

## Phase 7 — 검색 운영 (라벨·품질·근거)

서빙 계약은 그대로다. `user_id` 제거·BM25/하이브리드 서빙은 Phase 10, GitHub Actions·클라우드 배포는 Phase 11.

- [x] 쿼리 적합성 gold + 라벨 가이드 ([docs/LABELING.md](LABELING.md): 적합/애매/오탐, 쿼리 유형)
- [x] content FAISS(+MMR) vs gold: Recall·미탐·오탐 (`scripts/eval_query.py`). 서빙 기본 채널은 바꾸지 않음
- [x] 카탈로그 속성 품질 CSV (빈 title/brand/description, category 붕괴, price 결측)
- [x] 쿼리 조건 스니펫 선정 + UI 인용. RAG 환각 보조 지표 유지
- [x] OVERVIEW에 검색 운영(라벨·품질·근거). RecSys ablation은 EVAL 유산

아래 Phase 8+는 **데이터로 RecSys를 키우기보다** 운영 루프·LLMOps·검색 서빙·엔지니어링을 앞세운다. 멀티캣 RecSys는 Phase 12(후순위).

## Phase 8 — 검색·생성 운영 루프

실패 유형을 고치고 같은 지표로 다시 재는 습관을 제품 루프로 고정한다.

- [x] 쿼리 유형별 오탐·미탐 리포트 → 검색 lexical gate만 수정 후 재측정 (`eval_query` fp 62→0, P@10 0.36→0.545; RAG empty 0 유지). [EVAL.md](EVAL.md) Phase 8
- [ ] 규칙 gold 위에 **사람 검수 샘플** 소량 (적합/애매/오탐). 규칙≠사람 GT를 [LABELING.md](LABELING.md)에 명시 유지
- [ ] 카탈로그 공백(brand/desc/price)을 메타 품질 백로그로 문서화 (모델로 메우지 않음)

## Phase 9 — LLMOps (설명 경로)

병렬 호출 다음 단계. 토큰·비용을 운영 지표로 쌓는다.

- [ ] explain 캐시 (`query + item_id`)
- [ ] 배치 호출 + JSON/필수 필드 검증 + 빈 사유만 부분 재시도
- [ ] 프롬프트 버전·실험 로그 (입출력·empty·토큰을 남김)
- [ ] 첫 사유 스트리밍/점진 UI (체감 latency)
- [ ] 요청 로그에 p50·토큰·비용 집계 스크립트/표 갱신 ([EVAL.md](EVAL.md))

## Phase 10 — 검색 서빙 정리

Phase 7에서 미룬 서빙 계약.

- [ ] 데모/`user_id` 제거 또는 query-only를 기본 경로로 정리 ([OVERVIEW.md](OVERVIEW.md)와 맞춤)
- [ ] BM25 또는 lexical+dense 하이브리드 후보 → `eval_query`로 FAISS+MMR과 비교 후 **이길 때만** 서빙 반영
- [ ] 서빙 기본 채널 변경 시 EVAL·README 숫자 재현

## Phase 11 — 엔지니어링

- [ ] GitHub Actions: `eval_smoke` / 단위 테스트 (키 없는 경로; OpenAI 호출 CI 제외 또는 mock)
- [ ] 클라우드/공개 데모 배포 (Compose 기준)

## Phase 12 — RecSys 데이터 확장 (후순위)

모델 승을 위한 데이터 실험. **제품 서사의 주축이 아니다.** RAG 운영 스토리를 덮어쓰지 않는다.

- [ ] Amazon 카테고리 2~3개 union (`reviewerID`) 후 `user_n`·cold%·unique category 재집계
- [ ] warm 세그먼트에서 pop vs iALS 재측정; 전체 서빙 기본은 숫자로만 결정
- [ ] RecSys ablation과 Beauty RAG 데모 범위를 DATA/EVAL에 분리 기술

---

## 작업 원칙

1. Phase를 건너뛰지 않는다. 특히 **eval 없는 모델 추가** 금지.
2. LLM은 Phase 5 이전에도 넣지 않는다 (설명 레이어가 준비된 뒤).
3. README 숫자는 `scripts/eval_smoke.py` / eval 리포트에서 재현 가능해야 한다.
4. Phase 7 숫자는 `scripts/eval_query.py` / `scripts/report_catalog_quality.py`에서 재현한다. 라벨은 사람이 검수한 것처럼 포장하지 않는다.
5. query-only·BM25/하이브리드 서빙은 Phase 10, CI·배포는 Phase 11에서만 연다.
6. Phase 8–9에서 프롬프트·캐시·스니펫을 바꾸면 반드시 관련 지표를 재측정한다 (eval 없는 변경 금지).
7. Phase 12는 RecSys 전용이다. Beauty RAG·검색 운영 범위를 바꾸지 않은 채 ablation만 확장한다.

