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
- [x] latency·API 비용 표 (p50 14.16s, ~$0.0008/req, 환각 0, 사유 20/20)

## Phase 6 — Production polish (현재)

- [ ] Docker Compose (API + UI)
- [ ] 요청당 latency breakdown (retrieve / rank / rerank / rag)
- [ ] 데모 GIF, GitHub description·topics 정리
- [ ] (선택) GitHub Actions: lint + `pytest` smoke

---



## 작업 원칙

1. Phase를 건너뛰지 않는다. 특히 **eval 없는 모델 추가** 금지.
2. LLM은 Phase 5 이전에도 넣지 않는다 (설명 레이어가 준비된 뒤).
3. README 숫자는 `scripts/eval_smoke.py` / eval 리포트에서 재현 가능해야 한다.

