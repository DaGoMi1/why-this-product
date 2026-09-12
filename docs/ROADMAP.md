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

## Phase 2 — Hybrid retrieve (현재)

- [x] iALS retrieve + implicit 라벨 ablation (`all` / `>=3` / `>=4` / `>=5`) — 승자 `rating_ge_5`
- [x] two-tower (히스토리 mean-pool) — 짧게 실험 후 탈락 (학습 페어 6.6%, Recall@10 0.0000)
- [ ] content + CF hybrid / RRF fusion
- [x] retrieve ablation 표 (iALS 라벨, two-tower 탈락, `docs/EVAL.md`)

## Phase 3 — Ranking

- [ ] 랭커 후보 실험 (고정하지 않음): LightGBM/XGBoost 등 부스팅, DeepFM 등 DL, 필요 시 LambdaMART 등 LTR
- [ ] hard negative / feature 정의
- [ ] ranker ablation (baseline vs 후보들) 후 **오프라인 지표·latency 기준으로 하나 선정**
- [ ] 선정 이유·탈락 이유를 `docs/EVAL.md` 또는 README ablation 표에 기록

## Phase 4 — Re-rank & cold-start

- [ ] MMR (카테고리·임베딩 다양성)
- [ ] 신규 상품: 텍스트 임베딩 fallback
- [ ] cold-start 세그먼트 지표

## Phase 5 — RAG shopping assistant

- [ ] 상품 문서 청크 + FAISS (설명용)
- [ ] `POST /api/explain` — 근거 스니펫 + OpenAI로 한두 문장 이유 (`OPENAI_API_KEY` 필수)
- [ ] LLM: 후보 안에서만 선택/문장 다듬기 (키 없으면 기동 실패)
- [ ] latency·API 비용 표

## Phase 6 — Production polish

- [ ] Docker Compose (API + UI)
- [ ] 요청당 latency breakdown (retrieve / rank / rerank / rag)
- [ ] 데모 GIF, GitHub description·topics 정리
- [ ] (선택) GitHub Actions: lint + `pytest` smoke

---

## 작업 원칙

1. Phase를 건너뛰지 않는다. 특히 **eval 없는 모델 추가** 금지.
2. LLM은 Phase 5 이전에도 넣지 않는다 (설명 레이어가 준비된 뒤).
3. README 숫자는 `scripts/eval_smoke.py` / eval 리포트에서 재현 가능해야 한다.
