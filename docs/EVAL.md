# Evaluation

## 원칙

- 오프라인 지표 없이 모델을 README에 올리지 않는다.
- split은 [DATA.md](DATA.md)의 **temporal** 규칙을 따른다.
- 가능하면 세그먼트별 리포트: all / cold-user / cold-item.

## 핵심 지표

| 지표 | 용도 | 비고 |
|------|------|------|
| Recall@K | retrieve·최종 리스트 | K = 10, 50 |
| NDCG@K | 순위 품질 | K = 10 |
| Coverage | 롱테일·다양성 | 추천된 unique item / catalog |
| Intra-list diversity | MMR 효과 | 임베딩·카테고리 기준 |
| Latency p50/p95 | 서빙 | retrieve / rank / rerank / rag 분해 (Phase 6) |

(선택, 후속) IPS / 노출 편향 보정은 Phase 3+에서 검토.

## MVP 스모크 (`scripts/eval_smoke.py`)

Phase 1 완료 기준:

1. train으로 만든 popularity·FAISS가 로드된다
2. valid 유저 샘플 N명에 대해 Recall@10이 계산된다
3. popularity-only vs popularity+content 비교 한 줄이 출력된다
4. 실패 시 non-zero exit

### 최근 스모크 결과 (warm valid 200 users)

| 구성 | Recall@10 |
|------|----------|
| popularity | 0.1689 |
| popularity+content | 0.0400 |

이 데이터·split에서는 popularity가 강하다. content 채널은 쿼리/시드 아이템 유사도용으로 유지하고, Phase 2 hybrid CF·랭커로 보완한다.

## Ablation 템플릿 (README에 채울 표)

| 구성 | Recall@10 | NDCG@10 | Coverage | p50 ms |
|------|----------|---------|----------|--------|
| popularity | — | — | — | — |
| + content FAISS | — | — | — | — |
| + hybrid CF | — | — | — | — |
| + ranker (선정 모델) | — | — | — | — |
| + MMR | — | — | — | — |
| + RAG + OpenAI explain | — | — | — | cost |

랭커는 DeepFM에 고정하지 않는다. Phase 3에서 부스팅·DL·LTR 후보를 같은 feature/split으로 비교한 뒤, 지표·latency·구현 복잡도를 보고 고른다.

OpenAI는 설명/후보 내 선택에 **항상** 사용한다. 키 없는 fallback 경로는 두지 않는다.

## LLM 평가

- 추천 품질의 주 지표로 LLM 점수를 쓰지 않는다.
- LLM 관련은 **환각률(후보 밖 id)**, 설명 길이, 요청당 비용만 보조 지표로 둔다.
