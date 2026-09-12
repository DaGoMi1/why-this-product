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

Phase 1–2 스모크 기준:

1. train으로 만든 popularity·FAISS가 로드된다
2. valid 유저 샘플 N명에 대해 Recall@10이 계산된다
3. popularity-only vs popularity+content 비교가 출력된다
4. iALS artifact가 있으면 라벨 variant 4줄(Recall@10)이 출력된다
5. two-tower artifact가 있으면 Recall@10 한 줄이 출력된다
6. content 단독 `mean` vs `per_seed` Recall@10이 출력된다 (전체 + multi-seed 세그먼트)
7. 실패 시 non-zero exit

### 최근 스모크 결과 (warm valid 200 users)

| 구성 | Recall@10 |
|------|----------|
| popularity | 0.1689 |
| popularity+content | 0.0400 |
| iALS `all` | 0.0475 |
| iALS `rating_ge_3` | 0.0450 |
| iALS `rating_ge_4` | 0.0675 |
| iALS `rating_ge_5` | 0.1048 |
| two-tower (history mean-pool) | 0.0000 |
| content `mean` (단독) | 0.0450 |
| content `per_seed` (단독) | 0.0525 |

이 데이터·split에서는 popularity가 강하다. content 채널은 쿼리/시드 아이템 유사도용으로 유지하고, iALS는 별도 CF retrieve 채널로 둔다 (이 슬라이스에서는 hybrid 합치기 없음). two-tower는 짧게 실험 후 탈락했다. content 시드 쿼리 기본값은 `per_seed`.

## iALS implicit 라벨 (실험 전 예측 → 실측)

explicit 평점(1~5)은 회귀하지 않는다. train 상호작용만 양성으로 바꿀 때 기준을 나눈다.
`0`은 네거티브 샘플링이 아니다. 임계 미달 리뷰는 양성에서 **빼는 것**이고, 미관측 칸은 iALS가 약한 부정으로 취급한다.

이 데이터 평점은 정수라 `>= 4.5`와 `>= 5`는 같다. 만점 기준은 `rating_ge_5` 하나다.

EDA 전체 상호작용 평점:

- 1점 38,486 / 2점 19,854 / 3점 28,809 / 4점 50,952 / 5점 218,980
- 5점 약 61%, 4점 이상 약 76%, 3점 이상 약 84%, 1~2점 약 16%

| variant | 양성 | 사전 예측 (학습 전) | Recall@10 실측 |
|---------|------|---------------------|----------------|
| `all` | 리뷰만 있으면 1 | 낮은 평점도 긍정 → **네 경우 중 가장 낮을 것** | 0.0475 |
| `rating_ge_3` | `rating >= 3` | `all`보다 나을 것. 3점 중립 잡음은 남음 | 0.0450 |
| `rating_ge_4` | `rating >= 4` | `>=3`과 비슷하거나 조금 나을 것 | 0.0675 |
| `rating_ge_5` | `rating >= 5` | 깨끗함 vs 커버리지. `>=4`와 경합 | 0.1048 |

평가 ground truth는 variant와 무관하게 **warm valid 상호작용 존재** (Phase 1과 동일). 같은 warm valid 200 users.

실험 후: `all` vs `>=3` 차이는 거의 없고 `>=3`이 조금 더 낮아 예측과 어긋났다. `>=4`부터 오르고, **`>=5`가 iALS 네 경우 중 최고**. 5점이 이미 61%라 만점만 남겨도 학습량이 크게 줄지 않았고, 4점 이하를 빼는 쪽이 더 이득이었다. 서빙 기본값은 `ials.variant: rating_ge_5`. 그래도 popularity(0.1689)가 더 높다 — 유저당 리뷰 중앙값 1인 롱테일에서 CF 단독 retrieve는 아직 약하다.

## Two-Tower (히스토리 mean-pool, 짧은 탈락 실험)

유저 ID 임베딩은 쓰지 않는다. 유저 벡터 = 타깃 **이전** `rating >= 5` 아이템 임베딩의 평균(최대 10개). 아이템 타워는 ID 임베딩(dim 64). 손실은 in-batch softmax. 라벨은 iALS 승자 `rating_ge_5`만. API에는 넣지 않는다.

iALS와 차이: iALS는 리뷰 1개여도 `p_u(유저 임베딩)`가 생긴다. 이 모델은 타깃을 빼면 히스토리가 0인 유저를 **학습에 못 넣는다**.

EDA 전체: 유저 319,335명, 유저당 리뷰 중앙값 1 / 75%도 1 / 평균 1.12.

### 학습 전 예측

- 학습 페어는 전체 train의 소수일 것 (리뷰 2개 이상 + 이전 ge_5 히스토리 있는 유저만)
- Recall@10은 popularity(0.1689)보다 낮을 것
- iALS `rating_ge_5`(0.1048)보다도 낮을 가능성이 큼 (유저 전용 벡터 없음)
- content(0.0400)와는 경합. **맞지 않음**의 기준은 popularity·iALS를 못 넘는 것

같은 warm valid 200 users. 학습 로그(`scripts/train_two_tower.py`) 실측:

| 항목 | 사전 예측 | 실측 |
|------|-----------|------|
| train 유저 중 리뷰 1개인 비율 | 대부분 (EDA 중앙값·p75 = 1) | 229,880 / 254,641 (**90.3%**) |
| ge_5 양성 중 히스토리 없어 버린 비율 | 높을 것 | 156,561 / 175,329 (**89.3%**) |
| 학습 페어 수 | 전체 train 대비 소수 | 18,768 (train 285,664의 **6.6%**) |
| Recall@10 two-tower | popularity·iALS보다 낮을 것 | **0.0000** |

실험 후: 예측이 맞았다. train 유저의 90%는 리뷰 1개라 타깃을 빼면 히스토리가 없고, ge_5 양성의 89%가 학습에서 버려졌다. 남은 6.6% 페어로는 5 epoch 손실이 거의 안 내려갔다(6.22 → 6.13). 워밍 유저는 히스토리가 있어 popularity fallback이 거의 안 타는데, 그 리스트가 valid를 하나도 못 맞췄다. **이 데이터에서 Two-Tower retrieve는 탈락.** 기본 retrieve는 popularity, CF 채널은 iALS `rating_ge_5`. API에 two-tower는 넣지 않는다.

## Content 시드 쿼리 (`mean` vs `per_seed`)

content FAISS는 최근 시드(`history[-5:]`)로 비슷한 상품을 찾는다. 기존 `mean`은 시드 벡터를 평균낸 뒤 **한 번** 검색한다. 시드가 립스틱+샴푸처럼 멀면 평균은 둘 다 아닌 점이 된다.

`per_seed`는 시드마다 검색한 뒤 아이템 점수를 **최대 코사인**으로 합친다. 시드 1개면 두 모드는 같다. 지표는 popularity와 섞지 않은 **content 단독**이다. 지금 0.0400은 `popularity+content` 가중합이라 여기 기준이 아니다.

### 실험 전 예측

train 유저 90.3%는 리뷰 1개다. 차이는 시드 2개 이상인 소수에서만 난다.

- 시드 1개 유저: `mean` = `per_seed`
- 시드 2개+ 유저: `per_seed`가 `mean`보다 나을 것
- 전체 Recall@10: 소폭이거나 거의 같음
- multi-seed에서 `mean`을 못 이기면 이 수정은 약하다. 기본값은 `mean` 유지

같은 warm valid 200 users. 시드는 서빙과 같이 `history[-5:]`.

| 항목 | 사전 예측 | 실측 |
|------|-----------|------|
| 시드 2개+ 유저 수 (200명 중) | 소수 | 49 / 200 |
| Recall@10 content `mean` (전체) | popularity+content(0.0400)과 별개, 낮을 수 있음 | 0.0450 |
| Recall@10 content `per_seed` (전체) | `mean`과 비슷하거나 소폭 | **0.0525** |
| Recall@10 content `mean` (multi-seed) | `per_seed`보다 낮을 것 | 0.0204 |
| Recall@10 content `per_seed` (multi-seed) | `mean`보다 나을 것 | **0.0510** |

실험 후: 예측이 맞았다. 전체는 0.0450 → 0.0525로 소폭. multi-seed(49명)에서는 0.0204 → 0.0510으로 `mean`이 더 크게 졌다. 시드가 섞이면 평균 쿼리가 의미를 뭉갠다는 가설과 같다. 서빙 기본값은 `retrieval.content_query_mode: per_seed`. 그래도 popularity(0.1689)에는 한참 못 미친다. `popularity+content` 0.0400은 예전 `mean` 가중합 숫자로, hybrid/RRF는 다음 칸이다.

## Ablation 템플릿 (README에 채울 표)

| 구성 | Recall@10 | NDCG@10 | Coverage | p50 ms |
|------|----------|---------|----------|--------|
| popularity | 0.1689 | — | — | — |
| + content FAISS (`mean` 가중합) | 0.0400 | — | — | — |
| content `per_seed` (단독) | 0.0525 | — | — | — |
| iALS (`rating_ge_5`) | 0.1048 | — | — | — |
| two-tower (탈락) | 0.0000 | — | — | — |
| + hybrid CF | — | — | — | — |
| + ranker (선정 모델) | — | — | — | — |
| + MMR | — | — | — | — |
| + RAG + OpenAI explain | — | — | — | cost |

랭커는 DeepFM에 고정하지 않는다. Phase 3에서 부스팅·DL·LTR 후보를 같은 feature/split으로 비교한 뒤, 지표·latency·구현 복잡도를 보고 고른다.

OpenAI는 설명/후보 내 선택에 **항상** 사용한다. 키 없는 fallback 경로는 두지 않는다.

## LLM 평가

- 추천 품질의 주 지표로 LLM 점수를 쓰지 않는다.
- LLM 관련은 **환각률(후보 밖 id)**, 설명 길이, 요청당 비용만 보조 지표로 둔다.
