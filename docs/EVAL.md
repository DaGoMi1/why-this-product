# Evaluation

## 원칙

- 오프라인 지표 없이 모델을 README에 올리지 않는다.
- split은 [DATA.md](DATA.md)의 **temporal** 규칙을 따른다.
- 가능하면 세그먼트별 리포트: all / cold-user / cold-item.
- Phase 7 쿼리 라벨은 사람이 재라벨한 GT가 아니다. 규칙: [LABELING.md](LABELING.md).

## 핵심 지표

| 지표 | 용도 | 비고 |
|------|------|------|
| Recall@K (관련) | retrieve 풀 / 최종 리스트 | GT = valid `rating >= 5`. retrieve K=200, 최종 K=10 |
| NDCG@10 (등급) | 리스트 순위 (Recall과 별도) | 관련도 = valid 별점 1~5, 없으면 0. DCG gain = `2^rel - 1` |
| Coverage | 롱테일·다양성 | 추천 unique / catalog |
| Intra-list diversity | MMR 효과 | 리스트 안 1 - 평균 페어 코사인. unique category는 All_Beauty에서 상수 |
| Latency p50/p95 | 서빙 | Phase 6 |

주 문제는 별점 예측(RMSE)이 아니라 **리스트 추천**. 존재 Recall·관련>=4 숫자는 레거시.

## 지표 확정 (rating >= 5 통일)

긍정은 전부 **`rating >= 5`**. pop 카운트, iALS, 랭커 라벨, Recall GT가 같다. 4점은 Recall 정답이 아니다. NDCG만 1~5 등급을 따로 본다.

표본: warm 유저 중 valid 5점이 있는 사람 최대 200명.

세 갈래:

1. Retrieve 단독: 각 채널 top-10 + 풀 Recall@200
2. Retrieve × 랭커: 그 채널 200을 LGBM/XGB/Cat이 재정렬 → top-10 (학습은 pop-200 ge_5 한 번)
3. 부스팅 단독: 카탈로그(히스토리 제외)를 세 랭커가 직접 top-10

### 실험 전 예측

| 항목 | 사전 예측 | 실측 |
|------|-----------|------|
| 단독 retrieve 1위 (R@10 / NDCG@10) | pop top-10. >=4 때(0.2023)보다 Recall은 낮을 것 | **pop 0.1900 / 0.0698** (Recall만 낮아짐) |
| retrieve Recall@200 1위 | pop | **RRF(pop, content) 0.3500** (pop 0.3350. 예측과 어긋남) |
| 채널 풀 재정렬 vs 원래 순서 | 어느 채널이든 재정렬이 더 낮을 것 | 대체로 맞음. 예외: iALS+Cat 0.0500 > iALS 단독 0.0425 |
| 부스팅 단독 vs pop 단독 | 더 낮을 것 (학습이 pop-200) | **맞음.** 최고 Cat 0.0400 / 0.0275 |

실험 후: 최종 리스트는 여전히 **pop top-10**. RRF(pop, content)는 풀@200만 이기고 @10은 0.1100으로 pop에 못 미친다. 재정렬·카탈로그 단독은 전부 pop 이하. **서빙은 popularity 단일 (`min_rating=5`). `use_hybrid`·`use_ranker` 기본 off.**

랭커 학습(ge_5 시퀀스, pop-200): train 유저 254,641 / 5점 유저 156,561 / 2개+ 14,790 / 양성-in-pop **8,894** (dropped 39.9%). 행 1,769,599.

### Phase 3 종료

선정: **popularity 단일** (`min_rating=5`). 랭커는 서빙 기본이 아니다 (`use_ranker` off).

탈락 (최종 Recall@10 / NDCG@10, 관련 >=5):

- iALS 0.0425 / 0.0213, content `per_seed` 0.0550 / 0.0421
- RRF 최고 @10은 (pop, iALS) 0.1125 / 0.0451. @200 1위 RRF(pop, content) 0.3500은 풀 지표일 뿐
- 재정렬 최고 RRF(pop, content)+CatBoost 0.0575 / 0.0364
- 카탈로그 단독 최고 CatBoost 0.0400 / 0.0275

하지 않은 것: DeepFM, LambdaMART, 채널별 랭커 재학습, RMSE. 같은 희소 라벨·pop-200 후보에서 순서를 흔드는 모델이라 격자를 뒤집는 근거가 없다. 이 결론은 All_Beauty 리스트 추천 본체에만 해당한다.

## Phase 4: MMR + 신규 상품 fallback + cold 세그먼트

pop top-10 위에 다양성 리랭크만 얹는다. 랭커는 다시 안 돌린다.

풀: pop-200 + (히스토리 있으면 content `per_seed` 중 **train 미등장** 아이템 최대 20). MMR greedy, `λ_rel = 1 - lambda_diversity`. sim = 임베딩 코사인, 없으면 같은 카테고리=1.

서빙 규칙: ILD 또는 unique category가 pop보다 오르면 Recall이 조금 내려도 MMR on. ILD가 안 오르거나 Recall이 절대 0.04 이상 떨어지면(0.1900→0.15 미만) off. 후보 λ: 0.3 / 0.5 / 0.7.

표본: warm valid 5점 최대 200. 추가로 valid cold-user·cold-item GT 세그먼트 최대 200.

### 실험 전 예측

| 항목 | 사전 예측 | 실측 |
|------|-----------|------|
| MMR Recall@10 / NDCG@10 | pop(0.1900 / 0.0698)보다 낮을 것 | λ=0.3은 **0.1950 / 0.0844**로 오름. λ=0.5는 0.1700 / 0.0747. λ=0.7은 0.0275로 붕괴 |
| unique category · ILD | 오를 것 | unique cat는 전부 **1.0** (카탈로그 카테고리 단일). ILD는 0.759 → 0.799 / 0.864 / 0.913 |
| Coverage | 조금 오를 것 (cold 슬롯) | pop 0.00037 → λ=0.5 **0.00055**. pop+cold@10은 pop과 동일 |
| cold-user | 히스토리 없음 → pop과 같고 MMR만 다름 | pop 0.0860 / 0.0431 vs MMR0.5 0.0785 / 0.0396. ILD 0.757 → 0.800 |
| cold-item GT | pop은 0에 가깝고, content fallback이 조금이라도 칠 수 있음 | n=17 전부 **0**. fallback이 top-10에 못 들어옴 |

실험 후: unique category는 이 데이터에서 신호가 없다. ILD는 λ가 클수록 오른다. 규칙(ILD 상승 ∧ Recall 하락 < 0.04 ∧ Recall≥0.15)을 통과한 건 0.3과 0.5. 그중 ILD가 더 높은 **λ=0.5**를 서빙한다. 0.7은 Recall 0.0275로 탈락. 신규 상품 슬롯은 @10에 안 보여 Coverage 이득은 MMR 재정렬 몫이다.

### warm @10 (관련 >=5)

| 구성 | Recall@10 | NDCG@10 | Coverage | ILD | uniqCat |
|------|----------|---------|----------|-----|---------|
| pop | 0.1900 | 0.0698 | 0.00037 | 0.7585 | 1.0 |
| pop+cold (풀 앞 10) | 0.1900 | 0.0698 | 0.00037 | 0.7585 | 1.0 |
| MMR λ=0.3 | 0.1950 | 0.0844 | 0.00040 | 0.7993 | 1.0 |
| MMR λ=0.5 | **서빙** 0.1700 | 0.0747 | 0.00055 | 0.8638 | 1.0 |
| MMR λ=0.7 | 0.0275 | 0.0113 | 0.00068 | 0.9132 | 1.0 |

### 세그먼트 @10

| 세그먼트 | 구성 | n | Recall@10 | NDCG@10 | ILD |
|----------|------|---|----------|---------|-----|
| cold-user | pop | 200 | 0.0860 | 0.0431 | 0.7570 |
| cold-user | MMR 0.5 | 200 | 0.0785 | 0.0396 | 0.7997 |
| cold-item GT | pop / pop+cold / MMR 전부 | 17 | 0.0000 | 0.0000 | — |

**서빙: `user_id`는 pop-200 → MMR `lambda_diversity=0.5` (`use_mmr` 기본 on). `use_content` 기본 off. `query`는 검색이라 pop을 안 섞고 content FAISS → MMR.**

## MVP 스모크 (`scripts/eval_smoke.py`)

1. popularity `min_rating=5`
2. warm valid 5점 최대 200: pop@10 vs pop+cold vs MMR λ 0.3/0.5/0.7 (Recall / NDCG / Coverage / ILD / unique cat)
3. cold-user · cold-item 세그먼트 (각 최대 200)
4. 실패 시 non-zero exit


### retrieve Recall@200 (관련 >=5)

| 구성 | Recall@200 |
|------|------------|
| popularity (`min_rating=5`) | 0.3350 |
| iALS `rating_ge_5` | 0.1150 |
| content `per_seed` | 0.0875 |
| RRF(pop, content) | **0.3500** |
| RRF(pop, iALS) | 0.3175 |
| RRF(pop, iALS, content) | 0.3400 |
| two-tower | 0.0025 |

풀 1위는 RRF(pop, content). 최종 리스트 승자와 같지 않다.

### 단독 top-10 (관련 >=5 Recall / 등급 NDCG)

| 구성 | Recall@10 | NDCG@10 |
|------|----------|---------|
| popularity | **0.1900** | **0.0698** |
| iALS | 0.0425 | 0.0213 |
| content `per_seed` | 0.0550 | 0.0421 |
| RRF(pop, content) | 0.1100 | 0.0496 |
| RRF(pop, iALS) | 0.1125 | 0.0451 |
| RRF(pop, iALS, content) | 0.0825 | 0.0476 |
| catalog + LightGBM | 0.0050 | 0.0032 |
| catalog + XGBoost | 0.0050 | 0.0032 |
| catalog + CatBoost | 0.0400 | 0.0275 |

### retrieve × 랭커 top-10

| 풀 | +LGBM R/N | +XGB R/N | +Cat R/N |
|----|-----------|----------|----------|
| popularity | 0.0150 / 0.0108 | 0.0200 / 0.0188 | 0.0250 / 0.0144 |
| iALS | 0.0250 / 0.0112 | 0.0050 / 0.0032 | 0.0500 / 0.0298 |
| content | 0.0200 / 0.0095 | 0.0050 / 0.0032 | 0.0400 / 0.0280 |
| RRF(pop, content) | 0.0250 / 0.0113 | 0.0100 / 0.0046 | 0.0575 / 0.0364 |
| RRF(pop, iALS) | 0.0250 / 0.0125 | 0.0100 / 0.0071 | 0.0400 / 0.0251 |
| RRF(pop, iALS, content) | 0.0200 / 0.0091 | 0.0050 / 0.0036 | 0.0525 / 0.0313 |

재정렬 최고는 RRF(pop, content)+CatBoost 0.0575 / 0.0364. pop 단독 0.1900 / 0.0698 미달.

## 이전 리그: 관련 >=4 (통일 전)

pop 카운트·Recall이 >=4, 랭커는 존재 라벨. 승자 결정에 쓰지 않는다.

| 구성 | Recall@10 | NDCG@10 |
|------|----------|---------|
| pop 단일 | 0.2023 | 0.0699 |
| popularity + LightGBM | 0.0450 | 0.0199 |
| popularity + XGBoost | 0.0525 | 0.0215 |
| popularity + CatBoost | 0.0600 | 0.0271 |
| retrieve@200 pop | 0.3527 | — |

### 레거시: 존재 Recall (리뷰만 있으면 양성)

아래는 1점도 정답이던 이전 리그다. 승자 결정에 쓰지 않는다.

| 구성 | Recall@10 |
|------|----------|
| popularity (단일) | 0.1689 |
| popularity+content (가중합, `per_seed`) | 0.0525 |
| iALS `all` | 0.0475 |
| iALS `rating_ge_3` | 0.0450 |
| iALS `rating_ge_4` | 0.0675 |
| iALS `rating_ge_5` | 0.1048 |
| two-tower (history mean-pool) | 0.0000 |
| content `mean` (단독) | 0.0450 |
| content `per_seed` (단독) | 0.0525 |
| RRF(pop, content) | 0.0875 |
| RRF(pop, iALS) | 0.1114 |
| RRF(pop, iALS, content) | 0.0817 |
| LightGBM ranker (pop 200 재정렬) | 0.0813 |
| XGBoost ranker (pop 200 재정렬) | 0.0938 |
| CatBoost ranker (pop 200 재정렬) | 0.0650 |

이 표에서는 popularity 단일이 강하다. 랭커는 pop 순서를 흔들어 더 낮아졌다. 서빙 기본은 popularity.

## Funnel vs 단일 pop (retrieve@200 승자 + 기존 랭커)

질문: retrieve를 같은 K=200으로 고른 뒤, **1위 풀 200개**를 지금 있는 LightGBM / XGBoost / CatBoost로 top-10 하면, pop 단일 Recall@10(0.1689)을 넘나.

랭커는 재학습하지 않는다. pop-200에서 배운 모델을 승자 풀에 그대로 얹는다.

### 실험 전 예측

같은 warm valid 200.

| 항목 | 사전 예측 | 실측 |
|------|-----------|------|
| retrieve Recall@200 1위 | pop 또는 RRF(pop, iALS) | **popularity (0.3468)** |
| pop Recall@200 | @10(0.1689)보다 높을 것 (랭커 천장) | **0.3468** |
| 승자가 pop이면 펀넬 3줄 | 기존과 같음 (XGB 0.0938 등), pop 단일보다 낮음 | **같음. 최고 XGB 0.0938 < 0.1689** |
| 승자가 pop이 아니면 펀넬 3줄 | 천장은 그 Recall@200. 재학습 없는 랭커가 pop@10을 넘을 가능성은 낮음 | 해당 없음 (승자=pop) |

실험 후: retrieve@200도 pop이 1위다. RRF 세 조합은 0.308~0.313으로 pop(0.3468)에 못 미친다. iALS·content 단독 @200은 ~0.09. (예전 iALS@10 0.1048은 유저 벡터가 없으면 **pop fallback**을 섞은 값. 이번 retrieve 리그는 fallback 없이 채널만 본다.) pop@200 0.3468은 @10 0.1689의 약 두 배라 랭커 천장은 있다. 기존 부스팅은 그 순서를 흔들어 전부 하락. **세 줄 모두 pop 단일 이하 → 멀티스테이지 비활성. 서빙은 popularity 단일.**

### retrieve Recall@200 (실측)

| 구성 | Recall@200 | 비고 |
|------|------------|------|
| popularity | **0.3468** | 승자 |
| iALS `rating_ge_5` | 0.0892 | pop fallback 없음 |
| content `per_seed` | 0.0900 | |
| RRF(pop, content) | 0.3081 | |
| RRF(pop, iALS) | 0.3118 | |
| RRF(pop, iALS, content) | 0.3131 | RRF 중 최고, pop 미달 |
| two-tower | 0.0075 | |

승자: **popularity**

### funnel Recall@10 (승자 풀=pop 200 + 기존 랭커)

| 구성 | Recall@10 |
|------|----------|
| pop 단일 (비교 기준) | **0.1689** |
| popularity + LightGBM | 0.0813 |
| popularity + XGBoost | 0.0938 |
| popularity + CatBoost | 0.0650 |

## iALS implicit 라벨 (실험 전 예측 → 실측)

explicit 평점(1~5)은 회귀하지 않는다. train 상호작용만 양성으로 바꿀 때 기준을 나눈다.
`0`은 네거티브 샘플링이 아니다. 임계 미달 리뷰는 양성에서 **빼는 것**이고, 미관측 칸은 iALS가 약한 부정으로 취급한다.

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

실험 후: 예측이 맞았다. 전체는 0.0450 → 0.0525로 소폭. multi-seed(49명)에서는 0.0204 → 0.0510으로 `mean`이 더 크게 졌다. 시드가 섞이면 평균 쿼리가 의미를 뭉갠다는 가설과 같다. 서빙 기본값은 `retrieval.content_query_mode: per_seed`. 그래도 popularity(0.1689)에는 한참 못 미친다. `popularity+content` 0.0400은 `mean` 가중합 숫자다.

## Hybrid retrieve (RRF)

RRF는 점수가 아니라 **등수**만 더한다.

`RRF(i) = sum_c 1 / (60 + rank_c(i))`

리스트에 없는 채널은 0. two-tower는 넣지 않는다. content는 `per_seed`, iALS는 `rating_ge_5`.

가중합 `popularity+content`는 0.1689 → 0.0400으로 깎였다. 이번엔 순위만 합친다.

### 실험 전 예측

같은 warm valid 200 users.

| 구성 | 사전 예측 | 실측 |
|------|-----------|------|
| RRF(pop, content) | 가중합 0.0400보다는 나을 것. popularity(0.1689)를 넘기기는 어렵다 | **0.0875** |
| RRF(pop, iALS) | iALS(0.1048)보다는 나을 것. pop과 경합 | **0.1114** |
| RRF(pop, iALS, content) | 후보 풀은 넓어짐. content가 헤드를 밀면 pop보다 낮을 수 있음 | **0.0817** |

실험 후: 예측이 맞았다. RRF(pop, content)는 가중합(지금 `per_seed` 기준 0.0525, 예전 `mean` 0.0400)보다 낫고, RRF(pop, iALS)는 iALS 단독(0.1048)보다 낫다. 3채널은 content가 헤드를 밀어 0.0817로 내려갔다. **세 조합 모두 popularity(0.1689)를 못 넘겼다.** 서빙 기본은 popularity (`use_hybrid` 기본 false). 합칠 때는 가중합이 아니라 RRF를 쓴다. API는 `use_hybrid: true`로 3채널 RRF를 켤 수 있다.

## Ranker (LightGBM, popularity 후보 재정렬)

카탈로그 전체가 아니라 **popularity top-200**만 다시 줄 세운다. valid로 학습하지 않는다. 라벨은 train 안에서만 만든다.

- 대상: train 상호작용 2개 이상 유저
- 히스토리 = 마지막 제외, 양성 = 마지막 아이템 (존재 여부)
- 후보에 양성이 없으면 그 유저는 버림
- 음성 = 같은 후보의 나머지 (hard negative = 이미 인기라서 들어온 아이템)

피처는 train 통계만. 타깃 리뷰 텍스트·valid 평점 금지.

`log_pop_count`, `item_n`, `item_mean_rating`, `user_n`, `pop_rank`, `content_max_sim`, `ials_score`, `same_brand`, `same_category`

### 실험 전 예측

같은 warm valid 200. retrieve는 popularity 200 → 랭커 top-10.

| 항목 | 사전 예측 | 실측 |
|------|-----------|------|
| train 2개+ 유저 중 양성이 pop 후보에 있는 비율 | 작을 것 | 13,322 / 24,761 = **53.8%** (전체 train 유저의 5.2%) |
| 학습 페어 수 | 소수 유저 × ~200 | 13,322 유저 × 200 = **2,651,592** (양성 13,322 / 음성 2,638,270) |
| Recall@10 LightGBM | popularity(0.1689)를 못 넘을 가능성이 큼 | **0.0813** |

실험 후: 학습에 쓴 2개+ 유저는 전체의 9.7%(24,761)이고, 그중 양성-in-pop은 53.8%라 “아주 작다”기보다 **쓸 수 있는 유저가 적다**. Recall@10은 예측대로 popularity(0.1689)를 못 넘겼다. two-tower(0.0)보다는 낫고, iALS(0.1048)·RRF(pop, iALS)(0.1114)보다 낮다. pop 순서를 흐트러뜨린 대가. **서빙 기본은 popularity. `use_ranker`는 플래그만.**

## Ranker boosting (XGBoost · CatBoost, 같은 테이블)

피처·라벨·후보(pop 200)는 LightGBM과 **같다**. 트리 구현체만 바꾼다. 예산 `n_estimators=200`, `learning_rate=0.05`, `scale_pos_weight`. LightGBM `num_leaves=31`에 맞춰 XGBoost·CatBoost는 `max_depth=5`.

### 실험 전 예측

같은 warm valid 200. retrieve는 popularity 200 → 각 랭커 top-10.

| 구성 | 사전 예측 | 실측 |
|------|-----------|------|
| LightGBM (재측정, 테이블 동일) | 기존 0.0813과 같거나 거의 같을 것 | **0.0813** |
| XGBoost | LightGBM(0.0813)과 비슷할 것. pop(0.1689)은 못 넘을 가능성이 큼 | **0.0938** |
| CatBoost | LightGBM·XGBoost와 비슷할 것. 셋 사이 차이는 작을 것 | **0.0650** |

실험 후: 테이블은 그대로(13,322 유저 / 2,651,592행). LightGBM은 재측정이 같았다. XGBoost가 셋 중 최고(0.0938)지만 popularity(0.1689)와 iALS(0.1048)를 못 넘긴다. CatBoost는 0.0650으로 가장 낮다. “비슷할 것”은 방향은 맞았고, 셋 사이 간격은 예상보다 조금 크다. **서빙 기본은 popularity. `use_ranker`는 플래그만.**

## Ablation 템플릿 (관련 >=5 Recall / 등급 NDCG — 승자 결정용)

| 구성 | Recall@10 | NDCG@10 | Coverage | p50 ms |
|------|----------|---------|----------|--------|
| popularity (`min_rating=5`) | 0.1900 | 0.0698 | 0.00037 | — |
| + MMR λ=0.3 | 0.1950 | 0.0844 | 0.00040 | — |
| + MMR λ=0.5 (서빙) | 0.1700 | 0.0747 | 0.00055 | — |
| + MMR λ=0.7 | 0.0275 | 0.0113 | 0.00068 | — |
| RRF(pop, content) retrieve@10 | 0.1100 | 0.0496 | — | — |
| RRF(pop, content) retrieve@200 | 0.3500 | — | — | — |
| iALS (`rating_ge_5`) retrieve@10 | 0.0425 | 0.0213 | — | — |
| content `per_seed` retrieve@10 | 0.0550 | 0.0421 | — | — |
| + CatBoost (RRF pop+content 재정렬) | 0.0575 | 0.0364 | — | — |
| catalog + CatBoost | 0.0400 | 0.0275 | — | — |
| + RAG + OpenAI explain | — | — | — | 5448 ms (~$0.00032/req, k=5) |

Phase 3 본체 승자는 pop 순서. Phase 4 서빙은 pop-200 + MMR `lambda_diversity=0.5`. @200 풀 1위 RRF는 top-10 승자가 아니다.

존재(1점도 양성)·관련>=4 숫자는 위 레거시 표. DeepFM은 돌리지 않고 Phase 3를 닫는다.

OpenAI는 설명/후보 내 선택에 **항상** 사용한다. 키 없는 fallback 경로는 두지 않는다.

## Phase 5: RAG 설명 (보조 지표)

LLM은 추천 Recall을 올리지 않는다. 후보 `item_ids` 안에서만 한두 문장 이유를 쓴다. 청크는 **train 메타 + train 리뷰**만 (valid 누수 없음). retrieve FAISS와 인덱스를 분리한다. 키 없으면 `/api/explain`은 503.

표본: 영어 쿼리 20 + 데모 `user_id` 5, recommend `k=5`, explain `select_k=0` (UI 기본과 같음). 사유 125건. 주 지표와 섞지 않는다. 예전 스모크는 쿼리 2 × k=10 × `select_k=3`에서 환각 0/2·사유 20/20이었다.

### 실험 전 예측

| 항목 | 사전 예측 | 실측 |
|------|-----------|------|
| 환각률 (후보 밖 ASIN, reason 단위) | 프롬프트·id 필터로 **0** | **0.0000** (0/125 reasons, 0/25 requests) |
| 가격 ungrounded | 프롬프트 금지. 코드 가드는 없음 | **0/125** (언급 0) |
| 재고 ungrounded | 프롬프트 금지. 코드 가드는 없음 | **0/125** (언급 0) |
| 설명 길이 | 한두 문장 (대략 40–120자) | **66.4자** (125/125 reasons, empty 0) |
| p50 latency (explain 전체) | 수 초 (OpenAI, k=5) | **5.448s** (직렬: 후보 5개, ASIN당 1호출, select_k=0) |
| 요청당 비용 | gpt-4o-mini 기준 수 센트 미만 | **~$0.00032/요청** (25요청 합 $0.008040, 27822+6444 tok) |
| 키 없음 | 503, fallback 없음 | **503** (`POST /api/explain`) |

실험 후: `scripts/eval_rag.py`. 한 JSON에 N개를 맡기면 사유가 비는 경우가 있어 **상품마다 호출**한다. 후보 밖 ASIN·스니펫에 없는 가격·재고 언급은 이 표본에서 0. latency·비용은 왕복 수만큼 오르고 추천 Recall과 섞지 않는다. 키 없으면 설명 경로가 기동하지 않는다. 단계별 retrieve/rank/rerank 분해는 Phase 6 표를 쓴다.

직렬 호출이 LLM 대기의 합이 되므로, 이후 **건별 호출은 유지한 채 ThreadPool로 병렬화**했다 (`ml/rag/llm.py`). 토큰·비용은 같고 대기만 겹친다. 재측정은 Phase 6.

## Phase 6: 요청 latency (보조)

Recall과 섞지 않는다. `POST /api/recommend`는 `timings_ms.retrieve` / `rank` / `rerank`, `POST /api/explain`은 `timings_ms.rag`. 서빙 기본에서 rank는 0.

표본: query `hydrating serum`, k=5, `use_mmr=true`, 이어서 같은 5개 explain (`select_k=0`). `scripts/bench_explain_latency.py` (explain 3회).

| 단계 | 경로 | ms |
|------|------|----|
| retrieve | content FAISS | **70.6** |
| rank | 서빙 off | **0.0** |
| rerank | MMR λ=0.5 | **21.2** |
| recommend total | | **91.8** |
| rag (직렬, 이전) | 스니펫 + OpenAI ASIN당 1호출 순차 | **7885.6** |
| rag (병렬, 현재) | 동일 호출을 ThreadPool 병렬 | **p50 1401.6** (3회: 6624 / 1401 / 1109). cold 첫 회 제외 시 ~1.1–1.4s |
| 요청당 비용 | gpt-4o-mini | **~$0.00030** (토큰량은 직렬과 동일 규모) |

설명 지연은 여전히 거의 전부 OpenAI다. 병렬화로 **직렬 합 대기 → max(개별 대기)** 에 가깝게 줄였다. 추천 단계 합은 0.1초 전후. GitHub Actions는 넣지 않았다 (Phase 7도 CI는 범위 밖).

## Phase 7: 검색 운영 (라벨 · 미탐/오탐 · 속성 품질)

유저 Recall@10과 섞지 않는다. 서빙은 그대로 **query = content FAISS + MMR**. BM25/RRF·`user_id` 제거는 하지 않음.

Gold: `python -m scripts.build_query_gold` → `data/eval/query_gold.jsonl`. 제목·브랜드·`doc_text` 특징 토큰 AND, 쿼리당 최대 40. 프로토콜 [LABELING.md](LABELING.md).

검색: `python -m scripts.eval_query`. k=10, 전략 `content+mmr`.

- Recall@10 / NDCG@10 / Precision@10: **kept gold(최대 40)** 대비
- 오탐(fp): top-10인데 **토큰 규칙 자체**를 통과하지 못한 ASIN (cap 밖이지만 규칙은 맞는 상품은 fp가 아님)
- 미탐: kept gold가 top-10 밖. gold가 40이면 Recall 상한은 10/40=0.25

| 지표 | 실측 |
|------|------|
| mean Recall@10 | 0.0973 |
| mean Precision@10 | 0.3600 |
| mean NDCG@10 | 0.3823 |
| 미탐 건수 | 692 |
| 오탐 건수 | 62 |
| 쿼리 수 | 20 |

성분 쿼리(retinol night cream P@10 0.90, hyaluronic acid serum 0.80, mouthwash 0.80)가 고민·범용 유형보다 낫다. `hydrating serum` / `moisturizer for dry skin` / `body lotion`은 kept gold 대비 P@10=0에 가깝다. body lotion은 카탈로그 매칭 836개인데 gold는 제목 짧은 40개만 남겨 **캡 아티팩트**가 크다(fp는 1건뿐). hydrating serum은 top-10 중 9건이 토큰 규칙을 깨서 진짜 오탐에 가깝다 (세럼이 아니거나 hydrat/moistur가 없음).

CSV: `data/eval/query_misses.csv`, `data/eval/query_false_positives.csv`. Phase 7 시점 서빙은 content+MMR만 (lexical gate 없음).

### 카탈로그 속성 품질

`python -m scripts.report_catalog_quality`. 상품 32,488.

| 항목 | n | 비율 |
|------|---|------|
| title 공백 | 1 | ~0 |
| brand 공백 | 15,570 | 0.479 |
| description 공백 | 17,670 | 0.544 |
| price 결측 | 21,141 | 0.651 |
| doc_text가 ASIN뿐 | 2 | ~0 |
| unique category | **1** (`All Beauty`) | — |
| gap 플래그 ≥1 행 | 26,046 | 0.802 |

Phase 4 unique category 상수와 같다. 속성 택소노미가 거의 없어 검색은 제목·설명 텍스트에 의존한다. 요약 `data/eval/catalog_quality.csv`. 행 단위 갭 목록은 로컬 `catalog_quality_gaps.csv` (gitignore, ~2.8MB).

### 설명 스니펫

쿼리–청크 lexical overlap + dense 순위. 쿼리 토큰이 있는 리뷰를 메타보다 우선. LLM은 스니펫에 없는 성분/효과 금지. UI는 사유 아래 `[meta]` / `[review]` 인용. 환각 보조 지표는 Phase 5와 같음 (`eval_rag`).

## Phase 8: 운영 루프 (검색 lexical gate)

**한 축만 변경:** 검색. 스니펫·프롬프트 코드는 그대로.  
`QUERY_SPECS`에 있는 쿼리만 FAISS 풀에 gold와 같은 토큰 AND 게이트 → 그다음 MMR → top-k (`content+mmr+lex`). 스펙 밖 자유 쿼리는 필터 없음. 통과분이 k 미만이면 짧은 리스트를 그대로 반환(백필 없음).

재측정: `python -m scripts.eval_query` / `python -m scripts.eval_rag`.

### 검색 before → after

| 지표 | Phase 7 (content+mmr) | Phase 8 (content+mmr+lex) |
|------|------------------------|----------------------------|
| mean Recall@10 | 0.0973 | **0.1327** |
| mean Precision@10 | 0.3600 | **0.5450** |
| mean NDCG@10 | 0.3823 | **0.5123** |
| 미탐 | 692 | 664 |
| 오탐 (토큰 규칙 위반) | 62 | **0** |

오탐 0은 게이트가 eval의 `is_positive`와 동일해서, 규칙 위반 상품이 top-k에 안 들어오기 때문이다. kept-gold(캡 40) 밖이지만 규칙은 맞는 상품은 여전히 fp가 아니다.

`hydrating serum`: Phase 7 top-10 중 fp 9 → Phase 8 **fp 0**, P@10 **0.20** (kept gold 대비; Recall@10 0.05).

### intent별 (Phase 8)

| intent | n | R@10 | P@10 | NDCG@10 | miss | fp |
|--------|---|------|------|---------|------|-----|
| concern | 6 | 0.0958 | 0.3833 | 0.3867 | 217 | 0 |
| ingredient | 5 | 0.1983 | 0.8600 | 0.7223 | 132 | 0 |
| product | 9 | 0.1209 | 0.4778 | 0.4793 | 315 | 0 |

ingredient가 여전히 가장 강하고, concern은 개선됐지만 상대적 약세. `niacinamide serum`은 게이트 후 풀이 얇아 **n=1**만 반환된 적이 있음(짧은 리스트 허용).

### RAG 보조 (재측정)

검색 후보가 바뀌어 설명 대상 ASIN이 달라짐. 스니펫·프롬프트 미변경. 25요청, k=5, select_k=0, 병렬 explain.

| 항목 | Phase 5/6 근처 | Phase 8 재측정 |
|------|----------------|----------------|
| reasons / empty | 125 / 0 | **121** / 0 (일부 쿼리 top-k < 5) |
| 후보 밖 ASIN | 0 | **0** |
| 가격·재고 ungrounded | 0 | **0** |
| mean reason chars | 66.4 | **64.1** |
| p50 latency (explain) | ~1.4s (병렬) | **1.282s** |
| 요청당 비용 | ~$0.00030–0.00032 | **~$0.00036** (합 $0.009017 / 25) |

## LLM 평가

- 추천 품질의 주 지표로 LLM 점수를 쓰지 않는다.
- LLM 관련은 **후보 밖 ASIN**, 스니펫에 없는 가격·재고 언급, 설명 길이, 요청당 비용만 보조 지표로 둔다.
