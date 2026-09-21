# Query labeling

Phase 7 검색 적합성 라벨. **사람이 상품을 보고 재라벨한 세트가 아니다.**  
제목·브랜드·`doc_text`에 특징 토큰이 있는지를 규칙으로 양성을 만든다. 재생성: `python -m scripts.build_query_gold`.

쿼리 목록은 Phase 5 RAG 보조 평가와 같다 (`ml/eval/rag_aux.py`의 `EVAL_QUERIES`). 스펙은 `ml/eval/query_label.py`.

## 쿼리 유형

| 유형 | 의미 | 이 세트의 예 |
|------|------|-------------|
| ingredient | 성분·활성 | niacinamide serum, retinol night cream, vitamin c serum |
| concern | 피부 고민·사용감 | hydrating serum, moisturizer for dry skin, foundation for oily skin |
| product | 제품 유형 | mascara, sheet face mask, mouthwash |
| brand | 브랜드명 | **이 20개에는 없음** (커버리지 공백) |

한 쿼리가 유형을 겹치면 **더 구체적인 쪽**을 고른다. `niacinamide serum`은 ingredient, `hydrating serum`은 concern.

## 양성 (적합)

1. `title`, `brand`, `doc_text`를 소문자로 이어 붙인 텍스트에 스펙의 **모든 토큰 그룹**이 나타난다.
2. 그룹 안은 OR, 그룹 사이는 AND. 예: `hydrating serum` → (`hydrat` 또는 `moistur`) **그리고** `serum`.
3. 부분 문자열 매칭이다. `hydrat`는 hydrating / hydration을 친다.
4. 쿼리당 양성은 최대 40개. 넘으면 제목이 짧은 순 → `item_id` 순으로 자른다. **Precision@10은 이 40개 대비**라서 매칭이 많은 쿼리(body lotion 836건)는 점수가 깎인다. 오탐 CSV는 토큰 규칙을 깨는 top-10만 센다.

이 규칙은 재현 가능하지만 **의도 적합성의 상한이다.** 토큰이 있어도 다른 카테고리이거나, 토큰이 없어도 의미상 맞을 수 있다.

## 애매

라벨하지 않고 운영 메모로만 둔다.

- 제형은 맞는데 고민이 약한 경우: 세럼인데 hydrating 근거가 제목에만 있음
- 제품 경계: serum vs cream vs lotion (`retinol night cream`)
- 마스크: sheet face mask vs hair mask vs sleep mask (`mask`만 있으면 양성에서 탈락하도록 `sheet mask` / `face mask`를 씀)
- oily skin foundation: `oil`이 올리브오일 클렌저에 걸릴 수 있음 → `foundation` 그룹과 AND

## 오탐

`scripts/eval_query.py`에서 오탐 = **FAISS(+MMR) top-10인데 양성 규칙을 통과하지 못한 ASIN.**

흔한 원인:

- 임베딩이 같은 카테고리 이웃을 가져옴 (세럼 쿼리에 크림)
- 단어 일부만 비슷 (serum / serum-infused 바디로션)
- 카탈로그 텍스트가 빈약해서 `doc_text`가 ASIN뿐인 상품이 이웃에 섞임 → [EVAL.md](EVAL.md) 속성 품질

미탐 = 양성 ASIN이 top-10 밖. gold가 쿼리당 최대 40개라 Recall@10은 낮을 수 있다. Precision@10과 함께 본다.

## 금지

- valid/test 리뷰 텍스트로 라벨하지 않음 (메타 `doc_text`만)
- “내가 보기엔 이 세럼이 맞다”는 식의 사후 수정으로 gold를 고치지 않음. 규칙을 바꾸면 스크립트를 다시 돌리고 이 문서의 그룹을 고친다.
- LLM 점수로 적합/부적합을 정하지 않음
