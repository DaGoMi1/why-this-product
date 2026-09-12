# Fields

Amazon Reviews 2018 **All_Beauty**에서 쓰는 필드와, 파이프라인이 버리는 원본 필드.

매핑 구현: [`scripts/prepare_splits.py`](../scripts/prepare_splits.py)  
데이터셋·split: [DATA.md](DATA.md)

## Processed — interactions

파일: `data/processed/interactions_{train,valid,test}.parquet`

| 필드 | 타입 | 원본 | 의미 |
|------|------|------|------|
| `user_id` | str | `reviewerID` | 리뷰 작성자. 유저 식별자 |
| `item_id` | str | `asin` | 상품 ASIN |
| `rating` | float | `overall` | 별점 1–5 (이 데이터는 정수) |
| `timestamp` | int | `unixReviewTime` | 리뷰 시각 (unix seconds). temporal split 기준 |
| `review_text` | str \| null | `reviewText` | 리뷰 본문. RAG 근거용. 학습 feature로 쓰지 않음 |

키: `(user_id, item_id, timestamp)` 중복은 첫 행만 남긴다.  
`reviewerID` / `asin` / `unixReviewTime`이 없는 행은 버린다.

## Processed — items

파일: `data/processed/items.parquet`

| 필드 | 타입 | 원본 | 의미 |
|------|------|------|------|
| `item_id` | str | `asin` | 상품 ASIN |
| `title` | str | `title` | 상품명. 없으면 `""` |
| `brand` | str \| null | `brand` | 브랜드 |
| `category` | str \| null | `category`, 없으면 `main_cat` | 카테고리. 리스트면 공백으로 이어 붙임 |
| `description` | str \| null | `description` | 상품 설명. 리스트면 이어 붙임 |
| `price` | float \| null | `price` | 가격. `"$12.99"` 같은 문자열에서 첫 숫자만 파싱 |
| `doc_text` | str | (파생) | 임베딩·content FAISS용. `title. brand. description`. 셋 다 비면 `item_id` |

`asin`이 없는 메타 행은 버린다. 같은 `asin`은 첫 행만 남긴다.  
MVP는 메타에 있는 상품과의 상호작용만 남긴다.

## 원본 → processed

```text
reviewerID      → user_id
asin            → item_id
overall         → rating
unixReviewTime  → timestamp
reviewText      → review_text
title           → title
brand           → brand
category|main_cat → category
description     → description
price           → price
title+brand+description → doc_text
```

## Raw reviews (`All_Beauty.json.gz`)

줄 단위 JSON. 키가 행마다 다를 수 있다.

| 필드 | 사용 | 의미 |
|------|------|------|
| `reviewerID` | 사용 | 유저 ID |
| `asin` | 사용 | 상품 ID |
| `overall` | 사용 | 별점 1–5 |
| `unixReviewTime` | 사용 | 리뷰 unix 시각 |
| `reviewText` | 사용 | 리뷰 본문 |
| `verified` | 버림 | Amazon 구매 인증 여부 |
| `reviewTime` | 버림 | 읽기용 날짜 문자열 (`"01 5, 2018"`). `unixReviewTime`과 중복 |
| `reviewerName` | 버림 | 표시 이름 |
| `summary` | 버림 | 리뷰 한 줄 제목 |
| `style` | 버림 | 옵션 (색/사이즈 등). 일부 행만 있음 |
| `vote` | 버림 | 도움이 됨 수 (문자열일 수 있음) |
| `image` | 버림 | 리뷰 첨부 이미지 URL |

## Raw metadata (`meta_All_Beauty.json.gz`)

| 필드 | 사용 | 의미 |
|------|------|------|
| `asin` | 사용 | 상품 ID |
| `title` | 사용 | 상품명 |
| `brand` | 사용 | 브랜드 |
| `category` | 사용 | 카테고리 경로 (리스트인 경우 많음) |
| `main_cat` | `category` 없을 때 | 메인 카테고리 |
| `description` | 사용 | 상품 설명 (리스트인 경우 많음) |
| `price` | 사용 | 가격 문자열 또는 숫자 |
| `tech1` / `tech2` | 버림 | 스펙 HTML |
| `fit` | 버림 | 핏 정보 |
| `feature` | 버림 | 불릿 스펙 |
| `rank` | 버림 | 판매 순위 문자열 |
| `also_buy` | 버림 | 함께 구매 ASIN 목록 |
| `also_view` | 버림 | 함께 본 ASIN 목록 |
| `similar_item` | 버림 | 유사 상품 HTML |
| `details` | 버림 | 상세 키-값 |
| `date` | 버림 | 등록일 등 |
| `imageURL` / `imageURLHighRes` | 버림 | 상품 이미지 URL |
