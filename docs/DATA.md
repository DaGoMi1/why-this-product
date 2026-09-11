# Data

## 데이터셋

| 항목 | 값 |
|------|-----|
| 이름 | Amazon Reviews 2018 — **All_Beauty** |
| 용도 | 유저–아이템 상호작용 + 상품 텍스트(RAG) |
| 선택 이유 | 로컬에서 다루기 쉬운 크기, 메타/리뷰로 설명 가능 |

원본은 UCSD Amazon Reviews 2018 배포본(리뷰 JSON + 메타 JSON)을 사용합니다.  
다운로드 URL·파일명은 `scripts/download_data.py` 구현 시 이 문서에 고정합니다.

라이선스·이용 조건을 확인하고, raw는 Git에 올리지 않습니다 (`data/raw/` gitignore).

## 목표 스키마 (processed)

### interactions

| 컬럼 | 타입 | 설명 |
|------|------|------|
| user_id | str | reviewerID |
| item_id | str | asin |
| rating | float | 1–5 |
| timestamp | int | unix time |
| review_text | str \| null | 리뷰 본문 (RAG 근거용, 선택) |

### items

| 컬럼 | 타입 | 설명 |
|------|------|------|
| item_id | str | asin |
| title | str | 상품명 |
| brand | str \| null | 브랜드 |
| category | str \| null | 카테고리 경로 요약 |
| description | str \| null | 설명 텍스트 |
| price | float \| null | 있으면 |

임베딩 문서 예: `"{title}. {brand}. {description}"` (+ 상위 리뷰 스니펫은 RAG 단계에서 별도).

## Temporal split (필수)

랜덤 split은 사용하지 않습니다.

1. 상호작용을 `timestamp` 오름차순 정렬
2. 전역 시간 기준 train / valid / test 구간 분할  
   기본 비율: **0.8 / 0.1 / 0.1** (`configs/mvp.yaml`)
3. train에 등장하지 않은 user/item은 cold-start 세그먼트로 따로 리포트 (Phase 4)

**누수 방지**

- 테스트 구간의 리뷰·평점을 학습 feature로 쓰지 않음
- 인기도·아이템 통계는 **train 구간만**으로 계산
- content 임베딩은 메타데이터 중심; 리뷰를 넣을 경우 train 시점 이전 리뷰만

## 디렉터리

```text
data/raw/          # 다운로드 원본 (gitignore)
data/processed/    # parquet/jsonl, 작은 샘플만 커밋 가능
  interactions_*.parquet
  items.parquet
  faiss_index/     # gitignore (로컬 빌드)
```

## 재현

```bash
# Phase 1에서 구현
python -m scripts.download_data
python -m scripts.prepare_splits
python -m scripts.build_faiss
```
