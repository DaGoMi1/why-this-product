"""공통 경로와 MVP 설정 로드"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return ROOT


@lru_cache(maxsize=1)
def load_mvp_config() -> dict[str, Any]:        # MVP 설정 로드

    load_dotenv(ROOT / ".env")                  # .env 파일 로드    
    path = ROOT / "configs" / "mvp.yaml"        # mvp.yaml 파일 경로
    with path.open(encoding="utf-8") as f:      # mvp.yaml 파일 열기
        cfg = yaml.safe_load(f)                 # mvp.yaml 파일 로드
    
    data = cfg.setdefault("data", {})           # data 설정 추가

    data["raw_dir"] = os.getenv(                # raw_dir 설정 추가
        "DATA_RAW_DIR", data.get("raw_dir", "data/raw")
    )

    data["processed_dir"] = os.getenv(          # processed_dir 설정 추가
        "DATA_PROCESSED_DIR", data.get("processed_dir", "data/processed")
    )

    rag = cfg.setdefault("rag", {})             # rag 설정 추가

    rag["embedding_model"] = os.getenv(         # embedding_model 설정 추가
        "EMBEDDING_MODEL", rag.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    )

    rag["faiss_index_dir"] = os.getenv(         # faiss_index_dir 설정 추가
        "FAISS_INDEX_DIR", rag.get("faiss_index_dir", "data/processed/faiss_index")
    )
    return cfg                                  # MVP 설정 반환


# 상대 경로를 절대 경로로 변환
def resolve_path(relative: str | Path) -> Path:
    p = Path(relative)
    if p.is_absolute():     # 절대 경로이면 그대로 반환
        return p
    return ROOT / p         # 상대 경로를 절대 경로로 변환
