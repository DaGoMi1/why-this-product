"""아마존 리뷰 2018 All_Beauty를 data/raw/에 저장"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from ml.config import load_mvp_config, project_root, resolve_path

REVIEW_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/"
    "categoryFiles/All_Beauty.json.gz"
)
META_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/"
    "metaFiles2/meta_All_Beauty.json.gz"
)

FILES = {
    "All_Beauty.json.gz": REVIEW_URL,
    "meta_All_Beauty.json.gz": META_URL,
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] already exists: {dest} ({dest.stat().st_size:,} bytes)")
        return
    print(f"[download] {url}")
    print(f"         -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    print(f"[done] {dest.name} ({dest.stat().st_size:,} bytes)")


def main() -> None:
    cfg = load_mvp_config()
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    print(f"project root: {project_root()}")
    print(f"raw dir: {raw_dir}")
    for name, url in FILES.items():
        _download(url, raw_dir / name)
    print("All_Beauty raw files ready.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
