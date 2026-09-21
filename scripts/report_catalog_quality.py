"""Catalog attribute gaps for search-ops review. Spreadsheet-friendly CSV."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

from ml.config import load_mvp_config, resolve_path

SUMMARY_CSV = "data/eval/catalog_quality.csv"
GAPS_CSV = "data/eval/catalog_quality_gaps.csv"
SUMMARY_JSON = "data/eval/catalog_quality.json"


def _empty(val: object) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    text = str(val).strip()
    return text == "" or text.lower() == "nan"


def _str(val: object) -> str:
    if _empty(val):
        return ""
    return str(val).strip()


def catalog_quality(items: pd.DataFrame) -> tuple[dict, list[dict]]:
    n = len(items)
    title_empty = 0
    brand_empty = 0
    desc_empty = 0
    price_empty = 0
    doc_asin_only = 0
    cats: dict[str, int] = {}
    gaps: list[dict] = []
    for r in items.itertuples(index=False):
        item_id = str(r.item_id)
        title = _str(getattr(r, "title", ""))
        brand = _str(getattr(r, "brand", ""))
        desc = _str(getattr(r, "description", ""))
        doc = _str(getattr(r, "doc_text", ""))
        category = _str(getattr(r, "category", ""))
        price = getattr(r, "price", None)
        price_missing = _empty(price)
        t_empty = title == ""
        b_empty = brand == ""
        d_empty = desc == ""
        asin_only = (not doc) or doc == item_id
        title_empty += int(t_empty)
        brand_empty += int(b_empty)
        desc_empty += int(d_empty)
        price_empty += int(price_missing)
        doc_asin_only += int(asin_only)
        key = category or "(empty)"
        cats[key] = cats.get(key, 0) + 1
        if t_empty or b_empty or d_empty or price_missing or asin_only:
            gaps.append(
                {
                    "item_id": item_id,
                    "title": title,
                    "title_empty": int(t_empty),
                    "brand_empty": int(b_empty),
                    "description_empty": int(d_empty),
                    "price_empty": int(price_missing),
                    "doc_text_asin_only": int(asin_only),
                    "category": category,
                }
            )
    top_cat = sorted(cats.items(), key=lambda x: -x[1])[:5]
    summary = {
        "n_items": n,
        "title_empty": title_empty,
        "title_empty_rate": round(title_empty / n, 6) if n else 0.0,
        "brand_empty": brand_empty,
        "brand_empty_rate": round(brand_empty / n, 6) if n else 0.0,
        "description_empty": desc_empty,
        "description_empty_rate": round(desc_empty / n, 6) if n else 0.0,
        "price_empty": price_empty,
        "price_empty_rate": round(price_empty / n, 6) if n else 0.0,
        "doc_text_asin_only": doc_asin_only,
        "doc_text_asin_only_rate": round(doc_asin_only / n, 6) if n else 0.0,
        "n_unique_category": len(cats),
        "n_gap_rows": len(gaps),
        "top_categories": [{"category": c, "n": k} for c, k in top_cat],
    }
    return summary, gaps


def write_reports(
    summary: dict,
    gaps: list[dict],
    summary_csv: Path,
    gaps_csv: Path,
    summary_json: Path,
) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("n_items", summary["n_items"], 1.0, "catalog size"),
        ("title_empty", summary["title_empty"], summary["title_empty_rate"], "missing title"),
        ("brand_empty", summary["brand_empty"], summary["brand_empty_rate"], "missing brand"),
        (
            "description_empty",
            summary["description_empty"],
            summary["description_empty_rate"],
            "missing description",
        ),
        ("price_empty", summary["price_empty"], summary["price_empty_rate"], "missing price"),
        (
            "doc_text_asin_only",
            summary["doc_text_asin_only"],
            summary["doc_text_asin_only_rate"],
            "doc_text is empty or equal to ASIN",
        ),
        (
            "n_unique_category",
            summary["n_unique_category"],
            None,
            "Phase 4 unique cat ~1 on All_Beauty",
        ),
        ("n_gap_rows", summary["n_gap_rows"], None, "rows with at least one gap flag"),
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "n", "rate", "note"])
        writer.writeheader()
        for metric, n, rate, note in rows:
            writer.writerow({"metric": metric, "n": n, "rate": rate if rate is not None else "", "note": note})
    with gaps_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "item_id",
                "title",
                "title_empty",
                "brand_empty",
                "description_empty",
                "price_empty",
                "doc_text_asin_only",
                "category",
            ],
        )
        writer.writeheader()
        writer.writerows(gaps)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    cfg = load_mvp_config()
    items_path = resolve_path(cfg["data"]["processed_dir"]) / "items.parquet"
    if not items_path.exists():
        raise FileNotFoundError(f"Missing {items_path}. Run prepare_splits first.")
    cols = ["item_id", "title", "brand", "description", "price", "category", "doc_text"]
    items = pd.read_parquet(items_path, columns=cols)
    summary, gaps = catalog_quality(items)
    write_reports(
        summary,
        gaps,
        resolve_path(SUMMARY_CSV),
        resolve_path(GAPS_CSV),
        resolve_path(SUMMARY_JSON),
    )
    print("[catalog] n_items={n_items} unique_cat={n_unique_category}".format(**summary))
    print(
        "  title_empty={title_empty} ({title_empty_rate:.4f}) "
        "brand_empty={brand_empty} ({brand_empty_rate:.4f})".format(**summary)
    )
    print(
        "  desc_empty={description_empty} ({description_empty_rate:.4f}) "
        "price_empty={price_empty} ({price_empty_rate:.4f})".format(**summary)
    )
    print(
        "  doc_asin_only={doc_text_asin_only} ({doc_text_asin_only_rate:.4f}) "
        "gap_rows={n_gap_rows}".format(**summary)
    )
    print(f"[done] {SUMMARY_CSV}")
    print(f"[done] {GAPS_CSV}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
