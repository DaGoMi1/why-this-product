"""Catalog quality flags. No parquet."""

from __future__ import annotations

import pandas as pd

from scripts.report_catalog_quality import catalog_quality


def test_catalog_quality_counts_gaps() -> None:
    items = pd.DataFrame(
        [
            {
                "item_id": "B1",
                "title": "Serum",
                "brand": "Acme",
                "description": "A serum.",
                "price": 12.0,
                "category": "Beauty",
                "doc_text": "Serum. Acme. A serum.",
            },
            {
                "item_id": "B2",
                "title": "",
                "brand": None,
                "description": None,
                "price": None,
                "category": "Beauty",
                "doc_text": "B2",
            },
        ]
    )
    summary, gaps = catalog_quality(items)
    assert summary["n_items"] == 2
    assert summary["title_empty"] == 1
    assert summary["brand_empty"] == 1
    assert summary["description_empty"] == 1
    assert summary["price_empty"] == 1
    assert summary["doc_text_asin_only"] == 1
    assert summary["n_unique_category"] == 1
    assert len(gaps) == 1
    assert gaps[0]["item_id"] == "B2"
