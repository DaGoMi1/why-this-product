"""Two-Tower 짧은 학습: rating>=5, 히스토리 mean-pool, in-batch softmax."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ml.config import load_mvp_config, resolve_path
from ml.retrieval.two_tower import TwoTowerRetriever


class _HistTargetDataset(Dataset):
    def __init__(
        self,
        hist_idx: np.ndarray,
        hist_mask: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.hist_idx = torch.from_numpy(hist_idx)
        self.hist_mask = torch.from_numpy(hist_mask)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.hist_idx[i], self.hist_mask[i], self.targets[i]


class _TwoTower(nn.Module):
    def __init__(self, n_items: int, dim: int) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)

    def encode_items(self, item_idx: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_emb(item_idx), dim=-1)

    def encode_users(self, hist_idx: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        emb = self.item_emb(hist_idx)
        mask = hist_mask.unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(summed / denom, dim=-1)


def _coverage(train: pd.DataFrame, min_rating: float) -> dict[str, int | float]:
    user_counts = train.groupby("user_id").size()
    n_users = int(user_counts.shape[0])
    n_single = int((user_counts == 1).sum())
    n_multi = int((user_counts >= 2).sum())
    ge5 = train[train["rating"] >= min_rating]
    return {
        "train_rows": int(len(train)),
        "train_users": n_users,
        "single_review_users": n_single,
        "multi_review_users": n_multi,
        "single_review_pct": 100.0 * n_single / n_users if n_users else 0.0,
        "multi_review_pct": 100.0 * n_multi / n_users if n_users else 0.0,
        "ge5_positives": int(len(ge5)),
    }


def _build_pairs(
    train: pd.DataFrame,
    min_rating: float,
    last_n: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, int]:
    pos = train.loc[
        train["rating"] >= min_rating, ["user_id", "item_id", "timestamp"]
    ].copy()
    pos["user_id"] = pos["user_id"].astype(str)
    pos["item_id"] = pos["item_id"].astype(str)
    pos = pos.sort_values(["user_id", "timestamp"], kind="mergesort")

    item_ids = sorted(pos["item_id"].unique().tolist())
    item_index = {it: i + 1 for i, it in enumerate(item_ids)}

    hist_rows: list[list[int]] = []
    targets: list[int] = []
    dropped = 0
    ge5_count = 0

    for _, grp in pos.groupby("user_id", sort=False):
        seen: list[int] = []
        for row in grp.itertuples(index=False):
            ge5_count += 1
            tid = item_index[str(row.item_id)]
            if not seen:
                dropped += 1
                seen.append(tid)
                continue
            hist = seen[-last_n:]
            padded = hist + [0] * (last_n - len(hist))
            hist_rows.append(padded)
            targets.append(tid)
            seen.append(tid)

    if not hist_rows:
        raise ValueError("no two-tower training pairs (every ge_5 row had empty history)")

    hist_idx = np.asarray(hist_rows, dtype=np.int64)
    hist_mask = (hist_idx > 0).astype(np.float32)
    target_arr = np.asarray(targets, dtype=np.int64)
    print(
        f"[coverage] ge5_positives={ge5_count:,} "
        f"dropped_no_history={dropped:,} ({100.0 * dropped / ge5_count:.1f}%) "
        f"train_pairs={len(targets):,}"
    )
    return item_ids, hist_idx, hist_mask, target_arr, dropped


def train_one(
    train: pd.DataFrame,
    dim: int,
    last_n: int,
    batch_size: int,
    epochs: int,
    lr: float,
    min_rating: float,
    out_dir: Path,
) -> TwoTowerRetriever:
    cov = _coverage(train, min_rating)
    print(
        f"[coverage] train_rows={cov['train_rows']:,} "
        f"train_users={cov['train_users']:,} "
        f"single_review={cov['single_review_users']:,} ({cov['single_review_pct']:.1f}%) "
        f"multi_review={cov['multi_review_users']:,} ({cov['multi_review_pct']:.1f}%)"
    )
    print(f"[coverage] ge5_positives={cov['ge5_positives']:,}")

    item_ids, hist_idx, hist_mask, targets, _dropped = _build_pairs(
        train, min_rating=min_rating, last_n=last_n
    )
    pair_pct = 100.0 * len(targets) / cov["train_rows"] if cov["train_rows"] else 0.0
    print(f"[coverage] train_pairs / train_rows = {pair_pct:.1f}%")
    print(f"[two_tower] items={len(item_ids):,} pairs={len(targets):,} dim={dim} last_n={last_n}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _TwoTower(n_items=len(item_ids), dim=dim).to(device)
    loader = DataLoader(
        _HistTargetDataset(hist_idx, hist_mask, targets),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(targets) > batch_size,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        n_batches = 0
        for hist, mask, target in loader:
            hist = hist.to(device)
            mask = mask.to(device)
            target = target.to(device)
            user_vec = model.encode_users(hist, mask)
            item_vec = model.encode_items(target)
            logits = user_vec @ item_vec.T
            labels = torch.arange(target.size(0), device=device)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n_batches += 1
        print(f"[epoch {epoch}/{epochs}] loss={total / max(n_batches, 1):.4f}")

    model.eval()
    with torch.no_grad():
        idx = torch.arange(1, len(item_ids) + 1, device=device)
        item_factors = model.encode_items(idx).cpu().numpy().astype(np.float32)

    retriever = TwoTowerRetriever(
        item_ids=item_ids,
        item_factors=item_factors,
        last_n=last_n,
    )
    retriever.save(out_dir)
    print(f"[done] {out_dir}")
    return retriever


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    cfg = load_mvp_config()
    tt_cfg = cfg.get("two_tower", {})
    processed = resolve_path(cfg["data"]["processed_dir"])
    train_path = processed / "interactions_train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Run prepare_splits first.")

    train = pd.read_parquet(train_path, columns=["user_id", "item_id", "rating", "timestamp"])
    out_dir = resolve_path(tt_cfg.get("artifact_dir", "data/processed/two_tower"))
    train_one(
        train=train,
        dim=int(tt_cfg.get("dim", 64)),
        last_n=int(tt_cfg.get("last_n", 10)),
        batch_size=int(tt_cfg.get("batch_size", 512)),
        epochs=int(tt_cfg.get("epochs", 5)),
        lr=float(tt_cfg.get("lr", 1e-3)),
        min_rating=float(tt_cfg.get("min_rating", 5)),
        out_dir=out_dir,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
