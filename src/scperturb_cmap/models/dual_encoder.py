from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from scperturb_cmap.utils.device import get_device


class _Tower(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 64, p_dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.net(x)
        # L2 normalize
        eps = torch.finfo(z.dtype).eps
        z = z / (z.norm(p=2, dim=-1, keepdim=True) + eps)
        return z


class DualEncoder(nn.Module):
    """Dual-tower MLP encoder with L2-normalized outputs.

    The two towers do not share weights. Outputs embeddings of dimension ``embed_dim``.
    """

    def __init__(self, input_dim: int, embed_dim: int = 64, p_dropout: float = 0.1) -> None:
        super().__init__()
        self.left = _Tower(input_dim, embed_dim, p_dropout)
        self.right = _Tower(input_dim, embed_dim, p_dropout)

    def forward(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        if left.dim() == 1:
            left = left.unsqueeze(0)
        if right.dim() == 1:
            right = right.unsqueeze(0)
        zl = self.left(left)
        zr = self.right(right)
        sim = (zl * zr).sum(dim=-1)
        return zl, zr, sim


def build_pairs(baseline_df: pd.DataFrame, pos_k: int = 50, neg_k: int = 50) -> pd.DataFrame:
    """Build pair labels from a baseline connectivity table.

    Expects columns: ``target_id``, ``signature_id``, ``score``.
    Lower scores are considered better (inversion). Returns a DataFrame with
    columns ``left_id``, ``right_id``, ``label`` where label==1 for positive
    (inversion) and 0 for negative.
    """
    cols = set(baseline_df.columns)
    # Try to map flexible input column names
    target_col = (
        "target_id" if "target_id" in cols else ("query_id" if "query_id" in cols else None)
    )
    sig_col = "signature_id" if "signature_id" in cols else None
    score_col = "score" if "score" in cols else None
    if not (target_col and sig_col and score_col):
        raise ValueError(
            "baseline_df must contain target_id/query_id, signature_id, and score columns"
        )

    out_rows: List[Dict[str, object]] = []
    grouped = baseline_df.groupby(target_col, sort=False)
    for tgt, grp in grouped:
        grp_sorted = grp.sort_values(score_col, ascending=True)
        pos = grp_sorted.head(max(0, pos_k))
        neg = grp_sorted.tail(max(0, neg_k))
        for _, r in pos.iterrows():
            out_rows.append({"left_id": r[target_col], "right_id": r[sig_col], "label": 1})
        for _, r in neg.iterrows():
            out_rows.append({"left_id": r[target_col], "right_id": r[sig_col], "label": 0})
    return pd.DataFrame(out_rows, columns=["left_id", "right_id", "label"]) if out_rows else (
        pd.DataFrame(columns=["left_id", "right_id", "label"])
    )


class PairDataset(torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Pair dataset that yields (left_vec, right_vec, label) tensors.

    - ``pairs``: DataFrame with columns ``left_id``, ``right_id``, ``label``
    - ``vectors``: mapping from id to 1D numpy array of equal length
    - Tensors are moved to the selected device
    """

    def __init__(
        self,
        pairs: pd.DataFrame,
        vectors: Dict[str, np.ndarray],
        *,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.pairs = pairs.reset_index(drop=True)
        self.vectors = vectors
        self.device = device or get_device()
        self.dtype = dtype
        # Validate vectors are consistent dimensionality
        if len(vectors) > 0:
            first = next(iter(vectors.values()))
            self.input_dim = int(np.asarray(first).size)
        else:
            self.input_dim = 0

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        row = self.pairs.iloc[idx]
        lid = str(row["left_id"])
        rid = str(row["right_id"])
        y = float(row["label"])
        lvec = torch.tensor(
            np.asarray(self.vectors[lid], dtype=np.float32),
            dtype=self.dtype,
            device=self.device,
        )
        rvec = torch.tensor(
            np.asarray(self.vectors[rid], dtype=np.float32),
            dtype=self.dtype,
            device=self.device,
        )
        yvec = torch.tensor(y, dtype=self.dtype, device=self.device)
        return lvec, rvec, yvec
