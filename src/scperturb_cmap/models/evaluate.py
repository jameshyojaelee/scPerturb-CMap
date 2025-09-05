from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from scperturb_cmap.models.dual_encoder import DualEncoder


def _make_synthetic(input_dim: int = 16, num_targets: int = 8, seed: int = 1):
    rng = np.random.default_rng(seed)
    vectors: Dict[str, np.ndarray] = {}
    left_ids: List[str] = []
    pos_map: Dict[str, List[str]] = {}
    neg_map: Dict[str, List[str]] = {}
    for i in range(num_targets):
        tid = f"t{i}"
        left_ids.append(tid)
        t = rng.standard_normal(input_dim).astype(np.float32)
        vectors[tid] = t
        pos_ids: List[str] = []
        neg_ids: List[str] = []
        for j in range(3):
            sid = f"p{i}_{j}"
            vectors[sid] = -t + 0.05 * rng.standard_normal(input_dim).astype(np.float32)
            pos_ids.append(sid)
        for j in range(3):
            sid = f"n{i}_{j}"
            vectors[sid] = t + 0.05 * rng.standard_normal(input_dim).astype(np.float32)
            neg_ids.append(sid)
        pos_map[tid] = pos_ids
        neg_map[tid] = neg_ids
    return vectors, left_ids, pos_map, neg_map


def _recall_at_k_from_scores(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    # scores shape: [L, R], labels binary same shape
    L = scores.shape[0]
    hits = 0
    for i in range(L):
        idx = np.argsort(scores[i])[::-1][:k]
        if labels[i, idx].any():
            hits += 1
    return hits / max(1, L)


def evaluate_checkpoint(checkpoint_path: str, **kwargs) -> Dict[str, float]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    input_dim = int(cfg.get("input_dim", 16))
    model = DualEncoder(input_dim=input_dim, embed_dim=64)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    vectors, left_ids, pos_map, neg_map = _make_synthetic(input_dim=input_dim)
    # Build right id list
    right_ids = sorted(
        {rid for r in pos_map.values() for rid in r}
        | {rid for r in neg_map.values() for rid in r}
    )

    with torch.no_grad():
        ZL = []
        for lid in left_ids:
            z, _, _ = model(
                torch.tensor(vectors[lid]).float().unsqueeze(0),
                torch.tensor(vectors[lid]).float().unsqueeze(0),
            )
            ZL.append(z.squeeze(0).numpy())
        ZL = np.vstack(ZL)
        ZR = []
        for rid in right_ids:
            _, z, _ = model(
                torch.tensor(vectors[rid]).float().unsqueeze(0),
                torch.tensor(vectors[rid]).float().unsqueeze(0),
            )
            ZR.append(z.squeeze(0).numpy())
        ZR = np.vstack(ZR)

    # Retrieval with inversion (use -ZR)
    scores = ZL @ (-ZR).T
    labels = np.zeros_like(scores, dtype=bool)
    rid_index = {rid: i for i, rid in enumerate(right_ids)}
    for li, lid in enumerate(left_ids):
        for rid in pos_map[lid]:
            labels[li, rid_index[rid]] = True

    # Metrics
    rec1 = _recall_at_k_from_scores(scores, labels, k=1)
    rec5 = _recall_at_k_from_scores(scores, labels, k=5)
    # Flatten for spearman and auc
    y_true = labels.astype(int).ravel()
    y_score = scores.ravel()
    sp = float(spearmanr(y_true, y_score).correlation)
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = float("nan")
    return {"recall@1": rec1, "recall@5": rec5, "spearman": sp, "auc": auc}
