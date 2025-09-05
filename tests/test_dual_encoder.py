from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from scperturb_cmap.models.dual_encoder import DualEncoder, PairDataset, build_pairs


def test_dual_encoder_forward_shapes():
    torch.manual_seed(0)
    model = DualEncoder(input_dim=8, embed_dim=16)
    x1 = torch.randn(4, 8)
    x2 = torch.randn(4, 8)
    z1, z2, sim = model(x1, x2)
    assert z1.shape == (4, 16)
    assert z2.shape == (4, 16)
    assert sim.shape == (4,)
    # L2 normalization keeps norm ~1
    assert torch.allclose(z1.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(z2.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_pairdataset_and_tiny_training_step():
    torch.manual_seed(0)
    d = 16
    # Create a target vector t and its inverted vector for positive inversion pairs
    t = np.random.randn(d).astype(np.float32)
    s_pos = -t + 0.01 * np.random.randn(d).astype(np.float32)
    s_neg = t + 0.01 * np.random.randn(d).astype(np.float32)

    vectors = {
        "t1": t,
        "pos": s_pos,
        "neg": s_neg,
    }

    pairs = pd.DataFrame(
        [
            {"left_id": "t1", "right_id": "pos", "label": 1},  # inversion (desired sim ~ -1)
            {"left_id": "t1", "right_id": "neg", "label": 0},  # concordant (desired sim ~ +1)
        ]
    )
    ds = PairDataset(pairs, vectors)
    model = DualEncoder(input_dim=d, embed_dim=16, p_dropout=0.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    def batch_loss():
        left_all, right_all, y = [], [], []
        for i in range(len(ds)):
            left, right, yi = ds[i]
            left_all.append(left)
            right_all.append(right)
            # map label 1 -> target -1, label 0 -> target +1
            y.append(-1.0 if float(yi.item()) > 0.5 else 1.0)
        L = torch.stack(left_all)
        R = torch.stack(right_all)
        y_t = torch.tensor(y, dtype=torch.float32)
        _, _, sim = model(L, R)
        return torch.mean((sim - y_t) ** 2)

    # Train for a few steps; loss should go down
    loss0 = batch_loss().item()
    for _ in range(20):
        opt.zero_grad()
        loss = batch_loss()
        loss.backward()
        opt.step()
    loss1 = batch_loss().item()
    assert loss1 < loss0


def test_build_pairs_basic():
    df = pd.DataFrame(
        {
            "target_id": ["t1"] * 5,
            "signature_id": [f"s{i}" for i in range(5)],
            "score": [0.1, -0.5, 0.3, -0.6, 0.2],
        }
    )
    pairs = build_pairs(df, pos_k=2, neg_k=2)
    assert set(pairs.columns) == {"left_id", "right_id", "label"}
    # Top 2 lowest scores are s3 (-0.6) and s1 (-0.5)
    pos_set = set(pairs.loc[pairs["label"] == 1, "right_id"].tolist())
    assert pos_set == {"s1", "s3"}
