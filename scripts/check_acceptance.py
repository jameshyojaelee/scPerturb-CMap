#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.data.lincs_loader import load_lincs_long
from scperturb_cmap.io.schemas import TargetSignature
from scperturb_cmap.models.dual_encoder import DualEncoder

EX_DIR = Path("examples/data")
OUT_DIR = Path("examples/out")


def ensure_demo() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure demo LINCS exists; generate via print_demo_stats (which also writes parquet)
    from print_demo_stats import ensure_lincs_demo

    lincs_path = ensure_lincs_demo()
    return load_lincs_long(str(lincs_path))


def check_baseline_time(df_long: pd.DataFrame) -> dict:
    ts = TargetSignature(genes=["G1", "G2", "G10"], weights=[1.0, 1.0, -1.0])
    t0 = time.time()
    res = rank_drugs(ts, df_long, method="baseline", top_k=50)
    dt = time.time() - t0
    ranking_df = (
        res.ranking
        if isinstance(res.ranking, pd.DataFrame)
        else pd.DataFrame(res.ranking)
    )
    ok = (dt < 60.0) and (len(ranking_df) > 0)
    # Save output for inspection
    out_path = OUT_DIR / "results.parquet"
    ranking_df.to_parquet(out_path, engine="pyarrow", index=False)
    return {"ok": ok, "seconds": dt, "rows": len(ranking_df), "path": str(out_path)}


def _make_synth(input_dim: int = 16, n: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    vectors = {}
    left_ids = []
    pos_map = {}
    neg_map = {}
    for i in range(n):
        tid = f"t{i}"
        left_ids.append(tid)
        t = rng.standard_normal(input_dim).astype(np.float32)
        vectors[tid] = t
        pos_list = []
        neg_list = []
        for j in range(3):
            sid = f"p{i}_{j}"
            vectors[sid] = -t + 0.05 * rng.standard_normal(input_dim).astype(np.float32)
            pos_list.append(sid)
        for j in range(3):
            sid = f"n{i}_{j}"
            vectors[sid] = t + 0.05 * rng.standard_normal(input_dim).astype(np.float32)
            neg_list.append(sid)
        pos_map[tid] = pos_list
        neg_map[tid] = neg_list
    return vectors, left_ids, pos_map, neg_map


def _recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    L = scores.shape[0]
    hits = 0
    for i in range(L):
        topk = np.argsort(scores[i])[::-1][:k]
        if labels[i, topk].any():
            hits += 1
    return hits / max(1, L)


def compute_recall(model: DualEncoder, k: int = 5) -> float:
    model.eval()
    vectors, left_ids, pos_map, neg_map = _make_synth()
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
    scores = ZL @ (-ZR).T
    labels = np.zeros_like(scores, dtype=bool)
    ridx = {rid: i for i, rid in enumerate(right_ids)}
    for i, lid in enumerate(left_ids):
        for rid in pos_map[lid]:
            labels[i, ridx[rid]] = True
    return _recall_at_k(scores, labels, k)


def check_metric_improves() -> dict:
    # Baseline (untrained) recall
    m0 = DualEncoder(input_dim=16, embed_dim=64)
    r0 = compute_recall(m0, k=5)

    # Train quickly and load best
    import subprocess

    subprocess.run(
        [
            ".venv/bin/python",
            "-m",
            "scperturb_cmap.models.train",
            "epochs=3",
            "batch_size=64",
            "hydra.run.dir=.",
        ],
        check=True,
    )
    ckpt = torch.load("artifacts/best.pt", map_location="cpu")
    model = DualEncoder(input_dim=int(ckpt.get("config", {}).get("input_dim", 16)), embed_dim=64)
    model.load_state_dict(ckpt["state_dict"])
    r1 = compute_recall(model, k=5)
    improvement = r1 - r0
    ok = improvement >= 0.10
    return {"ok": ok, "baseline_recall@5": r0, "metric_recall@5": r1, "improvement": improvement}


def main() -> None:
    df_long = ensure_demo()
    baseline = check_baseline_time(df_long)
    metric = check_metric_improves()
    results = {"baseline_time": baseline, "metric_improvement": metric}
    print(json.dumps(results, indent=2))
    if not (baseline["ok"] and metric["ok"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

