#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.data.lincs_loader import load_lincs_long
from scperturb_cmap.data.pairs import prepare_pair_table
from scperturb_cmap.io.schemas import TargetSignature
from scperturb_cmap.models.dual_encoder import DualEncoder
from scperturb_cmap.models.train import load_real_dataset

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


def compute_recall(
    model: DualEncoder,
    vectors: Dict[str, np.ndarray],
    left_ids: List[str],
    pos_map: Dict[str, List[str]],
    device: str,
    k: int = 5,
) -> float:
    model.eval()
    device_t = torch.device(device)
    right_ids = sorted({sig for sigs in pos_map.values() for sig in sigs})
    extra = [rid for rid in vectors.keys() if rid not in set(left_ids) | set(right_ids)]
    right_ids.extend(sorted(extra))
    right_ids = sorted(dict.fromkeys(right_ids))

    with torch.no_grad():
        ZL = []
        for lid in left_ids:
            vec = torch.tensor(vectors[lid], dtype=torch.float32, device=device_t).unsqueeze(0)
            z, _, _ = model(vec, vec)
            ZL.append(z.squeeze(0).cpu().numpy())
        ZL = np.vstack(ZL)

        ZR = []
        for rid in right_ids:
            vec = torch.tensor(vectors[rid], dtype=torch.float32, device=device_t).unsqueeze(0)
            _, z, _ = model(vec, vec)
            ZR.append(z.squeeze(0).cpu().numpy())
        ZR = np.vstack(ZR)

    scores = ZL @ (-ZR).T
    labels = np.zeros_like(scores, dtype=bool)
    ridx = {rid: i for i, rid in enumerate(right_ids)}
    for i, lid in enumerate(left_ids):
        for rid in pos_map.get(lid, []):
            labels[i, ridx[rid]] = True
    return _recall_at_k(scores, labels, k)


def prepare_metric_dataset(
    metric_dir: Path,
    *,
    input_dim: int = 32,
    num_targets: int = 12,
    negatives_per_target: int = 3,
    seed: int = 1,
) -> dict:
    metric_dir.mkdir(parents=True, exist_ok=True)
    vectors, left_ids, pos_map, _ = _make_synth(input_dim=input_dim, n=num_targets, seed=seed)
    genes = [f"G{i+1}" for i in range(input_dim)]

    right_ids = sorted(rid for rid in vectors.keys() if rid not in set(left_ids))
    rows = []
    for rid in right_ids:
        vec = vectors[rid]
        compound = rid
        cell_line = "CL_POS" if rid.startswith("p") else "CL_NEG"
        for gene, value in zip(genes, vec):
            rows.append(
                {
                    "signature_id": rid,
                    "compound": compound,
                    "cell_line": cell_line,
                    "gene_symbol": gene,
                    "score": float(value),
                }
            )
    library_df = pd.DataFrame(rows)
    library_path = metric_dir / "metric_library.parquet"
    library_df.to_parquet(library_path, engine="pyarrow", index=False)

    targets_path = metric_dir / "metric_targets.jsonl"
    with targets_path.open("w") as f:
        for lid in left_ids:
            record = {
                "target_id": lid,
                "genes": genes,
                "weights": vectors[lid].astype(float).tolist(),
            }
            f.write(json.dumps(record) + "\n")

    positives = [
        {"target_id": lid, "signature_id": sid}
        for lid, sigs in pos_map.items()
        for sid in sigs
    ]
    positives_df = pd.DataFrame(positives)
    library_meta = library_df[["signature_id", "cell_line"]].drop_duplicates()
    pairs_df = prepare_pair_table(
        positives_df,
        library_meta=library_meta,
        negatives_per_target=negatives_per_target,
        match_cell_line=False,
        random_state=seed,
    )
    pairs_path = metric_dir / "metric_pairs.parquet"
    pairs_df.to_parquet(pairs_path, engine="pyarrow", index=False)

    return {
        "pairs_path": pairs_path,
        "targets_path": targets_path,
        "library_path": library_path,
        "negatives_per_target": negatives_per_target,
        "seed": seed,
    }


def check_metric_improves() -> dict:
    metric_dir = OUT_DIR / "metric_dataset"
    dataset = prepare_metric_dataset(metric_dir)

    vectors, left_ids, pos_map, _ = load_real_dataset(
        str(dataset["pairs_path"]),
        library_path=str(dataset["library_path"]),
        targets_path=str(dataset["targets_path"]),
        negatives_per_target=dataset["negatives_per_target"],
        seed=dataset["seed"],
    )

    input_dim = len(next(iter(vectors.values())))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    # Baseline (untrained) recall
    m0 = DualEncoder(input_dim=input_dim, embed_dim=64)
    m0.to(device_t)
    r0 = compute_recall(m0, vectors, left_ids, pos_map, device, k=5)

    # Train with real-data pipeline
    import subprocess

    subprocess.run(
        [
            ".venv/bin/python",
            "-m",
            "scperturb_cmap.models.train",
            f"pairs_path={dataset['pairs_path']}",
            f"targets_path={dataset['targets_path']}",
            f"library_path={dataset['library_path']}",
            f"negatives_per_target={dataset['negatives_per_target']}",
            "epochs=5",
            "batch_size=128",
            "hydra.run.dir=.",
        ],
        check=True,
    )

    ckpt = torch.load("workspace/artifacts/best.pt", map_location=device_t)
    trained_input_dim = int(ckpt.get("config", {}).get("input_dim", input_dim))
    model = DualEncoder(input_dim=trained_input_dim, embed_dim=64)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device_t)
    r1 = compute_recall(model, vectors, left_ids, pos_map, device, k=5)
    improvement = r1 - r0
    ok = improvement >= 0.10
    return {
        "ok": ok,
        "baseline_recall@5": r0,
        "metric_recall@5": r1,
        "improvement": improvement,
    }


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
