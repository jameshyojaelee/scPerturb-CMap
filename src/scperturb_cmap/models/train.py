from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra import main as hydra_main
from omegaconf import OmegaConf

from scperturb_cmap.data.lincs_loader import load_lincs_long, pivot_signatures
from scperturb_cmap.data.preprocess import harmonize_symbols, standardize_vector
from scperturb_cmap.io.schemas import TargetSignature
from scperturb_cmap.models.dual_encoder import DualEncoder
from scperturb_cmap.utils.device import get_device
from scperturb_cmap.utils.seed import set_global_seed

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # optimization
    lr: float = 1e-2
    weight_decay: float = 0.0
    epochs: int = 5
    batch_size: int = 32
    temperature: float = 0.2
    loss_type: str = "ntxent"  # or "triplet"

    # data
    input_dim: int = 16
    num_targets: int = 8
    pos_per_target: int = 4
    neg_per_target: int = 4
    seed: int = 0
    k: int = 5  # recall@k

    # misc
    device: str = "auto"  # auto | cpu | cuda | mps
    pairs_path: Optional[str] = None
    targets_path: Optional[str] = None
    library_path: Optional[str] = None
    negatives_per_target: int = 3


def set_seed(seed: int) -> None:
    set_global_seed(seed)


def make_synthetic(
    input_dim: int, num_targets: int, pos_per_target: int, neg_per_target: int, seed: int
) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, List[str]], Dict[str, List[str]]]:
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
        for j in range(pos_per_target):
            sid = f"p{i}_{j}"
            v = -t + 0.05 * rng.standard_normal(input_dim).astype(np.float32)
            vectors[sid] = v
            pos_ids.append(sid)
        for j in range(neg_per_target):
            sid = f"n{i}_{j}"
            v = t + 0.05 * rng.standard_normal(input_dim).astype(np.float32)
            vectors[sid] = v
            neg_ids.append(sid)
        pos_map[tid] = pos_ids
        neg_map[tid] = neg_ids

    return vectors, left_ids, pos_map, neg_map


def sample_pos_batch(
    left_ids: List[str], pos_map: Dict[str, List[str]], batch_size: int
) -> List[Tuple[str, str]]:
    eligible = [lid for lid in left_ids if pos_map.get(lid)]
    if not eligible:
        raise ValueError("No positive pairs available for sampling")
    batch: List[Tuple[str, str]] = []
    for _ in range(batch_size):
        lid = random.choice(eligible)
        rid = random.choice(pos_map[lid])
        batch.append((lid, rid))
    return batch


def sample_triplet_batch(
    left_ids: List[str],
    pos_map: Dict[str, List[str]],
    neg_map: Dict[str, List[str]],
    batch_size: int,
) -> List[Tuple[str, str, str]]:
    eligible = [lid for lid in left_ids if pos_map.get(lid) and neg_map.get(lid)]
    if not eligible:
        raise ValueError("Triplet sampling requires at least one positive and negative per target")
    batch: List[Tuple[str, str, str]] = []
    for _ in range(batch_size):
        lid = random.choice(eligible)
        pid = random.choice(pos_map[lid])
        nid = random.choice(neg_map[lid])
        batch.append((lid, pid, nid))
    return batch


def _load_targets(path: str, gene_reference: List[str]) -> Dict[str, np.ndarray]:
    ref_genes = harmonize_symbols(gene_reference)
    index_map = {g: i for i, g in enumerate(ref_genes)}

    targets: Dict[str, np.ndarray] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "genes" not in record or "weights" not in record:
            raise ValueError("Targets JSONL must contain 'genes' and 'weights'")
        sig = TargetSignature(genes=record["genes"], weights=record["weights"])
        vec = np.zeros(len(ref_genes), dtype=float)
        for g, w in zip(harmonize_symbols(sig.genes), sig.weights):
            if g in index_map:
                vec[index_map[g]] = w
        vec = standardize_vector(vec).astype(np.float32)
        key = record.get("target_id") or record.get("id") or sig.metadata.get("id")
        if key is None:
            raise ValueError("Targets JSONL must provide 'target_id' or 'id' for each record")
        targets[str(key)] = vec
    if not targets:
        raise ValueError("No targets were loaded from targets_path")
    return targets


def _build_maps(
    pairs: pd.DataFrame,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    left_ids = sorted(pairs["left_id"].unique())
    pos_map: Dict[str, List[str]] = {lid: [] for lid in left_ids}
    neg_map: Dict[str, List[str]] = {lid: [] for lid in left_ids}
    for _, row in pairs.iterrows():
        lid, rid, label = str(row["left_id"]), str(row["right_id"]), int(row["label"])
        if label == 1:
            pos_map.setdefault(lid, []).append(rid)
        else:
            neg_map.setdefault(lid, []).append(rid)
    return left_ids, pos_map, neg_map


def load_real_dataset(
    pairs_path: str,
    library_path: str,
    targets_path: str,
    negatives_per_target: int,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    if pairs_path.endswith(".parquet"):
        pairs_df = pd.read_parquet(pairs_path)
    else:
        pairs_df = pd.read_csv(pairs_path)
    if not {"left_id", "right_id", "label"}.issubset(set(pairs_df.columns)):
        raise ValueError("pairs_path must contain columns left_id,right_id,label")

    library = load_lincs_long(library_path)
    needed = set(pairs_df["right_id"].astype(str))
    library = library[library["signature_id"].astype(str).isin(needed)].copy()
    if library.empty:
        raise ValueError("No signatures from pairs found in library")

    M, genes, meta = pivot_signatures(library)
    right_vectors = {
        str(sig): M[i].astype(np.float32)
        for i, sig in enumerate(meta["signature_id"].astype(str))
    }

    targets = _load_targets(targets_path, harmonize_symbols(genes))

    vectors: Dict[str, np.ndarray] = {**right_vectors}
    for tid, vec in targets.items():
        vectors[str(tid)] = np.asarray(vec, dtype=np.float32)

    left_ids, pos_map, neg_map = _build_maps(pairs_df)

    # If negatives missing for some targets, sample from remaining signatures
    rng = np.random.default_rng(seed)
    all_signatures = sorted(right_vectors.keys())
    for lid in left_ids:
        if pos_map.get(lid):
            candidate = [s for s in all_signatures if s not in pos_map[lid]]
        else:
            candidate = all_signatures
        if not neg_map.get(lid) and candidate:
            k = min(max(1, negatives_per_target), len(candidate))
            neg_map[lid] = rng.choice(candidate, size=k, replace=False).tolist()

    return vectors, left_ids, pos_map, neg_map


def recall_at_k(
    model: DualEncoder,
    vectors: Dict[str, np.ndarray],
    left_ids: List[str],
    pos_map: Dict[str, List[str]],
    device: str,
    k: int,
) -> float:
    model.eval()
    with torch.no_grad():
        # Precompute right embeddings for all signature ids
        sig_ids = sorted({rid for rids in pos_map.values() for rid in rids})
        # Also include decoys from negatives to make it more realistic
        neg_ids = [sid for sid in vectors.keys() if sid not in left_ids and sid not in sig_ids]
        all_sids = sig_ids + neg_ids

        right_mat = torch.stack(
            [torch.tensor(vectors[s], dtype=torch.float32) for s in all_sids]
        )
        device_t = torch.device(device)
        right_mat = right_mat.to(device_t)
        _, right_z, _ = model(
            torch.zeros(
                (right_mat.shape[0], model.left.net[0].normalized_shape[0]),
                device=device_t,
            ),
            right_mat,
        )
        # Normalize again for safety
        right_z = F.normalize(right_z, p=2, dim=-1)

        hits = 0
        for lid in left_ids:
            left_vec = torch.tensor(vectors[lid], dtype=torch.float32, device=device_t).unsqueeze(0)
            left_z, _, _ = model(left_vec, left_vec)
            left_z = F.normalize(left_z, p=2, dim=-1)
            # Retrieval score uses inverted right to reflect inversion matching
            scores = (left_z @ (-right_z).T).squeeze(0)
            topk = torch.topk(scores, k=min(k, scores.numel())).indices.tolist()
            top_ids = {all_sids[i] for i in topk}
            pos_ids = set(pos_map[lid])
            if top_ids & pos_ids:
                hits += 1
    return hits / max(1, len(left_ids))


@hydra_main(config_path="../configs", config_name="train", version_base=None)
def run(cfg: OmegaConf) -> None:
    # Merge defaults
    tc = OmegaConf.merge(OmegaConf.structured(TrainConfig), cfg)
    cfg = tc  # type: ignore

    # Seeding and device
    set_seed(int(cfg.seed))
    device = cfg.device
    if device == "auto":
        device = get_device()
    device_t = torch.device("cuda" if device == "cuda" else ("mps" if device == "mps" else "cpu"))

    # Data
    if cfg.pairs_path and cfg.library_path and cfg.targets_path:
        vectors, left_ids, pos_map, neg_map = load_real_dataset(
            cfg.pairs_path,
            cfg.library_path,
            cfg.targets_path,
            cfg.negatives_per_target,
            cfg.seed,
        )
        if vectors:
            first_vec = next(iter(vectors.values()))
            cfg.input_dim = int(len(first_vec))
    else:
        vectors, left_ids, pos_map, neg_map = make_synthetic(
            cfg.input_dim, cfg.num_targets, cfg.pos_per_target, cfg.neg_per_target, cfg.seed
        )

    left_ids = [lid for lid in left_ids if pos_map.get(lid)]
    if not left_ids:
        raise ValueError("No positive pairs available; ensure pairs_path defines label==1 rows")

    # Split left_ids into train/val
    random.shuffle(left_ids)
    n_val = max(1, len(left_ids) // 5)
    val_left = left_ids[:n_val]
    train_left = left_ids[n_val:]
    if not train_left:
        train_left = val_left

    # Model and optimizer
    model = DualEncoder(input_dim=cfg.input_dim, embed_dim=64, p_dropout=0.1)
    model.to(device_t)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_recall = -1.0
    artifacts_dir = Path("workspace") / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"epoch": [], "loss": [], "val_recall@k": []}

    for epoch in range(int(cfg.epochs)):
        model.train()
        max_pos = max(len(pos_map[lid]) for lid in train_left)
        steps = max(1, math.ceil(len(train_left) * max_pos / max(1, cfg.batch_size)))
        running_loss = 0.0

        for _ in tqdm(range(steps), desc=f"epoch {epoch}"):
            optim.zero_grad()
            if cfg.loss_type.lower() == "ntxent":
                batch = sample_pos_batch(train_left, pos_map, cfg.batch_size)
                # Compose batch tensors; invert right vectors to create concordant pairs
                L = torch.stack(
                    [torch.tensor(vectors[left], dtype=torch.float32) for left, _ in batch]
                ).to(device_t)
                R = torch.stack(
                    [torch.tensor(-vectors[r], dtype=torch.float32) for _, r in batch]
                ).to(device_t)
                zL, zR, _ = model(L, R)
                zL = F.normalize(zL, p=2, dim=-1)
                zR = F.normalize(zR, p=2, dim=-1)
                logits = (zL @ zR.T) / float(cfg.temperature)
                labels = torch.arange(logits.shape[0], device=logits.device)
                loss = F.cross_entropy(logits, labels)
            else:  # triplet
                batch = sample_triplet_batch(train_left, pos_map, neg_map, cfg.batch_size)
                A = torch.stack(
                    [torch.tensor(vectors[left], dtype=torch.float32) for left, _, _ in batch]
                ).to(device_t)
                P = torch.stack(
                    [torch.tensor(-vectors[p], dtype=torch.float32) for _, p, _ in batch]
                ).to(device_t)
                N = torch.stack(
                    [torch.tensor(vectors[n], dtype=torch.float32) for _, _, n in batch]
                ).to(device_t)
                zA, _, _ = model(A, A)
                zP, _, _ = model(P, P)
                zN, _, _ = model(N, N)
                zA = F.normalize(zA, p=2, dim=-1)
                zP = F.normalize(zP, p=2, dim=-1)
                zN = F.normalize(zN, p=2, dim=-1)
                loss = F.triplet_margin_loss(zA, zP, zN, margin=0.2, p=2)

            loss.backward()
            optim.step()
            running_loss += float(loss.item())

        avg_loss = running_loss / steps
        val_recall = recall_at_k(model, vectors, val_left, pos_map, device, int(cfg.k))
        logger.info("epoch=%s loss=%.4f val_recall@%s=%.3f", epoch, avg_loss, cfg.k, val_recall)
        metrics["epoch"].append(epoch)
        metrics["loss"].append(avg_loss)
        metrics["val_recall@k"].append(val_recall)
        # Save best
        if val_recall > best_recall:
            best_recall = val_recall
            ckpt = {
                "state_dict": model.state_dict(),
                "config": OmegaConf.to_container(OmegaConf.create(cfg), resolve=True),
            }
            torch.save(ckpt, artifacts_dir / "best.pt")
            logger.info("Saved best checkpoint to %s", artifacts_dir / "best.pt")

    # Write metrics
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump({"best_recall@k": best_recall, **{k: v for k, v in metrics.items()}}, f)


if __name__ == "__main__":
    run()
