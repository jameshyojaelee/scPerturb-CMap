from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from scperturb_cmap.explainability.feature_importance import compute_gene_contributions
from scperturb_cmap.io.schemas import TargetSignature


def load_target_signature_from_json_bytes(data: bytes) -> TargetSignature:
    """
    Parse a TargetSignature from JSON or JSON Lines bytes payloads.

    Accepts objects, arrays (first element is used), or single-line JSON.
    """
    if not data:
        raise ValueError("Empty JSON payload.")

    try:
        payload = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError:
        lines = [line for line in data.decode("utf-8").splitlines() if line.strip()]
        if not lines:
            raise ValueError("No JSON content found.")
        payload = json.loads(lines[0])

    if isinstance(payload, list):
        if not payload:
            raise ValueError("JSON array was empty.")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("TargetSignature JSON must be an object or list of objects.")

    return TargetSignature.model_validate(payload)


def compute_contributions_from_library(
    target_sig: TargetSignature,
    library_df: pd.DataFrame,
    signature_id: str,
) -> pd.DataFrame:
    """
    Compute SHAP-like gene contributions for a library signature.

    Args:
        target_sig: TargetSignature representing the query.
        library_df: Long-form LINCS table with columns signature_id, gene_symbol, score.
        signature_id: Signature to explain.
    """
    required = {"signature_id", "gene_symbol", "score"}
    if not required.issubset(set(library_df.columns)):
        missing = ", ".join(sorted(required - set(library_df.columns)))
        raise ValueError(f"Library missing required columns: {missing}")

    subset = library_df[library_df["signature_id"] == signature_id]
    if subset.empty:
        raise ValueError(f"Signature '{signature_id}' not found in library.")

    target_map = {str(g).upper(): (g, float(w)) for g, w in zip(target_sig.genes, target_sig.weights)}
    score_map = {str(row.gene_symbol).upper(): float(row.score) for row in subset.itertuples()}

    overlap_keys = [g for g in target_map if g in score_map]
    if not overlap_keys:
        raise ValueError(f"No overlapping genes for signature '{signature_id}'.")

    ordered_genes = [target_map[g][0] for g in overlap_keys]
    target_vector = [target_map[g][1] for g in overlap_keys]
    drug_vector = [score_map[g] for g in overlap_keys]

    contrib = compute_gene_contributions(target_vector, drug_vector, ordered_genes)
    contrib["signature_id"] = signature_id
    return contrib.sort_values("abs_contribution", ascending=False).reset_index(drop=True)


def persist_exports(
    out_dir: Path,
    stem: str,
    csv_bytes: bytes,
    json_bytes: bytes,
    session_bytes: Optional[bytes] = None,
) -> Dict[str, Path]:
    """Write export artifacts to disk and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    csv_path.write_bytes(csv_bytes)
    json_path.write_bytes(json_bytes)
    paths["csv"] = csv_path
    paths["json"] = json_path

    if session_bytes:
        session_path = out_dir / f"{stem}_session.json"
        session_path.write_bytes(session_bytes)
        paths["session"] = session_path

    return paths
