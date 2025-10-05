from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scperturb_cmap.analysis.power import (
    bootstrap_rank_confidence,
    compute_signature_stability,
    estimate_signature_sample_size,
    permutation_significance_test,
    recommend_min_cells_per_cluster,
    simulate_false_discovery_rate,
)


def _make_toy_adata() -> ad.AnnData:
    X = np.array(
        [
            [10.0, 0.5, 1.0],
            [11.0, 0.4, 1.0],
            [1.0, 3.0, 1.0],
            [2.0, 2.5, 1.0],
        ]
    )
    adata = ad.AnnData(X=X)
    adata.obs_names = ["T1", "T2", "R1", "R2"]
    adata.var_names = ["G1", "G2", "G3"]
    adata.obs["cluster"] = ["A", "A", "B", "B"]
    adata.obs["batch"] = ["P1", "P1", "P1", "P2"]
    return adata


def test_sample_size_estimation_produces_history():
    adata = _make_toy_adata()
    result = estimate_signature_sample_size(
        adata,
        cluster_key="cluster",
        cluster="A",
        sample_sizes=[1, 2],
        replicates=5,
        random_state=0,
    )
    assert not result.history.empty
    assert set(result.summary["sample_size"].tolist()) <= {1, 2}
    assert 1 <= result.recommended_size <= 2
    assert result.history["correlation"].between(-1.0, 1.0).all()


def test_bootstrap_rank_confidence_returns_ci():
    df = pd.DataFrame(
        {
            "signature_id": ["s1", "s1", "s2", "s2"],
            "replicate_id": ["r1", "r2", "r1", "r2"],
            "score": [1.0, 1.2, 2.0, 2.2],
        }
    )
    res = bootstrap_rank_confidence(df, n_boot=100, random_state=1)
    assert set(res["signature_id"]) == {"s1", "s2"}
    assert (res["rank_ci_upper"] >= res["rank_ci_lower"]).all()


def test_signature_stability_detects_consistent_replicates():
    df = pd.DataFrame(
        {
            "signature_id": ["sig"] * 6,
            "replicate_id": ["r1", "r1", "r2", "r2", "r3", "r3"],
            "gene_symbol": ["g1", "g2", "g1", "g2", "g1", "g2"],
            "score": [1.0, -1.0, 1.1, -1.1, 0.9, -0.9],
        }
    )
    res = compute_signature_stability(df)
    assert res.iloc[0]["n_replicates"] == 3
    assert res.iloc[0]["mean_correlation"] > 0.95


def test_recommend_min_cells_per_cluster_returns_dataframe():
    adata = _make_toy_adata()
    out = recommend_min_cells_per_cluster(
        adata,
        cluster_key="cluster",
        sample_sizes=[1, 2],
        replicates=3,
        random_state=0,
    )
    rec = out["recommendations"]
    assert set(rec["cluster"]) == {"A", "B"}
    assert {"histories", "summaries"} <= set(out)


def test_simulate_false_discovery_rate_bounds():
    df = pd.DataFrame(
        {
            "signature_id": [f"s{i}" for i in range(10)],
            "score": np.linspace(0, 1, 10),
            "is_hit": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    res = simulate_false_discovery_rate(df, top_k=3, n_sim=200, random_state=2)
    assert 0.0 <= res["estimated_fdr"] <= 1.0
    assert res["null_distribution"].shape[0] == 200


def test_permutation_significance_test_two_sided():
    group_a = np.array([0.1, 0.2, 0.15, 0.18])
    group_b = np.array([0.3, 0.35, 0.4, 0.45])
    res = permutation_significance_test(
        group_a,
        group_b,
        n_permutations=200,
        random_state=3,
    )
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["null_distribution"].shape == (200,)
