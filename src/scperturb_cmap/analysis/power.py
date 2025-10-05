from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import default_rng

try:  # Optional SciPy dependency
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover - SciPy may be unavailable
    spearmanr = None  # type: ignore

from scperturb_cmap.data.signatures import target_from_cells


@dataclass
class SampleSizeResult:
    """Container for sample size estimation outputs."""

    history: pd.DataFrame
    summary: pd.DataFrame
    recommended_size: int
    threshold: float
    baseline_cells: int


def _validate_adata_cluster(
    adata,
    cluster_key: str,
    cluster: str,
    reference: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"cluster_key '{cluster_key}' not present in adata.obs")
    labels = adata.obs[cluster_key].astype(str)
    target_cells = labels.index[labels == str(cluster)].to_numpy()
    if reference == "rest":
        ref_cells = labels.index[labels != str(cluster)].to_numpy()
    else:
        ref_cells = labels.index[labels == str(reference)].to_numpy()
    if target_cells.size == 0:
        raise ValueError(f"No cells found for cluster '{cluster}'")
    if ref_cells.size == 0:
        raise ValueError("No reference cells available for the requested comparison")
    return target_cells, ref_cells


def _resolve_correlation(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    metric = metric.lower()

    def _safe_stat(x: np.ndarray) -> float:
        return float(np.nanstd(x, ddof=0))

    if metric == "pearson":
        def pearson(a: np.ndarray, b: np.ndarray) -> float:
            if _safe_stat(a) < 1e-12 or _safe_stat(b) < 1e-12:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        return pearson

    if metric == "spearman":
        if spearmanr is not None:
            def spearman(a: np.ndarray, b: np.ndarray) -> float:
                if _safe_stat(a) < 1e-12 or _safe_stat(b) < 1e-12:
                    return 0.0
                corr = spearmanr(a, b).correlation
                return float(corr) if not np.isnan(corr) else 0.0

            return spearman

        def fallback(a: np.ndarray, b: np.ndarray) -> float:
            if _safe_stat(a) < 1e-12 or _safe_stat(b) < 1e-12:
                return 0.0
            a_rank = pd.Series(a).rank(method="average").to_numpy()
            b_rank = pd.Series(b).rank(method="average").to_numpy()
            corr = np.corrcoef(a_rank, b_rank)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0

        return fallback

    if metric == "cosine":
        def cosine(a: np.ndarray, b: np.ndarray) -> float:
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom < 1e-12:
                return 0.0
            return float(np.dot(a, b) / denom)

        return cosine

    raise ValueError(f"Unsupported correlation metric: {metric}")


def estimate_signature_sample_size(
    adata,
    cluster_key: str,
    cluster: str,
    reference: str = "rest",
    *,
    sample_sizes: Optional[Sequence[int]] = None,
    replicates: int = 50,
    method: str = "rank_biserial",
    pseudobulk_key: Optional[str] = None,
    correlation_metric: str = "spearman",
    threshold: float = 0.7,
    random_state: Optional[int] = None,
) -> SampleSizeResult:
    """Estimate how many cells are needed to recover a stable signature.

    The function bootstraps subsamples of the target cluster and compares each
    resulting signature with the full cohort signature using the chosen
    correlation metric. It returns the full sampling history, summary statistics,
    and the smallest sample size meeting the desired threshold.
    """

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if threshold <= -1 or threshold > 1:
        raise ValueError("threshold must be within (-1, 1]")

    target_cells, ref_cells = _validate_adata_cluster(adata, cluster_key, cluster, reference)
    baseline_signature = target_from_cells(
        adata,
        target_cells.tolist(),
        ref_cells.tolist(),
        method=method,
        pseudobulk_key=pseudobulk_key,
    )
    baseline_weights = np.asarray(baseline_signature.weights, dtype=float)

    max_target = int(target_cells.size)
    if sample_sizes is None:
        grid = np.linspace(max(1, max_target // 5), max_target, num=min(7, max_target))
        sample_sizes = sorted({int(max(1, round(x))) for x in grid})
    filtered_sizes = sorted({int(s) for s in sample_sizes if 1 <= int(s) <= max_target})
    if not filtered_sizes:
        raise ValueError("No valid sample sizes after filtering against available cells")

    corr_fn = _resolve_correlation(correlation_metric)
    rng = default_rng(random_state)
    history_rows: List[Dict[str, Union[int, float]]] = []

    for size in filtered_sizes:
        for rep in range(replicates):
            target_sample = rng.choice(target_cells, size=size, replace=False).tolist()
            if not target_sample:
                continue
            ref_replace = size > ref_cells.size
            ref_sample = rng.choice(ref_cells, size=size, replace=ref_replace).tolist()
            sampled_signature = target_from_cells(
                adata,
                target_sample,
                ref_sample,
                method=method,
                pseudobulk_key=pseudobulk_key,
            )
            sample_weights = np.asarray(sampled_signature.weights, dtype=float)
            corr = corr_fn(baseline_weights, sample_weights)
            history_rows.append(
                {
                    "sample_size": size,
                    "replicate": rep,
                    "correlation": float(corr),
                    "abs_correlation": float(abs(corr)),
                    "reference_size": int(size),
                }
            )

    history = pd.DataFrame(history_rows)
    if history.empty:
        raise RuntimeError("No bootstrap samples were generated; check inputs")

    summary = (
        history
        .groupby("sample_size", sort=True)
        .agg(
            median_correlation=("correlation", "median"),
            mean_correlation=("correlation", "mean"),
            std_correlation=("correlation", "std"),
            median_abs_correlation=("abs_correlation", "median"),
        )
        .reset_index()
    )
    summary["std_correlation"] = summary["std_correlation"].fillna(0.0)

    meets = summary[summary["median_correlation"] >= threshold]
    if not meets.empty:
        recommended = int(meets.iloc[0]["sample_size"])
    else:
        recommended = int(summary.iloc[-1]["sample_size"])

    return SampleSizeResult(
        history=history,
        summary=summary,
        recommended_size=recommended,
        threshold=threshold,
        baseline_cells=max_target,
    )


def _resolve_aggregate(
    agg: Union[str, Callable[[np.ndarray, int], np.ndarray]],
) -> Callable[[np.ndarray, int], np.ndarray]:
    if callable(agg):
        return agg

    agg = agg.lower()
    if agg == "mean":
        return lambda x, axis: np.nanmean(x, axis=axis)
    if agg == "median":
        return lambda x, axis: np.nanmedian(x, axis=axis)
    if agg == "sum":
        return lambda x, axis: np.nansum(x, axis=axis)
    raise ValueError(f"Unsupported aggregate: {agg}")


def bootstrap_rank_confidence(
    df: pd.DataFrame,
    *,
    id_col: str = "signature_id",
    score_col: str = "score",
    replicate_col: str = "replicate_id",
    n_boot: int = 1000,
    ci: float = 0.95,
    aggfunc: Union[str, Callable[[np.ndarray, int], np.ndarray]] = "mean",
    ascending: bool = True,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Bootstrap ranking confidence intervals from replicate-level scores."""

    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if ci <= 0 or ci >= 1:
        raise ValueError("ci must be within (0, 1)")
    if id_col not in df.columns or score_col not in df.columns:
        raise KeyError("Required columns not present in dataframe")

    if replicate_col in df.columns:
        pivot = df.pivot_table(
            index=replicate_col,
            columns=id_col,
            values=score_col,
            aggfunc="mean",
        )
    else:
        tmp = df.reset_index(drop=True).copy()
        tmp["_replicate"] = tmp.index.astype(str)
        pivot = tmp.pivot_table(
            index="_replicate",
            columns=id_col,
            values=score_col,
            aggfunc="mean",
        )

    pivot = pivot.sort_index(axis=1)
    if pivot.empty:
        raise ValueError("No data available for bootstrap")

    agg_fn = _resolve_aggregate(aggfunc)
    rng = default_rng(random_state)
    ids = pivot.columns.to_list()
    n_reps = pivot.shape[0]

    observed_scores = agg_fn(pivot.to_numpy(dtype=float), axis=0)
    observed_ranks = _scores_to_ranks(observed_scores, ascending)

    samples: List[np.ndarray] = []
    for _ in range(n_boot):
        indices = rng.integers(0, n_reps, size=n_reps)
        sampled = pivot.to_numpy(dtype=float)[indices]
        aggregated = agg_fn(sampled, axis=0)
        ranks = _scores_to_ranks(aggregated, ascending)
        if np.all(np.isnan(ranks)):
            continue
        samples.append(ranks)

    if not samples:
        raise RuntimeError("All bootstrap samples were invalid (likely due to NaNs)")

    rank_samples = np.vstack(samples)
    lower = (1 - ci) / 2.0
    upper = 1 - lower

    result = pd.DataFrame({id_col: ids})
    result["rank_mean"] = np.nanmean(rank_samples, axis=0)
    result["rank_std"] = np.nanstd(rank_samples, axis=0, ddof=1)
    result["rank_ci_lower"] = np.nanquantile(rank_samples, lower, axis=0)
    result["rank_ci_upper"] = np.nanquantile(rank_samples, upper, axis=0)
    result["observed_rank"] = observed_ranks
    return result


def _scores_to_ranks(scores: np.ndarray, ascending: bool) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    ranks = np.full(scores.shape, np.nan, dtype=float)
    valid = ~np.isnan(scores)
    if not np.any(valid):
        return ranks
    valid_idx = np.where(valid)[0]
    valid_scores = scores[valid_idx]
    order = np.argsort(valid_scores if ascending else -valid_scores)
    rank_values = np.arange(1, order.size + 1, dtype=float)
    ranks_valid = np.empty_like(rank_values)
    ranks_valid[order] = rank_values
    ranks[valid_idx] = ranks_valid
    return ranks


def compute_signature_stability(
    df: pd.DataFrame,
    *,
    signature_col: str = "signature_id",
    replicate_col: str = "replicate_id",
    gene_col: str = "gene_symbol",
    score_col: str = "score",
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute per-signature stability metrics across replicates."""

    required = {signature_col, replicate_col, gene_col, score_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    method = method.lower()
    rows: List[Dict[str, Union[str, int, float]]] = []

    for sig, grp in df.groupby(signature_col):
        pivot = grp.pivot_table(
            index=replicate_col,
            columns=gene_col,
            values=score_col,
            aggfunc="mean",
        )
        mat = pivot.to_numpy(dtype=float)
        n_rep = mat.shape[0]
        if n_rep < 2:
            rows.append(
                {
                    signature_col: sig,
                    "n_replicates": n_rep,
                    "mean_correlation": np.nan,
                    "median_correlation": np.nan,
                    "min_correlation": np.nan,
                    "std_correlation": np.nan,
                    "stability_score": 1.0 if n_rep == 1 else np.nan,
                }
            )
            continue

        if method in {"pearson", "spearman", "kendall"}:
            corr_df = pivot.transpose().corr(method=method)
            corr_mat = corr_df.to_numpy(dtype=float)
        elif method == "cosine":
            denom = np.linalg.norm(mat, axis=1, keepdims=True)
            denom = np.where(denom < 1e-12, 1e-12, denom)
            normed = mat / denom
            corr_mat = normed @ normed.T
        else:
            raise ValueError(f"Unsupported method: {method}")

        iu = np.triu_indices_from(corr_mat, k=1)
        pairwise = corr_mat[iu]
        rows.append(
            {
                signature_col: sig,
                "n_replicates": n_rep,
                "mean_correlation": float(np.nanmean(pairwise)) if pairwise.size else np.nan,
                "median_correlation": float(np.nanmedian(pairwise)) if pairwise.size else np.nan,
                "min_correlation": float(np.nanmin(pairwise)) if pairwise.size else np.nan,
                "std_correlation": float(np.nanstd(pairwise, ddof=0)) if pairwise.size else np.nan,
                "stability_score": float(np.nanmean(pairwise)) if pairwise.size else np.nan,
            }
        )

    return pd.DataFrame(rows)


def recommend_min_cells_per_cluster(
    adata,
    cluster_key: str,
    *,
    reference: str = "rest",
    sample_sizes: Optional[Sequence[int]] = None,
    replicates: int = 50,
    threshold: float = 0.7,
    method: str = "rank_biserial",
    pseudobulk_key: Optional[str] = None,
    correlation_metric: str = "spearman",
    random_state: Optional[int] = None,
) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """Recommend minimum target cells per cluster to reach the desired stability."""

    if cluster_key not in adata.obs.columns:
        raise KeyError(f"cluster_key '{cluster_key}' not present in adata.obs")

    labels = adata.obs[cluster_key].astype(str)
    unique_clusters = sorted(labels.unique())
    rng = default_rng(random_state)

    recommendations: List[Dict[str, Union[str, int, float]]] = []
    histories: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, pd.DataFrame] = {}

    for cluster in unique_clusters:
        cluster_mask = labels == cluster
        n_cells = int(cluster_mask.sum())
        local_seed = None if random_state is None else int(rng.integers(0, 1_000_000_000))
        try:
            result = estimate_signature_sample_size(
                adata,
                cluster_key=cluster_key,
                cluster=cluster,
                reference=reference,
                sample_sizes=sample_sizes,
                replicates=replicates,
                method=method,
                pseudobulk_key=pseudobulk_key,
                correlation_metric=correlation_metric,
                threshold=threshold,
                random_state=local_seed,
            )
        except Exception as exc:
            recommendations.append(
                {
                    "cluster": cluster,
                    "n_cells": n_cells,
                    "recommended_cells": np.nan,
                    "median_correlation": np.nan,
                    "status": "error",
                    "message": str(exc),
                }
            )
            continue

        histories[cluster] = result.history
        summaries[cluster] = result.summary
        medians = result.summary.set_index("sample_size")["median_correlation"]
        med_corr = float(medians.get(result.recommended_size, np.nan))
        recommendations.append(
            {
                "cluster": cluster,
                "n_cells": n_cells,
                "recommended_cells": int(result.recommended_size),
                "median_correlation": med_corr,
                "status": "ok",
                "message": "",
            }
        )

    return {
        "recommendations": pd.DataFrame(recommendations),
        "histories": histories,
        "summaries": summaries,
    }


def simulate_false_discovery_rate(
    df: pd.DataFrame,
    *,
    score_col: str = "score",
    label_col: str = "is_hit",
    top_k: int = 50,
    n_sim: int = 1000,
    ascending: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Union[float, np.ndarray, int]]:
    """Estimate the false discovery rate via label permutation simulations."""

    if score_col not in df.columns or label_col not in df.columns:
        raise KeyError("score_col and label_col must be present in dataframe")
    if n_sim <= 0:
        raise ValueError("n_sim must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    ranked = df.sort_values(score_col, ascending=ascending).reset_index(drop=True)
    k = min(top_k, len(ranked))
    if k == 0:
        raise ValueError("Ranking dataframe is empty")

    labels = ranked[label_col].astype(int).to_numpy()
    observed_hits = int(labels[:k].sum())
    rng = default_rng(random_state)
    null_hits = np.empty(n_sim, dtype=float)

    for i in range(n_sim):
        perm = rng.permutation(labels)
        null_hits[i] = perm[:k].sum()

    expected_false = float(np.mean(null_hits))
    if observed_hits == 0:
        estimated_fdr = 1.0
    else:
        estimated_fdr = float(min(1.0, expected_false / observed_hits))

    p_value = float((np.sum(null_hits >= observed_hits) + 1) / (n_sim + 1))

    return {
        "top_k": int(k),
        "observed_hits": float(observed_hits),
        "expected_false_positives": expected_false,
        "estimated_fdr": estimated_fdr,
        "null_distribution": null_hits,
        "p_value": p_value,
    }


def permutation_significance_test(
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    statistic: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "difference_in_means",
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
    alternative: str = "two-sided",
) -> Dict[str, Union[float, np.ndarray, str]]:
    """Permutation test for comparing two groups of scores."""

    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")

    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("Both groups must contain at least one observation")

    stat_fn = _resolve_statistic(statistic)
    observed = stat_fn(a, b)

    combined = np.concatenate([a, b])
    n_a = a.size
    rng = default_rng(random_state)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm = rng.permutation(combined)
        null[i] = stat_fn(perm[:n_a], perm[n_a:])

    alternative = alternative.lower()
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    if alternative == "two-sided":
        p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_permutations + 1))
    elif alternative == "greater":
        p = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    else:
        p = float((np.sum(null <= observed) + 1) / (n_permutations + 1))

    return {
        "observed_statistic": float(observed),
        "p_value": p,
        "null_distribution": null,
        "statistic": stat_fn.__name__ if hasattr(stat_fn, "__name__") else str(statistic),
    }


def _resolve_statistic(
    statistic: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    if callable(statistic):
        return statistic

    stat = statistic.lower()
    if stat == "difference_in_means":
        def diff_means(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.nanmean(a) - np.nanmean(b))

        diff_means.__name__ = "difference_in_means"
        return diff_means

    if stat == "difference_in_medians":
        def diff_medians(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.nanmedian(a) - np.nanmedian(b))

        diff_medians.__name__ = "difference_in_medians"
        return diff_medians

    if stat == "cohens_d":
        def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
            mean_diff = np.nanmean(a) - np.nanmean(b)
            var_a = np.nanvar(a, ddof=1)
            var_b = np.nanvar(b, ddof=1)
            pooled = ((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2)
            pooled = max(pooled, 1e-12)
            return float(mean_diff / np.sqrt(pooled))

        cohens_d.__name__ = "cohens_d"
        return cohens_d

    raise ValueError(f"Unsupported statistic: {statistic}")


__all__ = [
    "SampleSizeResult",
    "estimate_signature_sample_size",
    "bootstrap_rank_confidence",
    "compute_signature_stability",
    "recommend_min_cells_per_cluster",
    "simulate_false_discovery_rate",
    "permutation_significance_test",
]
