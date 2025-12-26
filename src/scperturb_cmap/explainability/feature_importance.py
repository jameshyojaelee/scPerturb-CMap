"""
Gene-level feature importance and SHAP-like contribution analysis
Explains which genes drive drug rankings and their individual contributions
"""
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scperturb_cmap.data.preprocess import align_vectors, harmonize_symbols, standardize_vector
from scperturb_cmap.io.schemas import TargetSignature


def _first_occurrence_map(genes: Sequence[str]) -> Dict[str, int]:
    """Return the first index for each harmonized gene symbol."""
    lookup: Dict[str, int] = {}
    for idx, g in enumerate(harmonize_symbols(genes)):
        if g not in lookup:
            lookup[g] = idx
    return lookup


def _extract_signature(
    signature: Sequence[float] | Dict[str, float] | TargetSignature,
    gene_names: Optional[List[str]],
    role: str,
) -> Tuple[List[str], np.ndarray]:
    """Normalize a signature payload into (genes, values)."""
    if isinstance(signature, TargetSignature):
        return list(signature.genes), np.asarray(signature.weights, dtype=float)
    if isinstance(signature, dict):
        genes = list(signature.keys())
        values = np.asarray(list(signature.values()), dtype=float)
        return genes, values
    # Sequence-like values
    if gene_names is None:
        raise ValueError(f"gene_names must be provided when {role} signature is a vector")
    values = np.asarray(signature, dtype=float)
    if len(values) != len(gene_names):
        raise ValueError(f"{role} signature length ({len(values)}) does not match gene_names ({len(gene_names)})")
    return list(gene_names), values


def _cosine_component(target_vec: np.ndarray, drug_vec: np.ndarray) -> Tuple[np.ndarray, float]:
    """Cosine connectivity contributions (score = cosine).

    Under the library scoring convention, lower is better (more negative implies
    stronger inversion / anti-correlation).
    """
    t_std = standardize_vector(target_vec)
    d_std = standardize_vector(drug_vec)
    eps = max(np.finfo(float).eps, 1e-12)
    denom = (np.linalg.norm(t_std) + eps) * (np.linalg.norm(d_std) + eps)
    contrib = (t_std * d_std) / denom
    return contrib, float(contrib.sum())


def _gsea_running_contrib(ranked_genes: List[str], gene_set: set[str]) -> Tuple[List[float], float]:
    """Return per-position contributions and ES for a gene set."""
    N = len(ranked_genes)
    if N == 0 or not gene_set:
        return [0.0] * N, 0.0
    hits = [g in gene_set for g in ranked_genes]
    Nh = sum(hits)
    if Nh == 0:
        return [0.0] * N, 0.0
    Nm = N - Nh
    phit = 1.0 / Nh
    pmiss = 1.0 / Nm if Nm > 0 else 0.0
    running = 0.0
    best = 0.0
    worst = 0.0
    best_idx = -1
    worst_idx = -1
    increments: List[float] = []
    for i, h in enumerate(hits):
        delta = phit if h else -pmiss
        running += delta
        increments.append(delta)
        if running > best:
            best = running
            best_idx = i
        if running < worst:
            worst = running
            worst_idx = i
    if abs(best) >= abs(worst):
        es = best
        limit = best_idx
    else:
        es = worst
        limit = worst_idx
    contrib = [increments[i] if i <= limit else 0.0 for i in range(N)]
    return contrib, es


def _gsea_component(
    target_genes: List[str],
    target_vals: np.ndarray,
    drug_genes: List[str],
    drug_vals: np.ndarray,
    aligned_genes: List[str],
) -> Tuple[np.ndarray, float]:
    """GSEA-style contributions aligned to ``aligned_genes``."""
    t_map = _first_occurrence_map(target_genes)
    d_map = _first_occurrence_map(drug_genes)
    target_weights = np.asarray(target_vals, dtype=float)
    drug_scores = np.asarray(drug_vals, dtype=float)

    up = {g for g, idx in t_map.items() if target_weights[idx] > 0}
    down = {g for g, idx in t_map.items() if target_weights[idx] < 0}
    drug_lookup = {g: drug_scores[d_map[g]] for g in d_map}

    ranked = sorted(aligned_genes, key=lambda g: drug_lookup.get(g, 0.0), reverse=True)
    contrib_up, es_up = _gsea_running_contrib(ranked, up)
    contrib_down, es_down = _gsea_running_contrib(ranked, down)
    combined_ranked = [0.5 * (u - d) for u, d in zip(contrib_up, contrib_down)]
    contrib_map = {g: c for g, c in zip(ranked, combined_ranked)}
    aligned_contrib = np.array([contrib_map.get(g, 0.0) for g in aligned_genes], dtype=float)
    return aligned_contrib, float(0.5 * (es_up - es_down))


def _scale_contributions(
    combined: np.ndarray,
    target_score: Optional[float],
    component_sets: List[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Scale contributions to match a target score (if provided)."""
    if target_score is None:
        return combined, component_sets
    total = float(combined.sum())
    eps = 1e-12
    if abs(total) < eps:
        # If contributions cancel out, attribute the full score to the top contributor
        filler = np.zeros_like(combined)
        idx = int(np.argmax(np.abs(combined))) if combined.size else 0
        filler[idx] = float(target_score)
        return filler, [filler.copy() for _ in component_sets]
    scale = float(target_score) / total
    return combined * scale, [comp * scale for comp in component_sets]


class GeneContributionAnalyzer:
    """
    Analyzes individual gene contributions to drug connectivity scores
    Implements SHAP-like decomposition of ranking scores
    """
    
    def compute_contributions(
        self,
        target_signature: Sequence[float] | Dict[str, float] | TargetSignature,
        drug_signature: Sequence[float] | Dict[str, float],
        gene_names: Optional[List[str]] = None,
        *,
        method: str = "cosine",
        blend_weight: float = 0.5,
        target_score: Optional[float] = None,
        model_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute gene-level contributions to connectivity score
        
        Args:
            target_signature: Target vector, mapping, or TargetSignature
            drug_signature: Drug vector or mapping (gene -> score)
            gene_names: Optional reference gene order (e.g., from pivoted library)
            method: 'cosine', 'baseline' (cosine + GSEA), 'gsea', or 'metric'
            blend_weight: Weight for the GSEA component when method='baseline'
            target_score: Optional final score to rescale contributions to
            model_path: DualEncoder checkpoint for method='metric'
        
        Returns:
            DataFrame with gene contributions aligned to the scoring function
        """
        target_genes, target_values = _extract_signature(target_signature, gene_names, "target")
        drug_genes, drug_values = _extract_signature(drug_signature, gene_names, "drug")
        ref_genes = gene_names if gene_names else sorted(set(target_genes) | set(drug_genes))
        t_aligned, d_aligned, common = align_vectors(ref_genes, target_values, drug_genes, drug_values)
        if len(common) == 0:
            raise ValueError("No overlapping genes between target and drug signatures")

        label_map = {}
        for g in target_genes + drug_genes:
            h = harmonize_symbols([g])[0]
            if h not in label_map:
                label_map[h] = g
        display_genes = [label_map.get(g, g) for g in common]

        method_lower = method.lower()
        if method_lower == "metric":
            alpha = float(blend_weight)
            metric_df = self._metric_contributions(
                target_genes,
                target_values,
                drug_genes,
                drug_values,
                common,
                display_genes,
                target_score=None,
                model_path=model_path,
            )
            # Pure metric path
            if alpha >= 1.0 or model_path is None:
                if target_score is not None:
                    scaled, scaled_parts = _scale_contributions(
                        metric_df["contribution"].to_numpy(),
                        target_score,
                        [metric_df["contribution"].to_numpy()],
                    )
                    metric_df["contribution"] = scaled
                    metric_df["abs_contribution"] = np.abs(metric_df["contribution"])
                    metric_df["direction"] = np.where(
                        metric_df["contribution"] < 0, "helps", "hurts"
                    )
                    metric_df.attrs["score_estimate"] = float(target_score)
                return metric_df

            # Blend baseline (cosine+GSEA) with metric similarity, mirroring scoring
            baseline_df = self.compute_contributions(
                target_signature=t_aligned,
                drug_signature=d_aligned,
                gene_names=list(common),
                method="baseline",
                blend_weight=0.5,
                target_score=None,
            )
            baseline_small = baseline_df[
                ["harmonized_gene", "gene", "target_weight", "drug_weight", "contribution"]
            ].rename(
                columns={
                    "gene": "gene_base",
                    "target_weight": "target_weight_base",
                    "drug_weight": "drug_weight_base",
                    "contribution": "baseline_component",
                }
            )
            metric_small = metric_df[
                ["harmonized_gene", "gene", "target_weight", "drug_weight", "contribution"]
            ].rename(
                columns={
                    "gene": "gene_metric",
                    "target_weight": "target_weight_metric",
                    "drug_weight": "drug_weight_metric",
                    "contribution": "metric_component",
                }
            )
            merged = baseline_small.merge(metric_small, on="harmonized_gene", how="outer")
            merged["gene"] = merged["gene_base"].combine_first(merged["gene_metric"]).fillna(
                merged["harmonized_gene"]
            )
            merged["target_weight"] = merged["target_weight_base"].combine_first(
                merged["target_weight_metric"]
            )
            merged["drug_weight"] = merged["drug_weight_base"].combine_first(
                merged["drug_weight_metric"]
            )
            merged["baseline_component"] = merged["baseline_component"].fillna(0.0)
            merged["metric_component"] = merged["metric_component"].fillna(0.0)

            combined = (
                (1.0 - alpha) * merged["baseline_component"].to_numpy()
                + alpha * merged["metric_component"].to_numpy()
            )
            combined_scaled, scaled_components = _scale_contributions(
                combined,
                target_score,
                [
                    merged["baseline_component"].to_numpy(),
                    merged["metric_component"].to_numpy(),
                ],
            )
            merged["contribution"] = combined_scaled
            merged["baseline_component"] = scaled_components[0]
            merged["metric_component"] = (
                scaled_components[1] if len(scaled_components) > 1 else scaled_components[0]
            )
            merged["abs_contribution"] = np.abs(merged["contribution"])
            merged["direction"] = np.where(merged["contribution"] < 0, "helps", "hurts")
            merged = merged.sort_values("abs_contribution", ascending=False).reset_index(drop=True)
            merged["rank"] = range(1, len(merged) + 1)
            merged.attrs["score_estimate"] = float(
                target_score if target_score is not None else merged["contribution"].sum()
            )
            return merged[
                [
                    "gene",
                    "harmonized_gene",
                    "target_weight",
                    "drug_weight",
                    "contribution",
                    "abs_contribution",
                    "direction",
                    "baseline_component",
                    "metric_component",
                    "rank",
                ]
            ]

        components: List[np.ndarray] = []
        combined: np.ndarray
        score_estimate: Optional[float] = None

        cosine_comp = None
        gsea_comp = None
        cosine_score = None
        gsea_score = None

        if method_lower in {"cosine", "baseline"}:
            cosine_comp, cosine_score = _cosine_component(t_aligned, d_aligned)
            components.append(cosine_comp)

        if method_lower in {"gsea", "baseline"}:
            gsea_comp, gsea_score = _gsea_component(
                target_genes,
                target_values,
                drug_genes,
                drug_values,
                common,
            )
            components.append(gsea_comp)

        if method_lower == "cosine":
            combined = cosine_comp
            score_estimate = cosine_score
        elif method_lower == "gsea":
            combined = gsea_comp
            score_estimate = gsea_score
        elif method_lower == "baseline":
            alpha = float(blend_weight)
            if gsea_comp is None:
                combined = cosine_comp
                score_estimate = cosine_score
            else:
                combined = (1.0 - alpha) * cosine_comp + alpha * gsea_comp
                c_score = cosine_score if cosine_score is not None else 0.0
                g_score = gsea_score if gsea_score is not None else 0.0
                score_estimate = (1.0 - alpha) * c_score + alpha * g_score
        else:
            raise ValueError("method must be one of ['cosine', 'baseline', 'gsea', 'metric']")

        combined_scaled, scaled_components = _scale_contributions(combined, target_score, components)
        if target_score is not None:
            score_estimate = target_score
        cosine_scaled = None
        gsea_scaled = None
        if method_lower in {"cosine", "baseline"}:
            cosine_scaled = scaled_components[0]
        if method_lower in {"baseline", "gsea"} and gsea_comp is not None:
            gsea_scaled = scaled_components[-1] if len(scaled_components) > 1 else scaled_components[0]

        contrib_df = pd.DataFrame(
            {
                "gene": display_genes,
                "harmonized_gene": common,
                "target_weight": t_aligned,
                "drug_weight": d_aligned,
                "contribution": combined_scaled,
                "abs_contribution": np.abs(combined_scaled),
            }
        )
        if cosine_scaled is not None:
            contrib_df["cosine_component"] = cosine_scaled
        if gsea_scaled is not None:
            contrib_df["gsea_component"] = gsea_scaled

        contrib_df["direction"] = np.where(
            contrib_df["contribution"] < 0, "helps", "hurts"
        )
        contrib_df = contrib_df.sort_values("abs_contribution", ascending=False)
        contrib_df["rank"] = range(1, len(contrib_df) + 1)
        if score_estimate is not None:
            contrib_df.attrs["score_estimate"] = float(score_estimate)
        return contrib_df

    def _metric_contributions(
        self,
        target_genes: List[str],
        target_vals: np.ndarray,
        drug_genes: List[str],
        drug_vals: np.ndarray,
        aligned_genes: List[str],
        display_genes: List[str],
        *,
        target_score: Optional[float],
        model_path: Optional[str],
    ) -> pd.DataFrame:
        """Compute contributions using the DualEncoder similarity."""
        if model_path is None:
            raise ValueError("model_path is required for method='metric'")
        import torch  # defer import for optional dependency
        from scperturb_cmap.models.dual_encoder import DualEncoder

        target_aligned, drug_aligned, _ = align_vectors(aligned_genes, target_vals, drug_genes, drug_vals)
        if len(target_aligned) == 0:
            raise ValueError("No overlapping genes between target and drug signatures")

        ckpt = torch.load(model_path, map_location="cpu")
        input_dim = int(ckpt.get("config", {}).get("input_dim", len(aligned_genes)))
        model = DualEncoder(input_dim=input_dim, embed_dim=64)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        model.eval()

        left_vec = standardize_vector(target_aligned)
        right_vec = -np.asarray(drug_aligned, dtype=float)

        if left_vec.size < input_dim:
            pad = np.zeros(input_dim - left_vec.size, dtype=float)
            left_vec = np.concatenate([left_vec, pad])
        elif left_vec.size > input_dim:
            left_vec = left_vec[:input_dim]

        if right_vec.size < input_dim:
            pad_r = np.zeros(input_dim - right_vec.size, dtype=float)
            right_vec = np.concatenate([right_vec, pad_r])
        elif right_vec.size > input_dim:
            right_vec = right_vec[:input_dim]

        left = torch.tensor(left_vec, dtype=torch.float32).unsqueeze(0)
        right = torch.tensor(right_vec, dtype=torch.float32, requires_grad=True).unsqueeze(0)

        with torch.enable_grad():
            zL, zR, _ = model(left, right)
            zL = zL / (zL.norm(p=2, dim=-1, keepdim=True) + 1e-12)
            zR = zR / (zR.norm(p=2, dim=-1, keepdim=True) + 1e-12)
            sim = (zR @ zL.squeeze(0)).squeeze()
            score = -sim
            score.backward()

        grad = right.grad.detach().cpu().numpy().squeeze(0)
        contrib_full = grad * right.detach().cpu().numpy().squeeze(0)
        contrib_aligned = contrib_full[: len(aligned_genes)]
        score_est = float(score.detach().cpu().item())

        contrib_final, _ = _scale_contributions(contrib_aligned, target_score, [contrib_aligned])
        if target_score is not None:
            score_est = target_score

        contrib_df = pd.DataFrame(
            {
                "gene": display_genes,
                "harmonized_gene": aligned_genes,
                "target_weight": target_aligned,
                "drug_weight": drug_aligned,
                "contribution": contrib_final,
                "abs_contribution": np.abs(contrib_final),
                "direction": np.where(contrib_final < 0, "helps", "hurts"),
                "cosine_component": contrib_final,
                "metric_component": contrib_final,
            }
        )
        contrib_df = contrib_df.sort_values("abs_contribution", ascending=False)
        contrib_df["rank"] = range(1, len(contrib_df) + 1)
        contrib_df.attrs["score_estimate"] = float(score_est)
        return contrib_df
    
    def identify_key_genes(
        self,
        contributions: pd.DataFrame,
        top_n: int = 20,
        min_abs_contribution: float = 0.01
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify key genes driving the drug ranking
        
        Args:
            contributions: Gene contribution DataFrame
            top_n: Number of top genes to return
            min_abs_contribution: Minimum absolute contribution threshold
        
        Returns:
            (positive_drivers, negative_drivers) DataFrames
        """
        # Filter by minimum contribution
        sig_genes = contributions[contributions['abs_contribution'] >= min_abs_contribution]
        
        # Negative contributions lower the score (better rank)
        positive = sig_genes[sig_genes['contribution'] < 0].head(top_n)
        negative = sig_genes[sig_genes['contribution'] > 0].head(top_n)
        
        return positive, negative
    
    def compute_feature_importance(
        self,
        target_signature: Dict[str, float],
        drug_signatures: Dict[str, Dict[str, float]],
        top_k_drugs: int = 50
    ) -> pd.DataFrame:
        """
        Compute aggregate feature importance across top-ranked drugs
        
        Similar to permutation importance, measures how often each gene
        contributes to top drug rankings
        
        Args:
            target_signature: Dict mapping gene -> weight
            drug_signatures: Dict mapping drug_name -> {gene: weight}
            top_k_drugs: Number of top drugs to analyze
        
        Returns:
            DataFrame with gene importance metrics
        """
        # Get common genes
        target_genes = set(target_signature.keys())
        
        # Track gene importance across drugs
        gene_importance = {gene: {
            'mean_contribution': 0.0,
            'frequency_top_contributor': 0,
            'total_helpful': 0.0,
            'total_harmful': 0.0,
            'n_drugs': 0
        } for gene in target_genes}
        
        # Analyze each drug
        for drug_name, drug_sig in list(drug_signatures.items())[:top_k_drugs]:
            # Get common genes for this drug
            common_genes = target_genes & set(drug_sig.keys())
            
            # Convert to arrays
            genes_list = sorted(common_genes)
            target_array = np.array([target_signature[g] for g in genes_list])
            drug_array = np.array([drug_sig[g] for g in genes_list])
            
            # Compute contributions
            contrib_df = self.compute_contributions(
                target_array, drug_array, genes_list
            )
            
            # Update importance metrics
            for _, row in contrib_df.iterrows():
                gene = row['gene']
                gene_importance[gene]['mean_contribution'] += row['contribution']
                gene_importance[gene]['n_drugs'] += 1
                
                # Track if gene is top contributor
                if row['rank'] <= 10:
                    gene_importance[gene]['frequency_top_contributor'] += 1
                
                # Track direction
                if row['contribution'] < 0:
                    gene_importance[gene]['total_helpful'] += abs(row['contribution'])
                else:
                    gene_importance[gene]['total_harmful'] += abs(row['contribution'])
        
        # Convert to DataFrame and compute final metrics
        importance_df = pd.DataFrame.from_dict(gene_importance, orient='index')
        importance_df['gene'] = importance_df.index
        
        # Compute averages
        importance_df['mean_contribution'] = (
            importance_df['mean_contribution'] / importance_df['n_drugs']
        )
        importance_df['frequency_top_contributor'] = (
            importance_df['frequency_top_contributor'] / top_k_drugs
        )
        
        # Overall importance score (combines magnitude and frequency)
        importance_df['importance_score'] = (
            importance_df['mean_contribution'].abs() * 
            importance_df['frequency_top_contributor']
        )
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance_score', ascending=False)
        
        return importance_df.reset_index(drop=True)


def create_waterfall_plot(
    contributions: pd.DataFrame,
    drug_name: str,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create waterfall plot showing gene-level contributions to drug score
    
    Visualizes how individual genes contribute to the overall connectivity score,
    similar to SHAP waterfall plots
    
    Args:
        contributions: Gene contribution DataFrame
        drug_name: Name of the drug for title
        top_n: Number of top genes to show
        figsize: Figure size
        output_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Select top N genes by absolute contribution
    top_genes = contributions.nlargest(top_n, 'abs_contribution')
    
    # Reverse order for bottom-to-top plotting
    top_genes = top_genes.iloc[::-1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors: score-lowering (blue), score-raising (red)
    colors = ['#2E86AB' if c < 0 else '#A23B72' for c in top_genes['contribution']]
    
    # Create bars
    y_positions = np.arange(len(top_genes))
    
    # Plot horizontal bars
    bars = ax.barh(
        y_positions,
        top_genes['contribution'].values,
        color=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add value labels
    for i, (bar, contrib) in enumerate(zip(bars, top_genes['contribution'].values)):
        # Determine label position
        if contrib > 0:
            ha = 'left'
            x_pos = contrib + 0.02
        else:
            ha = 'right'
            x_pos = contrib - 0.02
        
        ax.text(
            x_pos, i, f'{contrib:.3f}',
            va='center', ha=ha, fontsize=9, fontweight='bold'
        )
    
    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(top_genes['gene'].values, fontsize=10)
    ax.set_xlabel('Contribution to Connectivity Score', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Gene Contributions to {drug_name} Ranking\n(Top {top_n} Genes)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', alpha=0.7, label='Helps ranking (score ↓)'),
        Patch(facecolor='#A23B72', alpha=0.7, label='Hurts ranking (score ↑)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Calculate total contribution
    total_contrib = top_genes['contribution'].sum()
    ax.text(
        0.02, 0.98, f'Total contribution (top {top_n}): {total_contrib:.3f}',
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_drug_contributions(
    target_signature: np.ndarray,
    drug_a_signature: np.ndarray,
    drug_b_signature: np.ndarray,
    gene_names: List[str],
    drug_a_name: str,
    drug_b_name: str,
    top_n: int = 15,
    figsize: Tuple[int, int] = (14, 8),
    method: str = "cosine",
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Compare gene contributions between two drugs to explain ranking differences
    
    Creates side-by-side comparison showing why Drug A ranks higher than Drug B
    
    Args:
        target_signature: Target weights
        drug_a_signature: Drug A weights
        drug_b_signature: Drug B weights
        gene_names: Gene symbols
        drug_a_name: Name of Drug A
        drug_b_name: Name of Drug B
        top_n: Number of genes to show
        figsize: Figure size
    
    Returns:
        (figure, comparison_dataframe)
    """
    analyzer = GeneContributionAnalyzer()
    
    # Compute contributions for both drugs
    contrib_a = analyzer.compute_contributions(
        target_signature, drug_a_signature, gene_names, method=method
    )
    contrib_b = analyzer.compute_contributions(
        target_signature, drug_b_signature, gene_names, method=method
    )
    
    # Merge and compute differences
    comparison = contrib_a[['gene', 'contribution']].merge(
        contrib_b[['gene', 'contribution']],
        on='gene',
        suffixes=('_a', '_b')
    )
    comparison['contribution_diff'] = comparison['contribution_a'] - comparison['contribution_b']
    comparison['abs_diff'] = comparison['contribution_diff'].abs()
    
    # Get top differentiating genes
    top_diff = comparison.nlargest(top_n, 'abs_diff')
    
    # Create comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Drug A contributions
    top_genes_a = contrib_a.head(top_n).iloc[::-1]
    colors_a = ['#2E86AB' if c < 0 else '#A23B72' for c in top_genes_a['contribution']]
    y_pos = np.arange(len(top_genes_a))
    ax1.barh(y_pos, top_genes_a['contribution'], color=colors_a, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_genes_a['gene'], fontsize=9)
    ax1.set_xlabel('Contribution', fontweight='bold')
    ax1.set_title(f'{drug_a_name}\n(Higher Ranked)', fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Drug B contributions
    top_genes_b = contrib_b.head(top_n).iloc[::-1]
    colors_b = ['#2E86AB' if c < 0 else '#A23B72' for c in top_genes_b['contribution']]
    ax2.barh(y_pos, top_genes_b['contribution'], color=colors_b, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_genes_b['gene'], fontsize=9)
    ax2.set_xlabel('Contribution', fontweight='bold')
    ax2.set_title(f'{drug_b_name}\n(Lower Ranked)', fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: Difference (why A > B)
    top_diff_plot = top_diff.iloc[::-1]
    colors_diff = ['#27AE60' if d > 0 else '#E74C3C' for d in top_diff_plot['contribution_diff']]
    y_pos_diff = np.arange(len(top_diff_plot))
    ax3.barh(y_pos_diff, top_diff_plot['contribution_diff'], color=colors_diff, alpha=0.7)
    ax3.set_yticks(y_pos_diff)
    ax3.set_yticklabels(top_diff_plot['gene'], fontsize=9)
    ax3.set_xlabel('Contribution Difference (A - B)', fontweight='bold')
    ax3.set_title('Key Differentiating Genes\n(Why A ranks higher)', fontweight='bold')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    return fig, comparison


def rank_gene_importance(
    contributions_list: List[pd.DataFrame],
    drug_names: List[str],
    top_n: int = 50
) -> pd.DataFrame:
    """
    Aggregate gene importance across multiple drugs
    
    Identifies genes that consistently drive connectivity scores
    
    Args:
        contributions_list: List of contribution DataFrames (one per drug)
        drug_names: Corresponding drug names
        top_n: Number of top genes to return
    
    Returns:
        DataFrame with aggregated importance scores
    """
    # Collect all genes
    all_genes = set()
    for contrib in contributions_list:
        all_genes.update(contrib['gene'].values)
    
    # Initialize importance tracking
    gene_stats = {gene: {
        'mean_contribution': [],
        'mean_abs_contribution': [],
        'frequency_helpful': 0,
        'frequency_harmful': 0,
        'frequency_top10': 0,
        'n_drugs': 0
    } for gene in all_genes}
    
    # Aggregate across drugs
    for contrib, drug_name in zip(contributions_list, drug_names):
        for _, row in contrib.iterrows():
            gene = row['gene']
            gene_stats[gene]['mean_contribution'].append(row['contribution'])
            gene_stats[gene]['mean_abs_contribution'].append(row['abs_contribution'])
            gene_stats[gene]['n_drugs'] += 1
            
            if row['contribution'] < 0:
                gene_stats[gene]['frequency_helpful'] += 1
            else:
                gene_stats[gene]['frequency_harmful'] += 1
            
            if row['rank'] <= 10:
                gene_stats[gene]['frequency_top10'] += 1
    
    # Convert to DataFrame
    importance_rows = []
    for gene, stats in gene_stats.items():
        if stats['n_drugs'] == 0:
            continue
        
            importance_rows.append({
            'gene': gene,
            'mean_contribution': np.mean(stats['mean_contribution']),
            'std_contribution': np.std(stats['mean_contribution']),
            'mean_abs_contribution': np.mean(stats['mean_abs_contribution']),
            'frequency_helpful': stats['frequency_helpful'] / stats['n_drugs'],
            'frequency_harmful': stats['frequency_harmful'] / stats['n_drugs'],
            'frequency_top10': stats['frequency_top10'] / len(contributions_list),
            'n_drugs_present': stats['n_drugs'],
            'consistency': 1
            - np.std(stats['mean_contribution'])
            / (np.abs(np.mean(stats['mean_contribution'])) + 1e-10)
        })
    
    importance_df = pd.DataFrame(importance_rows)
    
    # Overall importance score
    importance_df['importance_score'] = (
        importance_df['mean_abs_contribution'] * 
        importance_df['frequency_top10'] *
        importance_df['consistency']
    )
    
    # Sort and return top N
    importance_df = importance_df.sort_values('importance_score', ascending=False)
    
    return importance_df.head(top_n)


def compute_gene_contributions(
    target_signature: Sequence[float],
    drug_signature: Sequence[float],
    genes: List[str],
    **kwargs: object,
) -> pd.DataFrame:
    """Convenience wrapper returning gene contributions for a drug signature."""

    analyzer = GeneContributionAnalyzer()
    return analyzer.compute_contributions(
        np.asarray(target_signature, dtype=float),
        np.asarray(drug_signature, dtype=float),
        genes,
        **kwargs,
    )


# Convenience function for single drug analysis
def explain_drug_ranking(
    target_signature: Dict[str, float],
    drug_signature: Dict[str, float],
    drug_name: str,
    top_n: int = 20,
    create_plot: bool = True,
    output_dir: Optional[str] = None,
    method: str = "cosine",
    blend_weight: float = 0.5,
    target_score: Optional[float] = None,
    model_path: Optional[str] = None,
) -> Dict:
    """
    Complete explanation of why a drug achieved its ranking
    
    Args:
        target_signature: Target gene weights
        drug_signature: Drug gene weights
        drug_name: Drug name
        top_n: Number of top genes to analyze
        create_plot: Whether to create waterfall plot
        output_dir: Directory to save outputs
    
    Returns:
        Dictionary with contributions, key genes, and statistics
    """
    # Get common genes
    common_genes = sorted(set(target_signature.keys()) & set(drug_signature.keys()))
    
    if len(common_genes) == 0:
        raise ValueError("No common genes between target and drug signatures")
    
    # Convert to arrays
    target_array = np.array([target_signature[g] for g in common_genes])
    drug_array = np.array([drug_signature[g] for g in common_genes])
    
    # Compute contributions
    analyzer = GeneContributionAnalyzer()
    contributions = analyzer.compute_contributions(
        target_array,
        drug_array,
        common_genes,
        method=method,
        blend_weight=blend_weight,
        target_score=target_score,
        model_path=model_path,
    )
    
    # Identify key genes
    positive_genes, negative_genes = analyzer.identify_key_genes(
        contributions, top_n=top_n
    )
    
    # Create plot if requested
    fig = None
    if create_plot:
        output_path = None
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{drug_name}_waterfall.png')
        
        fig = create_waterfall_plot(
            contributions, drug_name, top_n=top_n, output_path=output_path
        )
    
    # Compile results
    results = {
        'drug_name': drug_name,
        'total_genes': len(common_genes),
        'contributions': contributions,
        'positive_drivers': positive_genes,
        'negative_drivers': negative_genes,
        'summary_stats': {
            'total_contribution': contributions['contribution'].sum(),
            'mean_contribution': contributions['contribution'].mean(),
            'n_score_lowering': (contributions['contribution'] < 0).sum(),
            'n_score_raising': (contributions['contribution'] > 0).sum(),
            'top_gene': contributions.iloc[0]['gene'],
            'top_contribution': contributions.iloc[0]['contribution']
        },
        'figure': fig
    }
    
    return results
