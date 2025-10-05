"""
Gene-level feature importance and SHAP-like contribution analysis
Explains which genes drive drug rankings and their individual contributions
"""
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class GeneContributionAnalyzer:
    """
    Analyzes individual gene contributions to drug connectivity scores
    Implements SHAP-like decomposition of ranking scores
    """
    
    def __init__(self, method: str = 'additive'):
        """
        Initialize analyzer
        
        Args:
            method: 'additive' (sum of contributions) or 'multiplicative'
        """
        self.method = method
        self.contributions = {}
        self.importance_rankings = {}
    
    def compute_contributions(
        self,
        target_signature: np.ndarray,
        drug_signature: np.ndarray,
        gene_names: List[str],
        baseline_score: float = 0.0
    ) -> pd.DataFrame:
        """
        Compute gene-level contributions to connectivity score
        
        Similar to SHAP values, each gene gets a contribution value that:
        - Sums to the total connectivity score
        - Represents the gene's individual impact
        - Can be positive (alignment) or negative (anti-alignment)
        
        Args:
            target_signature: Target gene expression weights (n_genes,)
            drug_signature: Drug perturbation weights (n_genes,)
            gene_names: Gene symbols
            baseline_score: Reference score (typically 0)
        
        Returns:
            DataFrame with gene contributions
        """
        # Ensure arrays are numpy
        target = np.asarray(target_signature, dtype=float)
        drug = np.asarray(drug_signature, dtype=float)
        
        if len(target) != len(drug) or len(target) != len(gene_names):
            raise ValueError("Target, drug, and gene_names must have same length")
        
        # Element-wise contribution (for cosine-based scoring)
        # Contribution = target_i * drug_i (negative correlation desired for reversal)
        contributions = -target * drug  # Negative because we want opposite direction
        
        # Normalize by magnitude to get interpretable scale
        # This ensures contributions sum to approximate connectivity score
        total_magnitude = np.sqrt(np.sum(target**2)) * np.sqrt(np.sum(drug**2))
        if total_magnitude > 1e-10:
            contributions = contributions / total_magnitude * np.sum(contributions)
        
        # Create DataFrame
        contrib_df = pd.DataFrame({
            'gene': gene_names,
            'target_weight': target,
            'drug_weight': drug,
            'contribution': contributions,
            'abs_contribution': np.abs(contributions),
            'direction': np.where(contributions > 0, 'beneficial', 'detrimental')
        })
        
        # Rank by absolute contribution
        contrib_df = contrib_df.sort_values('abs_contribution', ascending=False)
        contrib_df['rank'] = range(1, len(contrib_df) + 1)
        
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
        
        # Separate positive (beneficial) and negative (detrimental) contributors
        positive = sig_genes[sig_genes['contribution'] > 0].head(top_n)
        negative = sig_genes[sig_genes['contribution'] < 0].head(top_n)
        
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
            'total_beneficial': 0.0,
            'total_detrimental': 0.0,
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
                if row['contribution'] > 0:
                    gene_importance[gene]['total_beneficial'] += row['contribution']
                else:
                    gene_importance[gene]['total_detrimental'] += abs(row['contribution'])
        
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
    
    # Colors: beneficial (blue), detrimental (red)
    colors = ['#2E86AB' if c > 0 else '#A23B72' for c in top_genes['contribution']]
    
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
        Patch(facecolor='#2E86AB', alpha=0.7, label='Beneficial (Target ↔ Drug alignment)'),
        Patch(facecolor='#A23B72', alpha=0.7, label='Detrimental (Poor alignment)')
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
    figsize: Tuple[int, int] = (14, 8)
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
        target_signature, drug_a_signature, gene_names
    )
    contrib_b = analyzer.compute_contributions(
        target_signature, drug_b_signature, gene_names
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
    colors_a = ['#2E86AB' if c > 0 else '#A23B72' for c in top_genes_a['contribution']]
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
    colors_b = ['#2E86AB' if c > 0 else '#A23B72' for c in top_genes_b['contribution']]
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
        'frequency_positive': 0,
        'frequency_negative': 0,
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
            
            if row['contribution'] > 0:
                gene_stats[gene]['frequency_positive'] += 1
            else:
                gene_stats[gene]['frequency_negative'] += 1
            
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
            'frequency_positive': stats['frequency_positive'] / stats['n_drugs'],
            'frequency_negative': stats['frequency_negative'] / stats['n_drugs'],
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
) -> pd.DataFrame:
    """Convenience wrapper returning gene contributions for a drug signature."""

    analyzer = GeneContributionAnalyzer()
    return analyzer.compute_contributions(
        np.asarray(target_signature, dtype=float),
        np.asarray(drug_signature, dtype=float),
        genes,
    )


# Convenience function for single drug analysis
def explain_drug_ranking(
    target_signature: Dict[str, float],
    drug_signature: Dict[str, float],
    drug_name: str,
    top_n: int = 20,
    create_plot: bool = True,
    output_dir: Optional[str] = None
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
        target_array, drug_array, common_genes
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
            'n_beneficial': (contributions['contribution'] > 0).sum(),
            'n_detrimental': (contributions['contribution'] < 0).sum(),
            'top_gene': contributions.iloc[0]['gene'],
            'top_contribution': contributions.iloc[0]['contribution']
        },
        'figure': fig
    }
    
    return results
