"""
Uncertainty quantification and cell-line-specific predictions
Provides confidence intervals and prediction reliability metrics
"""
import warnings
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class UncertaintyQuantifier:
    """
    Quantifies uncertainty in drug predictions using bootstrap and cross-validation
    """
    
    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95):
        """
        Initialize uncertainty quantifier
        
        Args:
            n_bootstrap: Number of bootstrap iterations
            confidence_level: Confidence level for intervals (default: 95%)
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def bootstrap_scoring(
        self,
        target_signature: np.ndarray,
        drug_signature: np.ndarray,
        scoring_func: Callable,
        n_iterations: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Bootstrap resampling to estimate score uncertainty
        
        Args:
            target_signature: Target gene weights
            drug_signature: Drug gene weights
            scoring_func: Function that computes connectivity score
            n_iterations: Number of bootstrap iterations (uses self.n_bootstrap if None)
        
        Returns:
            Dictionary with mean, std, CI bounds
        """
        n_iter = n_iterations or self.n_bootstrap
        n_genes = len(target_signature)
        
        scores = []
        for _ in range(n_iter):
            # Resample genes with replacement
            indices = np.random.choice(n_genes, size=n_genes, replace=True)
            
            target_boot = target_signature[indices]
            drug_boot = drug_signature[indices]
            
            # Compute score
            score = scoring_func(target_boot, drug_boot)
            scores.append(score)
        
        scores = np.array(scores)
        
        # Calculate statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Confidence interval
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        ci_lower = np.percentile(scores, lower_percentile)
        ci_upper = np.percentile(scores, upper_percentile)
        
        return {
            'mean': mean_score,
            'std': std_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'cv': std_score / abs(mean_score) if mean_score != 0 else np.inf
        }
    
    def compute_confidence_intervals(
        self,
        scores: np.ndarray,
        method: str = 'percentile'
    ) -> Tuple[float, float]:
        """
        Compute confidence intervals for array of scores
        
        Args:
            scores: Array of scores (e.g., from bootstrap)
            method: 'percentile' or 'normal'
        
        Returns:
            (lower_bound, upper_bound)
        """
        if method == 'percentile':
            lower_percentile = (self.alpha / 2) * 100
            upper_percentile = (1 - self.alpha / 2) * 100
            ci_lower = np.percentile(scores, lower_percentile)
            ci_upper = np.percentile(scores, upper_percentile)
        
        elif method == 'normal':
            mean = np.mean(scores)
            std = np.std(scores)
            z_score = stats.norm.ppf(1 - self.alpha / 2)
            ci_lower = mean - z_score * std
            ci_upper = mean + z_score * std
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return ci_lower, ci_upper
    
    def jackknife_variance(
        self,
        target_signature: np.ndarray,
        drug_signature: np.ndarray,
        scoring_func: Callable
    ) -> Dict[str, float]:
        """
        Jackknife resampling for variance estimation
        
        Faster than bootstrap, useful for large gene sets
        
        Args:
            target_signature: Target weights
            drug_signature: Drug weights
            scoring_func: Scoring function
        
        Returns:
            Dictionary with variance estimates
        """
        n_genes = len(target_signature)
        
        # Compute full score
        full_score = scoring_func(target_signature, drug_signature)
        
        # Leave-one-out scores
        loo_scores = []
        for i in range(n_genes):
            mask = np.ones(n_genes, dtype=bool)
            mask[i] = False
            
            target_loo = target_signature[mask]
            drug_loo = drug_signature[mask]
            
            score_loo = scoring_func(target_loo, drug_loo)
            loo_scores.append(score_loo)
        
        loo_scores = np.array(loo_scores)
        
        # Jackknife variance
        mean_loo = np.mean(loo_scores)
        variance = (n_genes - 1) / n_genes * np.sum((loo_scores - mean_loo)**2)
        std = np.sqrt(variance)
        
        return {
            'score': full_score,
            'variance': variance,
            'std': std,
            'cv': std / abs(full_score) if full_score != 0 else np.inf
        }


def bootstrap_scoring(
    target_signature: np.ndarray,
    drug_signature: np.ndarray,
    scoring_func: Callable,
    n_iterations: Optional[int] = None,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """Convenience wrapper around :class:`UncertaintyQuantifier`."""

    quantifier = UncertaintyQuantifier(confidence_level=confidence_level)
    return quantifier.bootstrap_scoring(
        target_signature,
        drug_signature,
        scoring_func,
        n_iterations=n_iterations,
    )


def compute_confidence_intervals(
    scores: np.ndarray,
    method: str = 'percentile',
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """Compute confidence intervals using the default quantifier."""

    quantifier = UncertaintyQuantifier(confidence_level=confidence_level)
    return quantifier.compute_confidence_intervals(scores, method=method)


def cell_line_specific_predictions(
    target_signature: Dict[str, float],
    drug_signatures_by_cell_line: Dict[str, Dict[str, Dict[str, float]]],
    scoring_func: Callable,
    compute_uncertainty: bool = True
) -> pd.DataFrame:
    """
    Generate cell-line-specific predictions with confidence intervals
    
    Args:
        target_signature: Target gene weights
        drug_signatures_by_cell_line: Nested dict: cell_line -> drug -> {gene: weight}
        scoring_func: Function to compute connectivity score
        compute_uncertainty: Whether to compute bootstrap CIs
    
    Returns:
        DataFrame with cell-line-specific predictions and confidence intervals
    """
    quantifier = UncertaintyQuantifier()
    
    results = []
    
    for cell_line, drug_sigs in drug_signatures_by_cell_line.items():
        for drug_name, drug_sig in drug_sigs.items():
            # Get common genes
            common_genes = sorted(set(target_signature.keys()) & set(drug_sig.keys()))
            
            if len(common_genes) < 50:
                warnings.warn(
                    f"Insufficient gene overlap for {drug_name} in {cell_line}: "
                    f"{len(common_genes)} genes"
                )
                continue
            
            # Convert to arrays
            target_array = np.array([target_signature[g] for g in common_genes])
            drug_array = np.array([drug_sig[g] for g in common_genes])
            
            # Compute score
            score = scoring_func(target_array, drug_array)
            
            # Compute uncertainty if requested
            if compute_uncertainty:
                uncertainty = quantifier.bootstrap_scoring(
                    target_array,
                    drug_array,
                    scoring_func,
                    n_iterations=500  # Reduced for speed
                )
                
                results.append({
                    'cell_line': cell_line,
                    'drug': drug_name,
                    'score': score,
                    'score_mean': uncertainty['mean'],
                    'score_std': uncertainty['std'],
                    'ci_lower': uncertainty['ci_lower'],
                    'ci_upper': uncertainty['ci_upper'],
                    'ci_width': uncertainty['ci_width'],
                    'cv': uncertainty['cv'],
                    'n_genes': len(common_genes)
                })
            else:
                results.append({
                    'cell_line': cell_line,
                    'drug': drug_name,
                    'score': score,
                    'n_genes': len(common_genes)
                })
    
    return pd.DataFrame(results)


def aggregate_cell_line_predictions(
    cell_line_results: pd.DataFrame,
    aggregation: str = 'mean',
    weight_by_n_genes: bool = True
) -> pd.DataFrame:
    """
    Aggregate predictions across cell lines
    
    Args:
        cell_line_results: DataFrame from cell_line_specific_predictions
        aggregation: 'mean', 'median', or 'weighted_mean'
        weight_by_n_genes: Weight by number of genes (for weighted_mean)
    
    Returns:
        DataFrame with aggregated predictions per drug
    """
    if aggregation == 'mean':
        agg_func = {
            'score': 'mean',
            'score_std': 'mean',
            'n_genes': 'mean',
            'cell_line': lambda x: ', '.join(sorted(set(x)))
        }
        aggregated = cell_line_results.groupby('drug').agg(agg_func).reset_index()
    
    elif aggregation == 'median':
        agg_func = {
            'score': 'median',
            'score_std': 'median',
            'n_genes': 'median',
            'cell_line': lambda x: ', '.join(sorted(set(x)))
        }
        aggregated = cell_line_results.groupby('drug').agg(agg_func).reset_index()
    
    elif aggregation == 'weighted_mean':
        # Weight by number of genes
        def weighted_mean(group):
            if weight_by_n_genes:
                weights = group['n_genes'].values
            else:
                weights = np.ones(len(group))
            
            weights = weights / weights.sum()
            
            return pd.Series({
                'score': np.average(group['score'], weights=weights),
                'score_std': np.average(group['score_std'], weights=weights),
                'n_genes': group['n_genes'].mean(),
                'cell_lines': ', '.join(sorted(set(group['cell_line']))),
                'n_cell_lines': len(group)
            })
        
        aggregated = cell_line_results.groupby('drug').apply(weighted_mean).reset_index()
    
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    return aggregated


def compute_prediction_reliability(
    cell_line_results: pd.DataFrame,
    min_cell_lines: int = 2,
    max_cv: float = 0.5
) -> pd.DataFrame:
    """
    Assess prediction reliability based on consistency across cell lines
    
    Args:
        cell_line_results: Cell-line-specific predictions
        min_cell_lines: Minimum number of cell lines for high confidence
        max_cv: Maximum coefficient of variation for high confidence
    
    Returns:
        DataFrame with reliability scores
    """
    reliability_stats = []
    
    for drug, group in cell_line_results.groupby('drug'):
        n_cell_lines = len(group)
        
        # Compute cross-cell-line statistics
        mean_score = group['score'].mean()
        std_score = group['score'].std()
        cv_score = std_score / abs(mean_score) if mean_score != 0 else np.inf
        
        # Range
        score_range = group['score'].max() - group['score'].min()
        
        # Consistency (inverse of CV)
        consistency = 1 / (1 + cv_score)
        
        # Reliability score (combines consistency and replication)
        reliability = consistency * min(n_cell_lines / min_cell_lines, 1.0)
        
        # Confidence level
        if n_cell_lines >= min_cell_lines and cv_score <= max_cv:
            confidence = 'high'
        elif n_cell_lines >= 2 and cv_score <= 0.75:
            confidence = 'moderate'
        else:
            confidence = 'low'
        
        reliability_stats.append({
            'drug': drug,
            'n_cell_lines': n_cell_lines,
            'mean_score': mean_score,
            'std_score': std_score,
            'cv': cv_score,
            'score_range': score_range,
            'consistency': consistency,
            'reliability_score': reliability,
            'confidence_level': confidence
        })
    
    return pd.DataFrame(reliability_stats)


def create_uncertainty_plot(
    cell_line_results: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    output_path: Optional[str] = None
) -> 'Figure':
    """
    Create plot showing predictions with confidence intervals across cell lines
    
    Args:
        cell_line_results: Cell-line-specific predictions with CIs
        top_n: Number of top drugs to show
        figsize: Figure size
        output_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Get top drugs by mean score
    top_drugs = (
        cell_line_results.groupby('drug')['score']
        .mean()
        .sort_values()
        .head(top_n)
        .index
    )
    
    # Filter to top drugs
    plot_data = cell_line_results[cell_line_results['drug'].isin(top_drugs)].copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot with error bars
    for i, drug in enumerate(top_drugs):
        drug_data = plot_data[plot_data['drug'] == drug]
        
        # Plot points for each cell line
        x_positions = np.full(len(drug_data), i) + np.random.normal(0, 0.1, len(drug_data))
        
        ax.scatter(
            x_positions,
            drug_data['score'],
            alpha=0.6,
            s=100,
            c=range(len(drug_data)),
            cmap='tab10'
        )
        
        # Plot confidence intervals if available
        if 'ci_lower' in drug_data.columns:
            for j, (_, row) in enumerate(drug_data.iterrows()):
                ax.plot(
                    [x_positions[j], x_positions[j]],
                    [row['ci_lower'], row['ci_upper']],
                    'k-',
                    alpha=0.3,
                    linewidth=2
                )
        
        # Plot mean
        mean_score = drug_data['score'].mean()
        ax.plot(
            [i-0.3, i+0.3],
            [mean_score, mean_score],
            'r-',
            linewidth=3,
            alpha=0.7
        )
    
    # Formatting
    ax.set_xticks(range(len(top_drugs)))
    ax.set_xticklabels(top_drugs, rotation=45, ha='right')
    ax.set_ylabel('Connectivity Score', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Cell-Line-Specific Predictions (Top {top_n} Drugs)\n'
        'Points = individual cell lines, Red lines = mean, Black bars = 95% CI',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig
