"""
Pathway enrichment analysis for top-contributing genes
Integrates GO, KEGG, and Reactome databases to provide biological context
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False
    warnings.warn("gseapy not available. Install with: pip install gseapy")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

import matplotlib.pyplot as plt


class PathwayEnricher:
    """
    Enrichment analysis for gene sets using multiple pathway databases
    """
    
    SUPPORTED_LIBRARIES = [
        'GO_Biological_Process_2021',
        'GO_Molecular_Function_2021',
        'GO_Cellular_Component_2021',
        'KEGG_2021_Human',
        'Reactome_2022',
        'WikiPathways_2021_Human',
        'MSigDB_Hallmark_2020',
    ]
    
    def __init__(self, organism: str = 'human'):
        """
        Initialize pathway enricher
        
        Args:
            organism: Organism name ('human', 'mouse')
        """
        self.organism = organism
        self.enrichment_cache = {}
    
    def enrich_genes(
        self,
        gene_list: List[str],
        background: Optional[List[str]] = None,
        libraries: Optional[List[str]] = None,
        p_threshold: float = 0.05,
        min_genes: int = 3,
        max_genes: int = 500
    ) -> pd.DataFrame:
        """
        Perform pathway enrichment analysis
        
        Args:
            gene_list: List of gene symbols to test
            background: Background gene list (if None, uses all genes)
            libraries: Pathway libraries to query (default: GO + KEGG + Reactome)
            p_threshold: P-value threshold
            min_genes: Minimum genes per pathway
            max_genes: Maximum genes per pathway
        
        Returns:
            DataFrame with enrichment results
        """
        if not GSEAPY_AVAILABLE:
            raise ImportError("gseapy is required for pathway enrichment")
        
        # Default libraries
        if libraries is None:
            libraries = [
                'GO_Biological_Process_2021',
                'KEGG_2021_Human',
                'Reactome_2022'
            ]
        
        # Check libraries
        invalid = [lib for lib in libraries if lib not in self.SUPPORTED_LIBRARIES]
        if invalid:
            warnings.warn(f"Unsupported libraries: {invalid}")
            libraries = [lib for lib in libraries if lib in self.SUPPORTED_LIBRARIES]
        
        if not libraries:
            raise ValueError("No valid libraries specified")
        
        # Deduplicate genes
        gene_list = list(set(gene_list))
        
        if len(gene_list) < min_genes:
            warnings.warn(f"Gene list too small ({len(gene_list)} < {min_genes})")
            return pd.DataFrame()
        
        # Perform enrichment for each library
        all_results = []
        
        for library in libraries:
            try:
                # Run Enrichr
                enr = gp.enrichr(
                    gene_list=gene_list,
                    gene_sets=library,
                    organism=self.organism.capitalize(),
                    background=background,
                    cutoff=p_threshold,
                    no_plot=True
                )
                
                if enr.results.empty:
                    continue
                
                # Add library info
                enr.results['library'] = library
                all_results.append(enr.results)
                
            except Exception as e:
                warnings.warn(f"Enrichment failed for {library}: {e}")
                continue
        
        if not all_results:
            return pd.DataFrame()
        
        # Combine results
        combined = pd.concat(all_results, ignore_index=True)
        
        # Filter by gene count
        combined = combined[
            (combined['Overlap'].str.split('/').str[0].astype(int) >= min_genes) &
            (combined['Overlap'].str.split('/').str[1].astype(int) <= max_genes)
        ]
        
        # Sort by p-value
        combined = combined.sort_values('Adjusted P-value')
        
        # Clean up column names
        combined = combined.rename(columns={
            'Adjusted P-value': 'q_value',
            'P-value': 'p_value',
            'Overlap': 'overlap',
            'Odds Ratio': 'odds_ratio',
            'Combined Score': 'combined_score',
            'Genes': 'genes',
            'Term': 'pathway'
        })
        
        return combined
    
    def enrich_by_direction(
        self,
        positive_genes: List[str],
        negative_genes: List[str],
        libraries: Optional[List[str]] = None,
        p_threshold: float = 0.05
    ) -> Dict[str, pd.DataFrame]:
        """
        Separate enrichment for positively and negatively contributing genes
        
        Args:
            positive_genes: Genes with beneficial contributions
            negative_genes: Genes with detrimental contributions
            libraries: Pathway libraries to query
            p_threshold: P-value threshold
        
        Returns:
            Dictionary with 'positive' and 'negative' DataFrames
        """
        results = {}
        
        if len(positive_genes) >= 3:
            results['positive'] = self.enrich_genes(
                positive_genes, libraries=libraries, p_threshold=p_threshold
            )
        else:
            results['positive'] = pd.DataFrame()
        
        if len(negative_genes) >= 3:
            results['negative'] = self.enrich_genes(
                negative_genes, libraries=libraries, p_threshold=p_threshold
            )
        else:
            results['negative'] = pd.DataFrame()
        
        return results


def enrich_pathways(
    gene_list: List[str],
    background: Optional[List[str]] = None,
    libraries: Optional[List[str]] = None,
    p_threshold: float = 0.05,
    min_genes: int = 3,
    max_genes: int = 500,
) -> pd.DataFrame:
    """Convenience wrapper that enriches a gene set using default settings."""

    enricher = PathwayEnricher()
    return enricher.enrich_genes(
        gene_list,
        background=background,
        libraries=libraries,
        p_threshold=p_threshold,
        min_genes=min_genes,
        max_genes=max_genes,
    )


def integrate_go_kegg_reactome(
    gene_list: List[str],
    top_n_pathways: int = 20,
    p_threshold: float = 0.05
) -> Dict[str, pd.DataFrame]:
    """
    Integrated pathway enrichment across GO, KEGG, and Reactome
    
    Args:
        gene_list: Genes to analyze
        top_n_pathways: Number of top pathways to return per database
        p_threshold: FDR threshold
    
    Returns:
        Dictionary with results for each database
    """
    enricher = PathwayEnricher()
    
    results = {}
    
    # GO Biological Process
    go_bp = enricher.enrich_genes(
        gene_list,
        libraries=['GO_Biological_Process_2021'],
        p_threshold=p_threshold
    )
    if not go_bp.empty:
        results['GO_BP'] = go_bp.head(top_n_pathways)
    
    # KEGG
    kegg = enricher.enrich_genes(
        gene_list,
        libraries=['KEGG_2021_Human'],
        p_threshold=p_threshold
    )
    if not kegg.empty:
        results['KEGG'] = kegg.head(top_n_pathways)
    
    # Reactome
    reactome = enricher.enrich_genes(
        gene_list,
        libraries=['Reactome_2022'],
        p_threshold=p_threshold
    )
    if not reactome.empty:
        results['Reactome'] = reactome.head(top_n_pathways)
    
    return results


def visualize_pathway_network(
    enrichment_results: pd.DataFrame,
    top_n: int = 15,
    gene_overlap_threshold: float = 0.3,
    figsize: Tuple[int, int] = (12, 10),
    output_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Create network visualization of enriched pathways
    
    Pathways are connected if they share genes above threshold
    
    Args:
        enrichment_results: Enrichment DataFrame
        top_n: Number of pathways to include
        gene_overlap_threshold: Minimum Jaccard similarity to connect pathways
        figsize: Figure size
        output_path: Path to save figure
    
    Returns:
        Matplotlib figure (if networkx available)
    """
    if not NETWORKX_AVAILABLE:
        warnings.warn("NetworkX required for pathway network visualization")
        return None
    
    # Select top pathways
    top_pathways = enrichment_results.head(top_n)
    
    if len(top_pathways) < 2:
        warnings.warn("Not enough pathways for network visualization")
        return None
    
    # Build network
    G = nx.Graph()
    
    # Add nodes (pathways)
    for idx, row in top_pathways.iterrows():
        pathway_genes = set(row['genes'].split(';'))
        G.add_node(
            row['pathway'],
            genes=pathway_genes,
            q_value=row['q_value'],
            size=len(pathway_genes)
        )
    
    # Add edges (gene overlap)
    pathways = list(G.nodes())
    for i, pathway1 in enumerate(pathways):
        for pathway2 in pathways[i+1:]:
            genes1 = G.nodes[pathway1]['genes']
            genes2 = G.nodes[pathway2]['genes']
            
            # Jaccard similarity
            intersection = len(genes1 & genes2)
            union = len(genes1 | genes2)
            jaccard = intersection / union if union > 0 else 0
            
            if jaccard >= gene_overlap_threshold:
                G.add_edge(pathway1, pathway2, weight=jaccard, overlap=intersection)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node sizes (by gene count)
    node_sizes = [G.nodes[node]['size'] * 50 for node in G.nodes()]
    
    # Node colors (by significance)
    node_colors = [-np.log10(G.nodes[node]['q_value'] + 1e-10) for node in G.nodes()]
    
    # Draw network
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap='YlOrRd',
        alpha=0.7,
        edgecolors='black',
        linewidths=2
    )
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=[w*5 for w in weights],
        alpha=0.3,
        edge_color='gray'
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=8,
        font_weight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )
    
    # Colorbar for significance
    sm = plt.cm.ScalarMappable(
        cmap='YlOrRd',
        norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors))
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('-log10(FDR)', fontweight='bold')
    
    ax.set_title(
        'Pathway Enrichment Network\n(Edge width = gene overlap)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_enrichment_barplot(
    enrichment_results: pd.DataFrame,
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create barplot of top enriched pathways
    
    Args:
        enrichment_results: Enrichment DataFrame
        top_n: Number of pathways to show
        figsize: Figure size
        output_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Select top pathways
    top_pathways = enrichment_results.head(top_n).iloc[::-1]  # Reverse for plotting
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate -log10(q-value) for plotting
    neg_log_q = -np.log10(top_pathways['q_value'] + 1e-10)
    
    # Get gene counts
    gene_counts = top_pathways['overlap'].str.split('/').str[0].astype(int)
    
    # Create bars
    bars = ax.barh(
        range(len(top_pathways)),
        neg_log_q,
        color=plt.cm.RdYlBu_r(neg_log_q / neg_log_q.max()),
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add gene count labels
    for i, (bar, count) in enumerate(zip(bars, gene_counts)):
        ax.text(
            bar.get_width() + 0.1, i, f'n={count}',
            va='center', fontsize=9, fontweight='bold'
        )
    
    # Formatting
    ax.set_yticks(range(len(top_pathways)))
    ax.set_yticklabels(top_pathways['pathway'], fontsize=10)
    ax.set_xlabel('-log10(FDR)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Top {top_n} Enriched Pathways',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    
    # Add significance threshold line
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', 
               linewidth=2, label='FDR = 0.05', alpha=0.7)
    ax.legend(loc='lower right')
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def summarize_pathway_biology(
    enrichment_dict: Dict[str, pd.DataFrame],
    top_n_per_category: int = 5
) -> Dict[str, List[str]]:
    """
    Summarize top biological themes across pathway databases
    
    Args:
        enrichment_dict: Dictionary of enrichment results per database
        top_n_per_category: Number of top pathways per category
    
    Returns:
        Dictionary with biological theme summaries
    """
    summary = {}
    
    for database, results in enrichment_dict.items():
        if results.empty:
            continue
        
        top_pathways = results.head(top_n_per_category)
        
        summary[database] = {
            'pathways': top_pathways['pathway'].tolist(),
            'q_values': top_pathways['q_value'].tolist(),
            'gene_counts': [
                int(overlap.split('/')[0]) 
                for overlap in top_pathways['overlap']
            ],
            'representative_genes': [
                genes.split(';')[:5]  # First 5 genes per pathway
                for genes in top_pathways['genes']
            ]
        }
    
    return summary
