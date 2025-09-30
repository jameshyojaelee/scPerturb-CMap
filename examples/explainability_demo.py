"""
Comprehensive demo of scPerturb-CMap explainability features

This script demonstrates:
1. SHAP-like gene contribution analysis
2. Waterfall plots showing gene-level drivers
3. Pathway enrichment with GO/KEGG/Reactome
4. Automated narrative generation
5. Cell-line-specific predictions with confidence intervals
6. Drug A vs Drug B comparison mode
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# scPerturb-CMap imports
from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.io.schemas import TargetSignature

# Explainability imports
from scperturb_cmap.explainability.feature_importance import (
    GeneContributionAnalyzer,
    create_waterfall_plot,
    compare_drug_contributions,
    explain_drug_ranking,
)
from scperturb_cmap.explainability.pathway_enrichment import (
    PathwayEnricher,
    integrate_go_kegg_reactome,
    create_enrichment_barplot,
    visualize_pathway_network,
)
from scperturb_cmap.explainability.narrative_generator import (
    generate_drug_narrative,
    create_comparison_narrative,
    generate_batch_narratives,
)
from scperturb_cmap.explainability.uncertainty import (
    UncertaintyQuantifier,
    cell_line_specific_predictions,
    compute_prediction_reliability,
    create_uncertainty_plot,
)


def demo_gene_contributions():
    """
    Demo 1: Gene-level contribution analysis with waterfall plots
    """
    print("=" * 80)
    print("DEMO 1: Gene-Level Contribution Analysis")
    print("=" * 80)
    
    # Load example data
    target_signature = TargetSignature.from_json('examples/out/target.json')
    library = pd.read_parquet('examples/data/lincs_demo.parquet')
    
    # Get top drug from scoring
    result = rank_drugs(
        target_signature=target_signature,
        library=library,
        method='baseline',
        top_k=10
    )
    
    top_drug = result.ranking.iloc[0]
    drug_name = top_drug['compound']
    
    print(f"\nAnalyzing top-ranked drug: {drug_name}")
    print(f"Connectivity score: {top_drug['score']:.4f}")
    print(f"P-value: {top_drug.get('p_value', 'N/A')}")
    
    # Get drug signature from library
    drug_sig = library[library['compound'] == drug_name]
    drug_weights = dict(zip(drug_sig['gene_symbol'], drug_sig['score']))
    
    # Compute gene contributions
    analyzer = GeneContributionAnalyzer()
    
    common_genes = sorted(set(target_signature.genes) & set(drug_weights.keys()))
    target_array = np.array([target_signature.weights[target_signature.genes.index(g)] 
                             for g in common_genes])
    drug_array = np.array([drug_weights[g] for g in common_genes])
    
    contributions = analyzer.compute_contributions(
        target_array, drug_array, common_genes
    )
    
    print(f"\nTop 10 Contributing Genes:")
    print(contributions[['gene', 'contribution', 'direction']].head(10))
    
    # Create waterfall plot
    fig = create_waterfall_plot(
        contributions,
        drug_name=drug_name,
        top_n=20,
        output_path='examples/out/waterfall_top_drug.png'
    )
    print(f"\nWaterfall plot saved to: examples/out/waterfall_top_drug.png")
    plt.close(fig)
    
    return contributions, drug_name


def demo_pathway_enrichment(contributions, drug_name):
    """
    Demo 2: Pathway enrichment analysis
    """
    print("\n" + "=" * 80)
    print("DEMO 2: Pathway Enrichment Analysis")
    print("=" * 80)
    
    # Get top contributing genes
    top_beneficial = contributions[contributions['contribution'] > 0].head(50)
    gene_list = top_beneficial['gene'].tolist()
    
    print(f"\nEnriching {len(gene_list)} top-contributing genes...")
    
    # Integrated enrichment
    enrichment_results = integrate_go_kegg_reactome(
        gene_list,
        top_n_pathways=10,
        p_threshold=0.05
    )
    
    # Display results
    for database, results in enrichment_results.items():
        print(f"\n{database} Results:")
        if not results.empty:
            print(results[['pathway', 'q_value', 'overlap']].head(5))
            
            # Create barplot
            fig = create_enrichment_barplot(
                results,
                top_n=10,
                output_path=f'examples/out/enrichment_{database}_{drug_name}.png'
            )
            plt.close(fig)
        else:
            print("  No significant enrichment")
    
    # Create pathway network
    if enrichment_results.get('GO_BP') is not None and not enrichment_results['GO_BP'].empty:
        fig = visualize_pathway_network(
            enrichment_results['GO_BP'],
            top_n=15,
            output_path=f'examples/out/pathway_network_{drug_name}.png'
        )
        if fig:
            print(f"\nPathway network saved to: examples/out/pathway_network_{drug_name}.png")
            plt.close(fig)
    
    return enrichment_results


def demo_narrative_generation(contributions, enrichment_results, drug_name):
    """
    Demo 3: Automated narrative generation
    """
    print("\n" + "=" * 80)
    print("DEMO 3: Automated Narrative Generation")
    print("=" * 80)
    
    # Generate narrative
    from scperturb_cmap.explainability.narrative_generator import DrugNarrativeGenerator
    
    generator = DrugNarrativeGenerator()
    
    # Get metadata (mock for demo)
    metadata = {
        'moa': 'HDAC inhibitor',
        'target': 'HDAC1, HDAC2, HDAC3',
        'cell_line': 'MCF7',
        'literature': [
            'Smith et al. Nature 2020',
            'Jones et al. Cell 2021'
        ]
    }
    
    narrative = generator.generate_narrative(
        drug_name=drug_name,
        rank=1,
        score=-3.45,
        p_value=0.0023,
        contributions=contributions,
        enrichment_results=enrichment_results.get('GO_BP'),
        moa=metadata['moa'],
        targets=metadata['target'],
        cell_line=metadata['cell_line'],
        literature_refs=metadata['literature']
    )
    
    print(f"\nAutomated Explanation for {drug_name}:")
    print("-" * 80)
    print(narrative)
    print("-" * 80)
    
    # Save narrative
    with open(f'examples/out/narrative_{drug_name}.txt', 'w') as f:
        f.write(f"Automated Explanation: {drug_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(narrative)
    
    print(f"\nNarrative saved to: examples/out/narrative_{drug_name}.txt")
    
    return narrative


def demo_drug_comparison():
    """
    Demo 4: Drug A vs Drug B comparison mode
    """
    print("\n" + "=" * 80)
    print("DEMO 4: Drug A vs Drug B Comparison")
    print("=" * 80)
    
    # Load data
    target_signature = TargetSignature.from_json('examples/out/target.json')
    library = pd.read_parquet('examples/data/lincs_demo.parquet')
    
    # Get top 2 drugs
    result = rank_drugs(target_signature, library, method='baseline', top_k=10)
    drug_a = result.ranking.iloc[0]
    drug_b = result.ranking.iloc[1]
    
    drug_a_name = drug_a['compound']
    drug_b_name = drug_b['compound']
    
    print(f"\nComparing:")
    print(f"  Drug A: {drug_a_name} (rank #1, score={drug_a['score']:.4f})")
    print(f"  Drug B: {drug_b_name} (rank #2, score={drug_b['score']:.4f})")
    
    # Get signatures
    drug_a_sig = library[library['compound'] == drug_a_name]
    drug_b_sig = library[library['compound'] == drug_b_name]
    
    drug_a_weights = dict(zip(drug_a_sig['gene_symbol'], drug_a_sig['score']))
    drug_b_weights = dict(zip(drug_b_sig['gene_symbol'], drug_b_sig['score']))
    
    # Get common genes
    common_genes = sorted(
        set(target_signature.genes) & 
        set(drug_a_weights.keys()) & 
        set(drug_b_weights.keys())
    )
    
    target_array = np.array([
        target_signature.weights[target_signature.genes.index(g)] 
        for g in common_genes
    ])
    drug_a_array = np.array([drug_a_weights[g] for g in common_genes])
    drug_b_array = np.array([drug_b_weights[g] for g in common_genes])
    
    # Create comparison plot
    fig, comparison_df = compare_drug_contributions(
        target_array,
        drug_a_array,
        drug_b_array,
        common_genes,
        drug_a_name,
        drug_b_name,
        top_n=15
    )
    
    plt.savefig('examples/out/drug_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: examples/out/drug_comparison.png")
    plt.close(fig)
    
    # Generate comparison narrative
    analyzer = GeneContributionAnalyzer()
    contrib_a = analyzer.compute_contributions(target_array, drug_a_array, common_genes)
    contrib_b = analyzer.compute_contributions(target_array, drug_b_array, common_genes)
    
    narrative = create_comparison_narrative(
        drug_a_name, drug_b_name,
        rank_a=1, rank_b=2,
        score_a=drug_a['score'], score_b=drug_b['score'],
        contributions_a=contrib_a,
        contributions_b=contrib_b,
        comparison_df=comparison_df
    )
    
    print(f"\nComparison Narrative:")
    print("-" * 80)
    print(narrative)
    print("-" * 80)
    
    return comparison_df, narrative


def demo_uncertainty_quantification():
    """
    Demo 5: Cell-line-specific predictions with confidence intervals
    """
    print("\n" + "=" * 80)
    print("DEMO 5: Uncertainty Quantification & Cell-Line-Specific Predictions")
    print("=" * 80)
    
    # Load data
    target_signature = TargetSignature.from_json('examples/out/target.json')
    library = pd.read_parquet('examples/data/lincs_demo.parquet')
    
    # Create target dict
    target_dict = dict(zip(target_signature.genes, target_signature.weights))
    
    # Organize drug signatures by cell line
    drug_sigs_by_cell = {}
    for cell_line in library['cell_line'].unique()[:3]:  # Limit for demo
        cell_data = library[library['cell_line'] == cell_line]
        
        drug_sigs_by_cell[cell_line] = {}
        for compound in cell_data['compound'].unique()[:5]:  # Top 5 drugs per cell line
            comp_data = cell_data[cell_data['compound'] == compound]
            drug_sigs_by_cell[cell_line][compound] = dict(
                zip(comp_data['gene_symbol'], comp_data['score'])
            )
    
    # Simple scoring function
    def scoring_func(target, drug):
        return -np.corrcoef(target, drug)[0, 1]  # Negative correlation
    
    # Compute cell-line-specific predictions
    print("\nComputing cell-line-specific predictions with bootstrap CIs...")
    cell_line_results = cell_line_specific_predictions(
        target_dict,
        drug_sigs_by_cell,
        scoring_func,
        compute_uncertainty=True
    )
    
    print(f"\nCell-Line-Specific Results (top 10):")
    print(cell_line_results.head(10)[
        ['drug', 'cell_line', 'score', 'ci_lower', 'ci_upper', 'cv']
    ])
    
    # Compute reliability scores
    reliability = compute_prediction_reliability(cell_line_results)
    print(f"\nPrediction Reliability Scores:")
    print(reliability.head(10)[
        ['drug', 'n_cell_lines', 'mean_score', 'cv', 'confidence_level']
    ])
    
    # Create uncertainty plot
    fig = create_uncertainty_plot(
        cell_line_results,
        top_n=10,
        output_path='examples/out/uncertainty_plot.png'
    )
    print(f"\nUncertainty plot saved to: examples/out/uncertainty_plot.png")
    plt.close(fig)
    
    return cell_line_results, reliability


def run_complete_demo():
    """
    Run complete explainability demo
    """
    print("\n")
    print("=" * 80)
    print("scPerturb-CMap Explainability Framework Demo")
    print("=" * 80)
    print("\nThis demo showcases:")
    print("  1. Gene-level contribution analysis (SHAP-like)")
    print("  2. Pathway enrichment (GO/KEGG/Reactome)")
    print("  3. Automated narrative generation")
    print("  4. Drug A vs Drug B comparison")
    print("  5. Uncertainty quantification with confidence intervals")
    print("\n" + "=" * 80)
    
    # Create output directory
    Path('examples/out').mkdir(parents=True, exist_ok=True)
    
    # Run demos
    contributions, drug_name = demo_gene_contributions()
    enrichment_results = demo_pathway_enrichment(contributions, drug_name)
    narrative = demo_narrative_generation(contributions, enrichment_results, drug_name)
    comparison_df, comp_narrative = demo_drug_comparison()
    cell_line_results, reliability = demo_uncertainty_quantification()
    
    # Summary
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  - examples/out/waterfall_top_drug.png")
    print("  - examples/out/enrichment_*.png")
    print("  - examples/out/pathway_network_*.png")
    print("  - examples/out/narrative_*.txt")
    print("  - examples/out/drug_comparison.png")
    print("  - examples/out/uncertainty_plot.png")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    run_complete_demo()
