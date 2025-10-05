"""
Explainability API for scPerturb-CMap scoring results
Provides high-level interface for generating explanations
"""
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from scperturb_cmap.explainability.feature_importance import (
    GeneContributionAnalyzer,
    compare_drug_contributions,
    create_waterfall_plot,
)
from scperturb_cmap.explainability.narrative_generator import (
    DrugNarrativeGenerator,
    create_comparison_narrative,
    generate_batch_narratives,
)
from scperturb_cmap.explainability.pathway_enrichment import (
    PathwayEnricher,
    create_enrichment_barplot,
    integrate_go_kegg_reactome,
)
from scperturb_cmap.explainability.uncertainty import (
    UncertaintyQuantifier,
)
from scperturb_cmap.io.schemas import ScoreResult, TargetSignature


class ExplainabilityEngine:
    """
    High-level API for generating explanations for drug rankings
    """
    
    def __init__(self, enable_pathway_enrichment: bool = True):
        """
        Initialize explainability engine
        
        Args:
            enable_pathway_enrichment: Whether to run pathway enrichment
                                      (requires internet access for Enrichr)
        """
        self.analyzer = GeneContributionAnalyzer()
        self.enricher = PathwayEnricher() if enable_pathway_enrichment else None
        self.generator = DrugNarrativeGenerator()
        self.quantifier = UncertaintyQuantifier()
        
        self.enable_pathway_enrichment = enable_pathway_enrichment
    
    def explain_ranking(
        self,
        target_signature: TargetSignature,
        drug_signature: Dict[str, float],
        drug_metadata: Dict,
        output_dir: Optional[str] = None,
        create_plots: bool = True
    ) -> Dict:
        """
        Generate complete explanation for a single drug's ranking
        
        Args:
            target_signature: Target signature object
            drug_signature: Drug perturbation signature (gene -> weight)
            drug_metadata: Metadata dict with compound, rank, score, moa, etc.
            output_dir: Directory to save outputs
            create_plots: Whether to generate plots
        
        Returns:
            Dictionary with all explanation components
        """
        drug_name = drug_metadata.get('compound', 'Unknown')
        
        # Get common genes and convert to arrays
        target_genes = set(target_signature.genes)
        drug_genes = set(drug_signature.keys())
        common_genes = sorted(target_genes & drug_genes)
        
        target_array = np.array([
            target_signature.weights[target_signature.genes.index(g)]
            for g in common_genes
        ])
        drug_array = np.array([drug_signature[g] for g in common_genes])
        
        # 1. Compute gene contributions
        contributions = self.analyzer.compute_contributions(
            target_array, drug_array, common_genes
        )
        
        # 2. Identify key genes
        positive_genes, negative_genes = self.analyzer.identify_key_genes(
            contributions, top_n=50
        )
        
        # 3. Pathway enrichment (if enabled)
        enrichment_results = None
        if self.enable_pathway_enrichment and len(positive_genes) >= 5:
            try:
                enrichment_results = integrate_go_kegg_reactome(
                    positive_genes['gene'].tolist(),
                    top_n_pathways=20
                )
            except Exception as e:
                print(f"Warning: Pathway enrichment failed: {e}")
        
        # 4. Generate narrative
        narrative = self.generator.generate_narrative(
            drug_name=drug_name,
            rank=drug_metadata.get('rank', 0),
            score=drug_metadata.get('score', 0.0),
            p_value=drug_metadata.get('p_value', 1.0),
            contributions=contributions,
            enrichment_results=enrichment_results.get('GO_BP') if enrichment_results else None,
            moa=drug_metadata.get('moa'),
            targets=drug_metadata.get('target'),
            cell_line=drug_metadata.get('cell_line')
        )
        
        # 5. Create visualizations (if requested)
        plots = {}
        if create_plots and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Waterfall plot
            waterfall_path = output_path / f'{drug_name}_waterfall.png'
            waterfall_fig = create_waterfall_plot(
                contributions, drug_name, top_n=20, output_path=str(waterfall_path)
            )
            plots['waterfall'] = waterfall_fig
            
            # Enrichment plots
            if enrichment_results:
                for db_name, db_results in enrichment_results.items():
                    if not db_results.empty:
                        enrich_path = output_path / f'{drug_name}_enrichment_{db_name}.png'
                        enrich_fig = create_enrichment_barplot(
                            db_results, top_n=15, output_path=str(enrich_path)
                        )
                        plots[f'enrichment_{db_name}'] = enrich_fig
        
        # Compile results
        explanation = {
            'drug_name': drug_name,
            'metadata': drug_metadata,
            'contributions': contributions,
            'positive_drivers': positive_genes,
            'negative_drivers': negative_genes,
            'enrichment': enrichment_results,
            'narrative': narrative,
            'plots': plots,
            'summary': {
                'total_genes': len(common_genes),
                'n_beneficial_genes': len(positive_genes),
                'n_detrimental_genes': len(negative_genes),
                'top_gene': contributions.iloc[0]['gene'],
                'top_contribution': contributions.iloc[0]['contribution'],
            }
        }
        
        return explanation
    
    def explain_top_k_drugs(
        self,
        target_signature: TargetSignature,
        score_result: ScoreResult,
        library: pd.DataFrame,
        top_k: int = 20,
        output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate explanations for top K drugs
        
        Args:
            target_signature: Target signature
            score_result: Scoring results
            library: LINCS library DataFrame
            top_k: Number of top drugs to explain
            output_dir: Output directory for plots
        
        Returns:
            DataFrame with explanations added
        """
        top_drugs = score_result.ranking.head(top_k).copy()
        
        # Collect contributions for all drugs
        contributions_dict = {}
        enrichment_dict = {}
        
        for idx, row in top_drugs.iterrows():
            drug_name = row['compound']
            
            # Get drug signature from library
            drug_data = library[library['compound'] == drug_name]
            if drug_data.empty:
                continue
            
            drug_sig = dict(zip(drug_data['gene_symbol'], drug_data['score']))
            
            # Get metadata
            metadata = {
                'compound': drug_name,
                'rank': idx + 1,
                'score': row['score'],
                'p_value': row.get('p_value', 1.0),
                'moa': row.get('moa'),
                'target': row.get('target'),
                'cell_line': row.get('cell_line')
            }
            
            # Generate explanation
            explanation = self.explain_ranking(
                target_signature,
                drug_sig,
                metadata,
                output_dir=output_dir,
                create_plots=(output_dir is not None)
            )
            
            contributions_dict[drug_name] = explanation['contributions']
            if explanation['enrichment']:
                enrichment_dict[drug_name] = explanation['enrichment'].get('GO_BP')
        
        # Generate batch narratives
        top_drugs_explained = generate_batch_narratives(
            top_drugs,
            contributions_dict,
            enrichment_dict,
            top_n=top_k
        )
        
        return top_drugs_explained
    
    def compare_drugs(
        self,
        target_signature: TargetSignature,
        drug_a_signature: Dict[str, float],
        drug_b_signature: Dict[str, float],
        drug_a_metadata: Dict,
        drug_b_metadata: Dict,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Compare two drugs to explain ranking difference
        
        Args:
            target_signature: Target signature
            drug_a_signature: Drug A signature
            drug_b_signature: Drug B signature
            drug_a_metadata: Drug A metadata
            drug_b_metadata: Drug B metadata
            output_dir: Output directory
        
        Returns:
            Dictionary with comparison results
        """
        # Get common genes
        common_genes = sorted(
            set(target_signature.genes) & 
            set(drug_a_signature.keys()) & 
            set(drug_b_signature.keys())
        )
        
        target_array = np.array([
            target_signature.weights[target_signature.genes.index(g)]
            for g in common_genes
        ])
        drug_a_array = np.array([drug_a_signature[g] for g in common_genes])
        drug_b_array = np.array([drug_b_signature[g] for g in common_genes])
        
        # Compute contributions
        contrib_a = self.analyzer.compute_contributions(
            target_array, drug_a_array, common_genes
        )
        contrib_b = self.analyzer.compute_contributions(
            target_array, drug_b_array, common_genes
        )
        
        # Create comparison plot
        fig, comparison_df = compare_drug_contributions(
            target_array,
            drug_a_array,
            drug_b_array,
            common_genes,
            drug_a_metadata['compound'],
            drug_b_metadata['compound'],
            top_n=15
        )
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            comparison_filename = (
                f"comparison_{drug_a_metadata['compound']}_vs_"
                f"{drug_b_metadata['compound']}.png"
            )
            fig.savefig(
                output_path / comparison_filename,
                dpi=300,
                bbox_inches='tight',
            )
        
        # Generate narrative
        narrative = create_comparison_narrative(
            drug_a_metadata['compound'],
            drug_b_metadata['compound'],
            drug_a_metadata.get('rank', 1),
            drug_b_metadata.get('rank', 2),
            drug_a_metadata['score'],
            drug_b_metadata['score'],
            contrib_a,
            contrib_b,
            comparison_df
        )
        
        return {
            'drug_a': drug_a_metadata['compound'],
            'drug_b': drug_b_metadata['compound'],
            'contributions_a': contrib_a,
            'contributions_b': contrib_b,
            'comparison': comparison_df,
            'narrative': narrative,
            'plot': fig
        }


# Convenience functions for command-line use
def explain_top_drugs(
    target_json: str,
    results_parquet: str,
    library_parquet: str,
    top_k: int = 20,
    output_dir: str = 'explanations',
    enable_pathway_enrichment: bool = True
) -> pd.DataFrame:
    """
    Command-line interface for explaining top drugs
    
    Args:
        target_json: Path to target signature JSON
        results_parquet: Path to scoring results parquet
        library_parquet: Path to LINCS library parquet
        top_k: Number of drugs to explain
        output_dir: Output directory
        enable_pathway_enrichment: Enable pathway enrichment
    
    Returns:
        DataFrame with explanations
    """
    # Load data
    target = TargetSignature.from_json(target_json)
    results = pd.read_parquet(results_parquet)
    library = pd.read_parquet(library_parquet)
    
    # Create score result object
    score_result = ScoreResult(
        method='baseline',
        ranking=results,
        metadata={}
    )
    
    # Run explainability
    engine = ExplainabilityEngine(enable_pathway_enrichment=enable_pathway_enrichment)
    explained = engine.explain_top_k_drugs(
        target,
        score_result,
        library,
        top_k=top_k,
        output_dir=output_dir
    )
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    explained.to_parquet(output_path / 'explained_rankings.parquet')
    explained.to_csv(output_path / 'explained_rankings.csv', index=False)
    
    # Save narratives as text file
    with open(output_path / 'narratives.txt', 'w') as f:
        for idx, row in explained.iterrows():
            f.write(f"\n{'='*80}\n")
            f.write(f"Rank #{idx+1}: {row['compound']}\n")
            f.write(f"{'='*80}\n\n")
            f.write(row['narrative'])
            f.write("\n\n")
    
    print(f"\nExplanations saved to: {output_dir}/")
    print("  - explained_rankings.parquet")
    print("  - explained_rankings.csv")
    print("  - narratives.txt")
    print("  - Individual plots for each drug")
    
    return explained
