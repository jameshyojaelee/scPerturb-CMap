"""
Automated narrative generation explaining drug rankings
Creates human-readable explanations citing specific gene inversion patterns
"""
from typing import Dict, List, Optional

import pandas as pd


class DrugNarrativeGenerator:
    """
    Generates automated narratives explaining why drugs rank highly
    """
    
    def __init__(self):
        """Initialize narrative generator"""
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load narrative templates"""
        return {
            'intro': (
                "{drug_name} ({moa}) ranks #{rank} with a connectivity score of "
                "{score:.3f} (p={p_value:.4f})."
            ),
            'mechanism': (
                "This {mechanism_class} acts on {targets}, which {biological_rationale}."
            ),
            'gene_inversion': (
                "{drug_name} demonstrates strong inversion of {n_genes} key disease genes. "
                "Notably, {top_genes_narrative}."
            ),
            'pathway': (
                "Pathway enrichment analysis reveals {n_pathways} significantly affected "
                "pathways (FDR < 0.05), including {top_pathways}."
            ),
            'cell_line': (
                "These effects were observed in {cell_line} cells, a {cell_line_description} model."
            ),
            'validation': "Literature support: {validation_narrative}",
            'conclusion': (
                "In summary, {drug_name} is predicted to reverse the disease signature through "
                "{n_mechanisms} complementary mechanisms, with {confidence_level} confidence."
            ),
        }
    
    def generate_narrative(
        self,
        drug_name: str,
        rank: int,
        score: float,
        p_value: float,
        contributions: pd.DataFrame,
        enrichment_results: Optional[pd.DataFrame] = None,
        moa: Optional[str] = None,
        targets: Optional[str] = None,
        cell_line: Optional[str] = None,
        literature_refs: Optional[List[str]] = None
    ) -> str:
        """
        Generate complete narrative explaining drug ranking
        
        Args:
            drug_name: Drug name
            rank: Ranking position
            score: Connectivity score
            p_value: Statistical significance
            contributions: Gene contribution DataFrame
            enrichment_results: Pathway enrichment results
            moa: Mechanism of action
            targets: Drug targets
            cell_line: Cell line used
            literature_refs: Literature references
        
        Returns:
            Human-readable narrative
        """
        narrative_parts = []
        
        # Introduction
        intro = self.templates['intro'].format(
            drug_name=drug_name,
            moa=moa or "investigational compound",
            rank=rank,
            score=score,
            p_value=p_value
        )
        narrative_parts.append(intro)
        
        # Mechanism section
        if moa and targets:
            mechanism_narrative = self._generate_mechanism_narrative(
                drug_name, moa, targets
            )
            narrative_parts.append(mechanism_narrative)
        
        # Gene inversion section
        gene_narrative = self._generate_gene_inversion_narrative(
            drug_name, contributions
        )
        narrative_parts.append(gene_narrative)
        
        # Pathway section
        if enrichment_results is not None and not enrichment_results.empty:
            pathway_narrative = self._generate_pathway_narrative(
                enrichment_results
            )
            narrative_parts.append(pathway_narrative)
        
        # Cell line context
        if cell_line:
            cell_line_narrative = self._generate_cell_line_narrative(cell_line)
            narrative_parts.append(cell_line_narrative)
        
        # Validation/literature support
        if literature_refs:
            validation_narrative = self._generate_validation_narrative(
                literature_refs
            )
            narrative_parts.append(validation_narrative)
        
        # Conclusion
        conclusion = self._generate_conclusion(
            drug_name, contributions, enrichment_results
        )
        narrative_parts.append(conclusion)
        
        # Combine all parts
        full_narrative = " ".join(narrative_parts)
        
        return full_narrative
    
    def _generate_mechanism_narrative(
        self,
        drug_name: str,
        moa: str,
        targets: str
    ) -> str:
        """Generate mechanism of action narrative"""
        # Parse MOA to determine biological rationale
        moa_lower = moa.lower()
        
        rationale_map = {
            'kinase inhibitor': 'blocks aberrant signaling cascades driving disease progression',
            'hdac inhibitor': (
                'modulates chromatin accessibility to restore healthy gene expression patterns'
            ),
            'proteasome inhibitor': 'prevents degradation of key regulatory proteins',
            'receptor antagonist': 'blocks pathological receptor activation',
            'enzyme inhibitor': 'disrupts metabolic pathways sustaining disease phenotype',
        }
        
        biological_rationale = None
        for key, rationale in rationale_map.items():
            if key in moa_lower:
                biological_rationale = rationale
                break
        
        if biological_rationale is None:
            biological_rationale = "modulates disease-relevant molecular pathways"
        
        return self.templates['mechanism'].format(
            mechanism_class=moa,
            targets=targets,
            biological_rationale=biological_rationale
        )
    
    def _generate_gene_inversion_narrative(
        self,
        drug_name: str,
        contributions: pd.DataFrame
    ) -> str:
        """Generate narrative about gene-level inversions"""
        # Get top contributing genes
        top_genes = contributions.head(10)
        
        # Categorize by contribution direction
        beneficial = top_genes[top_genes['contribution'] > 0]
        
        # Create gene list narrative
        gene_narratives = []
        for _, row in beneficial.head(5).iterrows():
            gene = row['gene']
            contrib = row['contribution']
            
            # Describe the inversion
            if contrib > 0.1:
                strength = "strongly inverts"
            elif contrib > 0.05:
                strength = "moderately inverts"
            else:
                strength = "inverts"
            
            gene_narratives.append(f"{gene} ({strength}, Δ={contrib:.3f})")
        
        top_genes_text = ", ".join(gene_narratives)
        
        return self.templates['gene_inversion'].format(
            drug_name=drug_name,
            n_genes=len(beneficial),
            top_genes_narrative=top_genes_text
        )
    
    def _generate_pathway_narrative(
        self,
        enrichment_results: pd.DataFrame
    ) -> str:
        """Generate pathway enrichment narrative"""
        top_pathways = enrichment_results.head(5)
        
        # Extract pathway names and clean them
        pathway_names = []
        for pathway in top_pathways['pathway']:
            # Simplify long pathway names
            if len(pathway) > 60:
                pathway = pathway[:57] + "..."
            pathway_names.append(pathway)
        
        pathways_text = "; ".join(pathway_names)
        
        return self.templates['pathway'].format(
            n_pathways=len(enrichment_results),
            top_pathways=pathways_text
        )
    
    def _generate_cell_line_narrative(self, cell_line: str) -> str:
        """Generate cell line context narrative"""
        # Cell line descriptions
        descriptions = {
            'MCF7': 'luminal breast cancer',
            'A549': 'lung adenocarcinoma',
            'HT29': 'colorectal carcinoma',
            'PC3': 'prostate carcinoma',
            'HEPG2': 'hepatocellular carcinoma',
            'HL-60': 'promyelocytic leukemia',
            'A375': 'malignant melanoma',
            'MDAMB231': 'triple-negative breast cancer',
            'BT549': 'triple-negative breast cancer'
        }
        
        description = descriptions.get(cell_line, "disease-relevant")
        
        return self.templates['cell_line'].format(
            cell_line=cell_line,
            cell_line_description=description
        )
    
    def _generate_validation_narrative(
        self,
        literature_refs: List[str]
    ) -> str:
        """Generate literature validation narrative"""
        if len(literature_refs) >= 3:
            support_level = "strong preclinical and clinical evidence"
        elif len(literature_refs) >= 2:
            support_level = "moderate experimental evidence"
        else:
            support_level = "preliminary evidence"
        
        refs_text = "; ".join(literature_refs[:3])  # Show first 3 references
        
        return self.templates['validation'].format(
            validation_narrative=f"{support_level} ({refs_text})"
        )
    
    def _generate_conclusion(
        self,
        drug_name: str,
        contributions: pd.DataFrame,
        enrichment_results: Optional[pd.DataFrame]
    ) -> str:
        """Generate conclusion summary"""
        # Count mechanisms (gene-level + pathway-level)
        n_gene_mechanisms = len(contributions[contributions['contribution'] > 0.05])
        n_pathway_mechanisms = 0
        if enrichment_results is not None and not enrichment_results.empty:
            n_pathway_mechanisms = len(enrichment_results[enrichment_results['q_value'] < 0.05])
        
        total_mechanisms = min(n_gene_mechanisms + n_pathway_mechanisms, 10)
        
        # Assess confidence
        top_contrib = contributions.iloc[0]['contribution'] if not contributions.empty else 0
        if top_contrib > 0.2:
            confidence = "high"
        elif top_contrib > 0.1:
            confidence = "moderate"
        else:
            confidence = "preliminary"
        
        return self.templates['conclusion'].format(
            drug_name=drug_name,
            n_mechanisms=total_mechanisms,
            confidence_level=confidence
        )


def generate_drug_narrative(
    drug_name: str,
    rank: int,
    score: float,
    p_value: float,
    contributions: pd.DataFrame,
    enrichment: Optional[pd.DataFrame] = None,
    metadata: Optional[Dict] = None,
) -> str:
    """Backward-compatible wrapper for narrative generation."""

    return explain_ranking(
        drug_name=drug_name,
        rank=rank,
        score=score,
        p_value=p_value,
        contributions=contributions,
        enrichment=enrichment,
        metadata=metadata,
    )


def explain_ranking(
    drug_name: str,
    rank: int,
    score: float,
    p_value: float,
    contributions: pd.DataFrame,
    enrichment: Optional[pd.DataFrame] = None,
    metadata: Optional[Dict] = None
) -> str:
    """
    Generate explanation for a single drug's ranking
    
    Args:
        drug_name: Drug name
        rank: Ranking position
        score: Connectivity score
        p_value: P-value
        contributions: Gene contributions
        enrichment: Pathway enrichment results
        metadata: Additional metadata (MOA, targets, etc.)
    
    Returns:
        Explanation narrative
    """
    generator = DrugNarrativeGenerator()
    
    # Extract metadata if provided
    moa = metadata.get('moa') if metadata else None
    targets = metadata.get('target') if metadata else None
    cell_line = metadata.get('cell_line') if metadata else None
    literature = metadata.get('literature') if metadata else None
    
    narrative = generator.generate_narrative(
        drug_name=drug_name,
        rank=rank,
        score=score,
        p_value=p_value,
        contributions=contributions,
        enrichment_results=enrichment,
        moa=moa,
        targets=targets,
        cell_line=cell_line,
        literature_refs=literature
    )
    
    return narrative


def create_comparison_narrative(
    drug_a_name: str,
    drug_b_name: str,
    rank_a: int,
    rank_b: int,
    score_a: float,
    score_b: float,
    contributions_a: pd.DataFrame,
    contributions_b: pd.DataFrame,
    comparison_df: pd.DataFrame
) -> str:
    """
    Generate narrative explaining why Drug A ranks higher than Drug B
    
    Args:
        drug_a_name: Drug A name
        drug_b_name: Drug B name
        rank_a: Drug A rank
        rank_b: Drug B rank
        score_a: Drug A score
        score_b: Drug B score
        contributions_a: Drug A gene contributions
        contributions_b: Drug B gene contributions
        comparison_df: DataFrame with contribution differences
    
    Returns:
        Comparison narrative
    """
    # Calculate key differences
    score_diff = score_a - score_b
    rank_diff = rank_b - rank_a  # A should rank higher (lower number)
    
    # Find key differentiating genes
    top_diff = comparison_df.nlargest(5, 'abs_diff')
    
    # Build narrative
    intro = (
        f"{drug_a_name} ranks #{rank_a} (score={score_a:.3f}), "
        f"{rank_diff} positions higher than {drug_b_name} (rank #{rank_b}, score={score_b:.3f}). "
        f"The connectivity score difference of {score_diff:.3f} is primarily driven by "
        f"differential effects on {len(top_diff)} key genes."
    )
    
    # Analyze top differentiating genes
    gene_explanations = []
    for _, row in top_diff.iterrows():
        gene = row['gene']
        diff = row['contribution_diff']
        
        if diff > 0:
            gene_explanations.append(
                f"{gene} ({drug_a_name} inverts {abs(diff):.3f} more effectively)"
            )
        else:
            gene_explanations.append(
                f"{gene} ({drug_b_name} inverts {abs(diff):.3f} more effectively)"
            )
    
    gene_section = (
        f"Key differentiating genes include: {'; '.join(gene_explanations)}."
    )
    
    # Overall assessment
    if score_diff > 0.5:
        assessment = f"{drug_a_name} demonstrates substantially stronger inversion"
    elif score_diff > 0.2:
        assessment = f"{drug_a_name} shows moderately stronger inversion"
    else:
        assessment = f"{drug_a_name} exhibits slightly stronger inversion"
    
    conclusion = (
        f"In summary, {assessment} of the target signature compared to {drug_b_name}, "
        f"primarily through more effective modulation of disease-critical genes."
    )
    
    return f"{intro} {gene_section} {conclusion}"


def generate_batch_narratives(
    results_df: pd.DataFrame,
    contributions_dict: Dict[str, pd.DataFrame],
    enrichment_dict: Optional[Dict[str, pd.DataFrame]] = None,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Generate narratives for top N drugs in batch
    
    Args:
        results_df: Results DataFrame with rankings
        contributions_dict: Dictionary mapping drug_name -> contributions DataFrame
        enrichment_dict: Dictionary mapping drug_name -> enrichment DataFrame
        top_n: Number of top drugs to explain
    
    Returns:
        DataFrame with narratives added
    """
    generator = DrugNarrativeGenerator()
    
    # Select top N
    top_drugs = results_df.head(top_n).copy()
    
    narratives = []
    for idx, row in top_drugs.iterrows():
        drug_name = row['compound']
        
        # Get contributions
        if drug_name not in contributions_dict:
            narratives.append("Explanation not available")
            continue
        
        contrib = contributions_dict[drug_name]
        
        # Get enrichment if available
        enrich = None
        if enrichment_dict and drug_name in enrichment_dict:
            enrich = enrichment_dict[drug_name]
        
        # Generate narrative
        metadata = {
            'moa': row.get('moa'),
            'target': row.get('target'),
            'cell_line': row.get('cell_line')
        }
        
        narrative = generator.generate_narrative(
            drug_name=drug_name,
            rank=idx + 1,
            score=row['score'],
            p_value=row.get('p_value', 0.05),
            contributions=contrib,
            enrichment_results=enrich,
            moa=metadata['moa'],
            targets=metadata['target'],
            cell_line=metadata['cell_line']
        )
        
        narratives.append(narrative)
    
    top_drugs['narrative'] = narratives
    
    return top_drugs
