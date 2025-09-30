"""
Explainability and interpretability module for scPerturb-CMap
Provides SHAP-like feature importance, gene-level contributions, and automated narratives
"""

from .feature_importance import (
    compute_gene_contributions,
    rank_gene_importance,
    create_waterfall_plot,
    compare_drug_contributions,
)

from .pathway_enrichment import (
    enrich_pathways,
    integrate_go_kegg_reactome,
    visualize_pathway_network,
)

from .narrative_generator import (
    generate_drug_narrative,
    explain_ranking,
    create_comparison_narrative,
)

from .uncertainty import (
    compute_confidence_intervals,
    cell_line_specific_predictions,
    bootstrap_scoring,
)

__all__ = [
    'compute_gene_contributions',
    'rank_gene_importance',
    'create_waterfall_plot',
    'compare_drug_contributions',
    'enrich_pathways',
    'integrate_go_kegg_reactome',
    'visualize_pathway_network',
    'generate_drug_narrative',
    'explain_ranking',
    'create_comparison_narrative',
    'compute_confidence_intervals',
    'cell_line_specific_predictions',
    'bootstrap_scoring',
]
