"""
Explainability and interpretability module for scPerturb-CMap
Provides SHAP-like feature importance, gene-level contributions, and automated narratives
"""

from .feature_importance import (
    compare_drug_contributions,
    compute_gene_contributions,
    create_waterfall_plot,
    rank_gene_importance,
)
from .narrative_generator import (
    create_comparison_narrative,
    explain_ranking,
    generate_drug_narrative,
)
from .pathway_enrichment import (
    enrich_pathways,
    integrate_go_kegg_reactome,
    visualize_pathway_network,
)
from .uncertainty import (
    bootstrap_scoring,
    cell_line_specific_predictions,
    compute_confidence_intervals,
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
