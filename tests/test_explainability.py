"""
Tests for explainability framework
"""

import numpy as np
import pandas as pd
import pytest

from scperturb_cmap.explainability.feature_importance import (
    GeneContributionAnalyzer,
    compare_drug_contributions,
    create_waterfall_plot,
    explain_drug_ranking,
)
from scperturb_cmap.explainability.narrative_generator import DrugNarrativeGenerator
from scperturb_cmap.explainability.pathway_enrichment import PathwayEnricher
from scperturb_cmap.explainability.uncertainty import UncertaintyQuantifier
from scperturb_cmap.io.schemas import TargetSignature


class TestGeneContributions:
    """Test gene contribution analysis"""
    
    def test_compute_contributions(self):
        """Test basic contribution computation"""
        analyzer = GeneContributionAnalyzer()
        
        # Create simple test data
        target = np.array([1.0, -1.0, 2.0, -2.0])
        drug = np.array([-1.0, 1.0, -2.0, 2.0])  # Perfect inverse
        genes = ['GENE1', 'GENE2', 'GENE3', 'GENE4']
        
        contrib = analyzer.compute_contributions(target, drug, genes, target_score=2.5)
        
        # Check structure
        assert len(contrib) == 4
        assert 'gene' in contrib.columns
        assert 'contribution' in contrib.columns
        assert 'direction' in contrib.columns
        
        # Contributions should rescale to the provided target score
        assert contrib.attrs.get("score_estimate") == 2.5
        assert pytest.approx(contrib["contribution"].sum(), rel=1e-6) == 2.5
        assert set(contrib["direction"].unique()).issubset({"helps", "hurts"})
    
    def test_identify_key_genes(self):
        """Test key gene identification"""
        analyzer = GeneContributionAnalyzer()
        
        target = np.random.randn(100)
        drug = -target + np.random.randn(100) * 0.1  # Mostly inverse
        genes = [f'GENE{i}' for i in range(100)]
        
        contrib = analyzer.compute_contributions(target, drug, genes)
        positive, negative = analyzer.identify_key_genes(contrib, top_n=10)
        
        assert len(positive) <= 10
        assert len(negative) <= 10
    
    def test_waterfall_plot_creation(self):
        """Test waterfall plot generation"""
        analyzer = GeneContributionAnalyzer()
        
        target = np.random.randn(50)
        drug = -target
        genes = [f'GENE{i}' for i in range(50)]
        
        contrib = analyzer.compute_contributions(target, drug, genes)
        
        # Create plot without saving
        fig = create_waterfall_plot(contrib, 'TestDrug', top_n=20)
        
        assert fig is not None
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_drug_comparison(self):
        """Test drug A vs B comparison"""
        target = np.random.randn(50)
        drug_a = -target + np.random.randn(50) * 0.1
        drug_b = -target + np.random.randn(50) * 0.5  # More noise
        genes = [f'GENE{i}' for i in range(50)]
        
        fig, comparison = compare_drug_contributions(
            target, drug_a, drug_b, genes, 'DrugA', 'DrugB'
        )
        
        assert comparison is not None
        assert 'contribution_diff' in comparison.columns
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_contribution_direction_changes_on_gene_flip(self):
        """Top driver flips sign when the underlying gene is flipped."""
        analyzer = GeneContributionAnalyzer()
        genes = ["A", "B", "C"]
        target = TargetSignature(genes=genes, weights=[2.0, -1.0, 0.5], metadata={})
        inverted_drug = {"A": -2.0, "B": 1.0, "C": -0.5}
        concordant_drug = {"A": 2.0, "B": 1.0, "C": -0.5}

        contrib_inverted = analyzer.compute_contributions(target, inverted_drug, genes, method="cosine")
        contrib_concordant = analyzer.compute_contributions(target, concordant_drug, genes, method="cosine")

        a_inv = contrib_inverted.loc[contrib_inverted["gene"] == "A", "contribution"].iloc[0]
        a_conc = contrib_concordant.loc[contrib_concordant["gene"] == "A", "contribution"].iloc[0]

        assert a_inv * a_conc < 0  # Sign flips
        # Concordant version should lower the overall score
        assert contrib_concordant["contribution"].sum() < contrib_inverted["contribution"].sum()

    def test_metric_blend_respects_baseline_component(self, monkeypatch):
        """Metric explainability blends baseline + metric contributions."""
        analyzer = GeneContributionAnalyzer()
        genes = ["A", "B", "C"]
        target = TargetSignature(genes=genes, weights=[1.0, 0.5, -0.5], metadata={})
        drug = {"A": -1.0, "B": -0.5, "C": 0.5}

        def fake_metric(self, *args, **kwargs):
            aligned_genes = kwargs.get("aligned_genes") or args[4]
            display_genes = kwargs.get("display_genes") or args[5]
            contrib = np.full(len(aligned_genes), 0.5)
            return pd.DataFrame(
                {
                    "gene": display_genes,
                    "harmonized_gene": aligned_genes,
                    "target_weight": np.ones(len(aligned_genes)),
                    "drug_weight": -np.ones(len(aligned_genes)),
                    "contribution": contrib,
                    "abs_contribution": np.abs(contrib),
                    "direction": ["hurts"] * len(aligned_genes),
                    "rank": range(1, len(aligned_genes) + 1),
                }
            )

        monkeypatch.setattr(GeneContributionAnalyzer, "_metric_contributions", fake_metric)

        contrib = analyzer.compute_contributions(
            target,
            drug,
            genes,
            method="metric",
            blend_weight=0.2,
            target_score=1.0,
            model_path="dummy",
        )

        assert "baseline_component" in contrib.columns
        assert "metric_component" in contrib.columns
        assert pytest.approx(contrib["contribution"].sum(), rel=1e-6) == 1.0
        # Blending should change the final contributions relative to the metric-only path
        assert not np.allclose(contrib["contribution"], contrib["metric_component"])


class TestPathwayEnrichment:
    """Test pathway enrichment analysis"""
    
    def test_enricher_initialization(self):
        """Test pathway enricher initialization"""
        enricher = PathwayEnricher()
        
        assert enricher.organism == 'human'
        assert len(enricher.SUPPORTED_LIBRARIES) > 0
    
    @pytest.mark.skip(reason="Requires internet connection")
    def test_enrich_genes(self):
        """Test gene enrichment (requires internet)"""
        enricher = PathwayEnricher()
        
        # Example immune genes
        genes = ['CD8A', 'CD8B', 'CD4', 'IL2', 'IFNG', 'TNF', 'GZMB', 'PRF1']
        
        results = enricher.enrich_genes(
            genes,
            libraries=['GO_Biological_Process_2021'],
            p_threshold=0.05
        )
        
        # May or may not find enrichment with only 8 genes
        assert isinstance(results, pd.DataFrame)


class TestNarrativeGeneration:
    """Test automated narrative generation"""
    
    def test_generator_initialization(self):
        """Test narrative generator initialization"""
        generator = DrugNarrativeGenerator()
        
        assert generator.templates is not None
        assert 'intro' in generator.templates
    
    def test_generate_narrative(self):
        """Test narrative generation"""
        generator = DrugNarrativeGenerator()
        
        # Create mock data
        analyzer = GeneContributionAnalyzer()
        target = np.random.randn(50)
        drug = -target
        genes = [f'GENE{i}' for i in range(50)]
        
        contrib = analyzer.compute_contributions(target, drug, genes)
        
        narrative = generator.generate_narrative(
            drug_name='TestDrug',
            rank=1,
            score=-3.45,
            p_value=0.001,
            contributions=contrib,
            moa='Test inhibitor',
            targets='TARGET1, TARGET2'
        )
        
        assert isinstance(narrative, str)
        assert 'TestDrug' in narrative
        assert len(narrative) > 100


class TestUncertaintyQuantification:
    """Test uncertainty quantification"""
    
    def test_quantifier_initialization(self):
        """Test uncertainty quantifier initialization"""
        quantifier = UncertaintyQuantifier(n_bootstrap=100)
        
        assert quantifier.n_bootstrap == 100
        assert quantifier.confidence_level == 0.95
    
    def test_bootstrap_scoring(self):
        """Test bootstrap scoring"""
        quantifier = UncertaintyQuantifier(n_bootstrap=100)
        
        target = np.random.randn(50)
        drug = -target + np.random.randn(50) * 0.1
        
        def scoring_func(t, d):
            return -np.corrcoef(t, d)[0, 1]
        
        uncertainty = quantifier.bootstrap_scoring(
            target, drug, scoring_func, n_iterations=100
        )
        
        assert 'mean' in uncertainty
        assert 'std' in uncertainty
        assert 'ci_lower' in uncertainty
        assert 'ci_upper' in uncertainty
        assert uncertainty['ci_lower'] < uncertainty['ci_upper']
    
    def test_jackknife_variance(self):
        """Test jackknife variance estimation"""
        quantifier = UncertaintyQuantifier()
        
        target = np.random.randn(50)
        drug = -target
        
        def scoring_func(t, d):
            return -np.corrcoef(t, d)[0, 1]
        
        result = quantifier.jackknife_variance(target, drug, scoring_func)
        
        assert 'score' in result
        assert 'variance' in result
        assert 'std' in result
        assert result['variance'] >= 0


class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_complete_explanation_workflow(self):
        """Test complete explanation generation"""
        # Create mock data
        np.random.seed(42)
        n_genes = 100
        
        target_dict = {f'GENE{i}': np.random.randn() for i in range(n_genes)}
        drug_dict = {f'GENE{i}': -target_dict[f'GENE{i}'] + np.random.randn()*0.1 
                     for i in range(n_genes)}
        
        # Run explanation
        
        result = explain_drug_ranking(
            target_signature=target_dict,
            drug_signature=drug_dict,
            drug_name='TestDrug',
            top_n=20,
            create_plot=False
        )
        
        assert 'contributions' in result
        assert 'positive_drivers' in result
        assert 'summary_stats' in result
        assert result['drug_name'] == 'TestDrug'


def test_explainability_api():
    """Test high-level explainability API"""
    from scperturb_cmap.api.explain import ExplainabilityEngine
    from scperturb_cmap.io.schemas import TargetSignature
    
    # Create engine
    engine = ExplainabilityEngine(enable_pathway_enrichment=False)
    
    assert engine.analyzer is not None
    assert engine.generator is not None
    assert engine.quantifier is not None
    
    # Create mock data
    target = TargetSignature(
        genes=[f'GENE{i}' for i in range(50)],
        weights=list(np.random.randn(50)),
        metadata={}
    )
    
    drug_sig = {f'GENE{i}': -target.weights[i] + np.random.randn()*0.1 
                for i in range(50)}
    
    metadata = {
        'compound': 'TestDrug',
        'rank': 1,
        'score': -3.5,
        'moa': 'Test inhibitor'
    }
    
    # Generate explanation
    explanation = engine.explain_ranking(
        target_signature=target,
        drug_signature=drug_sig,
        drug_metadata=metadata,
        create_plots=False
    )
    
    assert explanation['drug_name'] == 'TestDrug'
    assert 'contributions' in explanation
    assert 'narrative' in explanation
    assert 'summary' in explanation


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
