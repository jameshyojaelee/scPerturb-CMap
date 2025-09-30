# Explainability Framework

The scPerturb-CMap explainability framework provides SHAP-like interpretability for drug rankings, explaining **why** each compound is predicted to reverse your disease signature.

## Overview

Traditional connectivity mapping produces ranked lists of compounds but leaves the **mechanistic rationale** unclear. Our explainability framework addresses this by providing:

1. **Gene-Level Attribution**: Which specific genes drive each drug's ranking
2. **Pathway Context**: What biological processes are affected
3. **Automated Narratives**: Human-readable explanations citing evidence
4. **Uncertainty Quantification**: Confidence intervals and prediction reliability
5. **Comparative Analysis**: Why Drug A ranks higher than Drug B

---

## Features

### 1. SHAP-Like Gene Contributions

**What it does:** Decomposes the overall connectivity score into individual gene contributions, similar to SHAP (SHapley Additive exPlanations) values.

**Output:**
- Each gene gets a contribution score representing its impact on the ranking
- Contributions sum to approximate the total connectivity score
- Positive contributions = beneficial gene inversion
- Negative contributions = poor alignment or off-target effects

**Example:**
```python
from scperturb_cmap.explainability.feature_importance import GeneContributionAnalyzer

analyzer = GeneContributionAnalyzer()
contributions = analyzer.compute_contributions(
    target_signature=target_weights,
    drug_signature=drug_weights,
    gene_names=gene_list
)

print(contributions.head(10))
```

**Output:**
```
        gene  target_weight  drug_weight  contribution  direction
0        TOX           4.12       -3.87         0.425  beneficial
1      PDCD1           3.87       -3.21         0.389  beneficial
2      IFNG          -4.52        4.21         0.376  beneficial
3       GZMB          -3.98        3.67         0.342  beneficial
4       VIM            5.21       -2.43         0.298  beneficial
...
```

---

### 2. Waterfall Plots

**What it does:** Visualizes gene-level contributions in an intuitive waterfall format, showing how individual genes contribute to the overall score.

**Example:**
```python
from scperturb_cmap.explainability.feature_importance import create_waterfall_plot

fig = create_waterfall_plot(
    contributions=contributions,
    drug_name='JQ1',
    top_n=20,
    output_path='waterfall_JQ1.png'
)
```

**Interpretation:**
- **Blue bars**: Genes with beneficial contributions (good alignment)
- **Red bars**: Genes with detrimental contributions (poor alignment)
- **Bar length**: Magnitude of contribution
- **Total shown**: Sum of contributions from displayed genes

![Example Waterfall Plot](../figs/waterfall_example.png)

---

### 3. Pathway Enrichment Analysis

**What it does:** Identifies enriched biological pathways among top-contributing genes, providing biological context using GO, KEGG, and Reactome databases.

**Example:**
```python
from scperturb_cmap.explainability.pathway_enrichment import integrate_go_kegg_reactome

# Get top beneficial genes
top_genes = contributions[contributions['contribution'] > 0].head(50)['gene'].tolist()

# Run enrichment
enrichment = integrate_go_kegg_reactome(
    gene_list=top_genes,
    top_n_pathways=20,
    p_threshold=0.05
)

# Results for each database
print(enrichment['GO_BP'].head())  # GO Biological Process
print(enrichment['KEGG'].head())    # KEGG pathways
print(enrichment['Reactome'].head())  # Reactome pathways
```

**Output:**
```
                                          pathway  q_value     overlap
0   T cell receptor signaling pathway (GO:0050852)  0.0012  15/287
1   Regulation of immune response (GO:0050776)      0.0023  18/412
2   Interferon-gamma-mediated signaling (GO:0060333) 0.0034  12/95
...
```

**Visualization:**
```python
from scperturb_cmap.explainability.pathway_enrichment import (
    create_enrichment_barplot,
    visualize_pathway_network
)

# Barplot of top pathways
fig = create_enrichment_barplot(enrichment['GO_BP'], top_n=15)

# Network showing pathway relationships
fig = visualize_pathway_network(enrichment['GO_BP'], top_n=20)
```

---

### 4. Automated Narrative Generation

**What it does:** Generates human-readable explanations for drug rankings, citing specific gene inversions and pathway alterations.

**Example:**
```python
from scperturb_cmap.explainability.narrative_generator import explain_ranking

narrative = explain_ranking(
    drug_name='JQ1',
    rank=1,
    score=-3.45,
    p_value=0.0023,
    contributions=contributions,
    enrichment=enrichment['GO_BP'],
    metadata={
        'moa': 'BET bromodomain inhibitor',
        'target': 'BRD2, BRD3, BRD4',
        'cell_line': 'A549'
    }
)

print(narrative)
```

**Output:**
```
JQ1 (BET bromodomain inhibitor) ranks #1 with a connectivity score of -3.450 (p=0.0023). 
This BET bromodomain inhibitor acts on BRD2, BRD3, BRD4, which blocks aberrant signaling 
cascades driving disease progression. JQ1 demonstrates strong inversion of 47 key disease 
genes. Notably, TOX (strongly inverts, Δ=0.425), PDCD1 (strongly inverts, Δ=0.389), 
IFNG (strongly inverts, Δ=0.376), GZMB (strongly inverts, Δ=0.342), VIM (moderately 
inverts, Δ=0.298). Pathway enrichment analysis reveals 12 significantly affected pathways 
(FDR < 0.05), including T cell receptor signaling pathway (GO:0050852); Regulation of 
immune response (GO:0050776); Interferon-gamma-mediated signaling (GO:0060333). 
These effects were observed in A549 cells, a lung adenocarcinoma model. In summary, 
JQ1 is predicted to reverse the disease signature through 8 complementary mechanisms, 
with high confidence.
```

---

### 5. Cell-Line-Specific Predictions with Confidence Intervals

**What it does:** Provides predictions for each cell line separately with bootstrap-derived confidence intervals, enabling assessment of prediction robustness.

**Example:**
```python
from scperturb_cmap.explainability.uncertainty import cell_line_specific_predictions

# Organize drug signatures by cell line
drug_sigs_by_cell = {
    'MCF7': {drug_name: drug_signature_dict},
    'A549': {drug_name: drug_signature_dict},
    'HT29': {drug_name: drug_signature_dict}
}

# Scoring function
def scoring_func(target, drug):
    return -np.corrcoef(target, drug)[0, 1]

# Get cell-line-specific predictions
predictions = cell_line_specific_predictions(
    target_signature=target_dict,
    drug_signatures_by_cell_line=drug_sigs_by_cell,
    scoring_func=scoring_func,
    compute_uncertainty=True
)

print(predictions.head(10))
```

**Output:**
```
  cell_line    drug     score  score_mean  score_std  ci_lower  ci_upper  ci_width      cv
0      MCF7     JQ1    -3.456      -3.432      0.187    -3.798    -3.065     0.733   0.054
1      A549     JQ1    -3.298      -3.281      0.234    -3.742    -2.821     0.921   0.071
2      HT29     JQ1    -2.987      -2.965      0.298    -3.551    -2.379     1.172   0.101
...
```

**Reliability Assessment:**
```python
from scperturb_cmap.explainability.uncertainty import compute_prediction_reliability

reliability = compute_prediction_reliability(predictions)
print(reliability.head())
```

**Output:**
```
     drug  n_cell_lines  mean_score  std_score     cv  confidence_level
0     JQ1             3      -3.247      0.195  0.060              high
1  SAHA              3      -2.987      0.421  0.141          moderate
2  Dasatinib         2      -2.743      0.587  0.214               low
...
```

**Visualization:**
```python
from scperturb_cmap.explainability.uncertainty import create_uncertainty_plot

fig = create_uncertainty_plot(predictions, top_n=15)
```

---

### 6. Drug A vs Drug B Comparison Mode

**What it does:** Explains why Drug A ranks higher than Drug B by identifying key differentiating genes.

**Example:**
```python
from scperturb_cmap.explainability.feature_importance import compare_drug_contributions

fig, comparison = compare_drug_contributions(
    target_signature=target_array,
    drug_a_signature=drug_a_array,
    drug_b_signature=drug_b_array,
    gene_names=gene_list,
    drug_a_name='JQ1',
    drug_b_name='Vorinostat',
    top_n=15
)
```

**Output Plot:** Side-by-side comparison showing:
1. Drug A gene contributions
2. Drug B gene contributions
3. Difference (A - B) highlighting key differentiators

**Narrative:**
```python
from scperturb_cmap.explainability.narrative_generator import create_comparison_narrative

narrative = create_comparison_narrative(
    drug_a_name='JQ1',
    drug_b_name='Vorinostat',
    rank_a=1,
    rank_b=4,
    score_a=-3.45,
    score_b=-2.87,
    contributions_a=contrib_a,
    contributions_b=contrib_b,
    comparison_df=comparison
)

print(narrative)
```

**Output:**
```
JQ1 ranks #1 (score=-3.450), 3 positions higher than Vorinostat (rank #4, score=-2.870). 
The connectivity score difference of 0.580 is primarily driven by differential effects on 
5 key genes. Key differentiating genes include: TOX (JQ1 inverts 0.312 more effectively); 
PDCD1 (JQ1 inverts 0.287 more effectively); BRD4 (JQ1 inverts 0.245 more effectively); 
SNAI1 (Vorinostat inverts 0.156 more effectively); CDH1 (Vorinostat inverts 0.134 more 
effectively). In summary, JQ1 exhibits moderately stronger inversion of the target signature 
compared to Vorinostat, primarily through more effective modulation of disease-critical genes.
```

---

## Complete Workflow Example

### Using the High-Level API

```python
from scperturb_cmap.api.explain import ExplainabilityEngine
from scperturb_cmap.io.schemas import TargetSignature
import pandas as pd

# Load data
target = TargetSignature.from_json('target.json')
results = pd.read_parquet('results.parquet')
library = pd.read_parquet('lincs_library.parquet')

# Initialize explainability engine
engine = ExplainabilityEngine(enable_pathway_enrichment=True)

# Explain top 20 drugs
from scperturb_cmap.io.schemas import ScoreResult
score_result = ScoreResult(method='baseline', ranking=results, metadata={})

explained = engine.explain_top_k_drugs(
    target_signature=target,
    score_result=score_result,
    library=library,
    top_k=20,
    output_dir='explanations'
)

# View results
print(explained[['compound', 'score', 'narrative']].head())

# Compare two drugs
drug_a_sig = dict(zip(
    library[library['compound'] == 'JQ1']['gene_symbol'],
    library[library['compound'] == 'JQ1']['score']
))
drug_b_sig = dict(zip(
    library[library['compound'] == 'Vorinostat']['gene_symbol'],
    library[library['compound'] == 'Vorinostat']['score']
))

comparison = engine.compare_drugs(
    target_signature=target,
    drug_a_signature=drug_a_sig,
    drug_b_signature=drug_b_sig,
    drug_a_metadata={'compound': 'JQ1', 'rank': 1, 'score': -3.45},
    drug_b_metadata={'compound': 'Vorinostat', 'rank': 4, 'score': -2.87},
    output_dir='explanations'
)

print(comparison['narrative'])
```

---

## Command-Line Interface

```bash
# Explain top 20 drugs with all features
python -m scperturb_cmap.api.explain \
  --target target.json \
  --results results.parquet \
  --library lincs_library.parquet \
  --top-k 20 \
  --output-dir explanations \
  --enable-pathway-enrichment

# Outputs:
#   explanations/explained_rankings.parquet
#   explanations/explained_rankings.csv
#   explanations/narratives.txt
#   explanations/{drug_name}_waterfall.png (for each drug)
#   explanations/{drug_name}_enrichment_{database}.png
```

---

## Integration with Scoring Pipeline

### Add Explainability to Existing Workflow

```python
from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.api.explain import ExplainabilityEngine

# Step 1: Score compounds (as usual)
target = TargetSignature.from_json('target.json')
library = pd.read_parquet('lincs_library.parquet')

result = rank_drugs(
    target_signature=target,
    library=library,
    method='baseline',
    top_k=100
)

# Step 2: Explain top hits
engine = ExplainabilityEngine()
explained = engine.explain_top_k_drugs(
    target,
    result,
    library,
    top_k=20,
    output_dir='explanations'
)

# Step 3: Filter by confidence
high_confidence = explained[explained['narrative'].str.contains('high confidence')]
print(f"High-confidence predictions: {len(high_confidence)}")
```

---

## Performance Considerations

### Bootstrap Sampling
- **Default iterations**: 1000 (for publication-quality CIs)
- **Fast mode**: 100-500 iterations (for exploratory analysis)
- **Very large gene sets**: Use jackknife instead (faster, less accurate)

```python
# Fast uncertainty quantification
quantifier = UncertaintyQuantifier(n_bootstrap=100, confidence_level=0.95)

# Or use jackknife (faster)
from scperturb_cmap.explainability.uncertainty import UncertaintyQuantifier
quantifier = UncertaintyQuantifier()
uncertainty = quantifier.jackknife_variance(target, drug, scoring_func)
```

### Pathway Enrichment
- **Internet required**: Enrichr API calls (can fail if offline)
- **Caching**: Results are cached per gene list (speeds up repeated queries)
- **Disable if offline**: `enable_pathway_enrichment=False`

---

## Interpreting Outputs

### Gene Contributions
- **Positive contribution**: Gene is inverted in beneficial direction
  - Target ↑, Drug ↓ → reversal (good)
  - Target ↓, Drug ↑ → reversal (good)
- **Negative contribution**: Gene is inverted in detrimental direction or not inverted
  - Target ↑, Drug ↑ → amplification (bad for reversal)
  - Target ↓, Drug ↓ → amplification (bad for reversal)

### Confidence Levels
- **High**: CV < 0.5, ≥2 cell lines, consistent across replicates
- **Moderate**: CV 0.5-0.75, ≥2 cell lines
- **Low**: CV > 0.75 or only 1 cell line

### P-values and Q-values
- **Connectivity p-value**: Statistical significance of connectivity score
- **Pathway q-value**: FDR-corrected enrichment p-value
- **Threshold**: Typically q < 0.05 for significance

---

## Best Practices

1. **Always check gene overlap**: Low overlap (<100 genes) reduces confidence
2. **Use multiple cell lines**: Increases prediction robustness
3. **Validate top genes**: Confirm biological relevance of key contributors
4. **Cross-reference pathways**: Look for concordance across GO/KEGG/Reactome
5. **Interpret narratives carefully**: Auto-generated text may require domain expertise
6. **Use comparison mode**: When choosing between similar compounds

---

## Troubleshooting

### Issue: Pathway enrichment fails
**Solution**: Check internet connection, reduce gene list size, or disable enrichment

### Issue: Confidence intervals too wide
**Solution**: Increase cell lines, increase bootstrap iterations, or check for gene overlap

### Issue: Narratives seem generic
**Solution**: Add more metadata (MOA, targets, literature) to `drug_metadata` dict

### Issue: Contributions don't sum to score
**Solution**: This is expected; contributions approximate score but may differ due to normalization

---

## Citation

If you use the explainability framework in publications, please cite:

```bibtex
@software{scperturb_cmap_explainability,
  title={scPerturb-CMap Explainability Framework},
  author={scPerturb-CMap Development Team},
  year={2025},
  url={https://github.com/jameslee/scPerturb-CMap}
}
```

---

## Additional Resources

- [Explainability Demo](../examples/explainability_demo.py)
- [API Documentation](api.md#explainability)
- [Case Studies with Explanations](../case_studies/)
- [SHAP Original Paper](https://arxiv.org/abs/1705.07874)

---

**Questions?** Open an issue on GitHub or contact support@scperturb-cmap.org
