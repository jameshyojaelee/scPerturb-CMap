# scPerturb-CMap Case Studies

This directory contains comprehensive, real-world case studies demonstrating the full scPerturb-CMap workflow from single-cell data to validated drug predictions.

## Case Studies

### 1. NSCLC CD8+ T Cell Exhaustion

**Directory**: `nsclc_cd8/`

**Disease Context**: Non-small cell lung cancer with exhausted tumor-infiltrating T cells

**Objective**: Reverse CD8+ T cell exhaustion to restore anti-tumor immunity

**Key Findings**:
- **Top Drug**: JQ1 (BET bromodomain inhibitor)
- **Top 20 Compounds**: With literature citations and validation status
- **Key Genes**: TOX, PDCD1, HAVCR2, LAG3 (canonical exhaustion markers)
- **Pathways**: T cell receptor signaling, chromatin remodeling
- **Validation**: Strong literature support (3+ papers)

**Files**:
- `CASE_STUDY_NSCLC_CD8.md` - Complete analysis and validation plan
- `data/` - Example datasets
- `scripts/` - Analysis scripts
- `results/` - Output rankings and QC
- `figures/` - Plots and visualizations

---

### 2. EMT in Triple-Negative Breast Cancer

**Directory**: `emt_breast/`

**Disease Context**: Epithelial-mesenchymal transition driving metastasis and therapy resistance

**Objective**: Reverse EMT to reduce metastatic potential and re-sensitize to chemotherapy

**Key Findings**:
- **Top Drug**: Galunisertib (TGF-β receptor inhibitor)
- **Top 15 Compounds**: With MOA pathway analysis
- **Key Genes**: VIM, FN1, SNAI2, ZEB1, CDH1 (EMT hallmarks)
- **Pathways**: TGF-β signaling (4.2x enriched), HDAC-mediated epigenetic regulation
- **Clinical Translation**: Phase II trial design included

**Files**:
- `CASE_STUDY_EMT_BREAST.md` - Complete analysis with MOA enrichment
- `data/` - TNBC single-cell data
- `scripts/` - Target construction and scoring
- `results/` - Drug rankings with pathway networks
- `figures/` - MOA enrichment and network plots

---

### 3. IFN-High Macrophages in Inflammatory Disease

**Directory**: `ifn_macrophages/`

**Disease Context**: Interferon-driven inflammatory macrophages in autoimmune disease

**Objective**: Dampen interferon response while preserving antimicrobial function

**Key Findings**:
- **Top Drug**: Ruxolitinib (JAK1/2 inhibitor)
- **Cell-Line Specific**: Predictions with confidence intervals per cell line
- **Key Genes**: STAT1, IRF1, CXCL10, ISG15 (interferon signature)
- **Pathways**: JAK/STAT signaling, type I/II interferon response
- **Clinical**: FDA-approved, repurposing for inflammatory disease

**Files**:
- `CASE_STUDY_IFN_MACROPHAGES.md` - Analysis with cell-line comparisons
- `data/` - Macrophage scRNA-seq data
- `scripts/` - Cell-line-specific scoring
- `results/` - Predictions with uncertainty quantification
- `figures/` - Cell-line comparison plots

---

## Workflow Demonstrated

Each case study shows the complete pipeline:

1. **Data Acquisition**: scRNA-seq dataset selection and QC
2. **Cell Selection**: Identifying disease-relevant cell population
3. **Target Signature**: Differential expression and L1000 mapping
4. **Quality Control**: Signature validation and QC interpretation
5. **Scoring**: Baseline and metric-based connectivity mapping
6. **Results Analysis**: Top compounds, MOA enrichment, pathway networks
7. **Explainability**: Gene contributions, waterfall plots, automated narratives
8. **Experimental Validation**: Tiered validation plan (in vitro → in vivo → clinical)
9. **Literature Support**: Citations and validation status

## Using These Case Studies

### As Templates

Each case study can serve as a template for your own analysis:

```bash
# Copy case study structure
cp -r case_studies/nsclc_cd8/ case_studies/my_disease/

# Adapt the workflow to your data
cd case_studies/my_disease/
# Edit scripts to use your .h5ad file
# Run analysis
# Compare results to case study
```

### As Validation

Use these to validate your installation:

```bash
# Run complete case study
cd case_studies/nsclc_cd8/scripts/
python run_analysis.py

# Compare outputs to published results
diff results/ ../published_results/
```

### As Training Material

Work through case studies to learn the platform:

1. Read the complete case study markdown
2. Understand the biological context
3. Follow the workflow step-by-step
4. Reproduce the results
5. Interpret the outputs

## Additional Resources

- [Quickstart Guide](../docs/quickstart.md)
- [API Documentation](../docs/api.md)
- [Explainability Framework](../docs/explainability.md)
- [Experimental Validation](../docs/guides/EXPLAINABILITY_FEATURES.md#experimental-validation)

## Citation

If you use these case studies in publications, please cite:

```bibtex
@article{scperturb_cmap_2025,
  title={scPerturb-CMap: Single-Cell Connectivity Mapping for Drug Repurposing},
  author={scPerturb-CMap Development Team},
  journal={TBD},
  year={2025}
}
```

## Questions?

- Open an issue for case study questions
- Email: support@scperturb-cmap.org
- Slack: #case-studies channel
