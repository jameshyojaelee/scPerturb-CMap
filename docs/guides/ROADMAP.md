# scPerturb-CMap Roadmap

_Last updated: September 29, 2025_

This document outlines the planned development trajectory for scPerturb-CMap, organized by priority and expected release version. Timelines are approximate and subject to change based on community feedback and resource availability.

---

## Vision & Strategy

**Long-term goal**: Establish scPerturb-CMap as the standard tool for translating single-cell disease signatures into actionable drug repurposing hypotheses.

**Strategic priorities**:
1. **Validation**: Build confidence through real-world case studies and benchmarking
2. **Usability**: Lower barriers for biologists without computational expertise
3. **Scalability**: Enable batch processing and cloud deployment
4. **Intelligence**: Improve ranking accuracy through better models and features
5. **Ecosystem**: Foster community contributions and signature sharing

---

## Release Timeline

```
v0.2.0 (Current) ──► v0.3.0 ──► v0.4.0 ──► v0.5.0 ──► v1.0.0
   |                 |          |          |          |
 Sep 2025        Q1 2026    Q2 2026    Q3-Q4 2026  Q1 2027
                   ↓            ↓          ↓          ↓
              Validation   Intelligence  Scale    Maturity
              & Usability  & Safety      & Cloud  Release
```

---

## v0.3.0 - Validation & Usability (Target: Q1 2026)

**Theme**: Build confidence and lower barriers to entry

### Real-World Case Studies ⭐⭐⭐
- [ ] **NSCLC CD8+ T cell exhaustion**
  - Full workflow from published dataset (e.g., GSE139555)
  - Top-20 ranked compounds with literature validation
  - Experimental validation plan with suggested assays
  - Case study document: `docs/cases/nsclc_cd8.md`
  
- [ ] **EMT breast cancer cells**
  - Patient-derived or PDX-derived scRNA-seq
  - MOA pathway analysis showing TGFβ/WNT enrichment
  - Cell-line-specific predictions (MCF7 vs. MDA-MB-231)
  - Case study document: `docs/cases/emt_breast.md`
  
- [ ] **IFN-high macrophages (inflammation)**
  - COVID-19 or IBD-derived macrophage signatures
  - Anti-inflammatory compound ranking
  - Comparison with known therapeutics (anti-TNF, JAK inhibitors)
  - Case study document: `docs/cases/ifn_macrophage.md`

**Impact**: Demonstrates real-world utility; provides templates for users

### Batch Processing ⭐⭐⭐
- [ ] **CLI command**: `scperturb-cmap score-batch`
  - Accept directory of target JSONs or JSONL file
  - Parallel processing with progress bars (tqdm)
  - Unified output with target_id column
  
- [ ] **Comparative analysis tools**
  - Cross-target heatmap: drugs × targets
  - Hierarchical clustering of signature similarity
  - Rank correlation matrix across targets
  - Consensus top-N drugs appearing in multiple targets
  
- [ ] **Streamlit UI updates**
  - Upload multiple `.h5ad` files
  - Side-by-side comparison mode
  - Interactive heatmap with drill-down
  
- [ ] **Export formats**
  - Multi-sheet Excel with one target per sheet
  - Combined CSV with pivot-friendly format
  - JSON with nested structure

**Impact**: Enables systematic comparisons across patients/conditions

### Enhanced Gene Mapping ⭐⭐⭐
- [ ] **Synonym database integration**
  - Download HGNC, Ensembl, Entrez gene info
  - Build local SQLite cache for fast lookup
  - API fallback via `mygene.info`
  
- [ ] **Fuzzy matching**
  - Levenshtein distance for typos (e.g., "MARCH1" → "MARCHF1")
  - Confidence scores (exact=1.0, fuzzy=0.8, etc.)
  - User-reviewable mapping table
  
- [ ] **Interactive disambiguation**
  - Streamlit UI for ambiguous genes
  - Show all possible mappings with context
  - Save user choices as mapping file
  
- [ ] **QC enhancements**
  - Gene-level quality scores in QC report
  - Unmapped gene list with suggestions
  - `--strict-matching` mode rejecting fuzzy matches

**Impact**: Reduces silent errors from gene symbol mismatches

### Explainability Tools ⭐⭐
- [ ] **Gene-level contribution analysis**
  - Compute per-gene contribution to connectivity score
  - Waterfall plots showing top positive/negative genes
  - Export gene contribution matrix
  
- [ ] **Pathway enrichment**
  - Integrate GO/KEGG/Reactome via `gseapy`
  - Enrichment for top-contributing genes
  - Pathway-drug association heatmap
  
- [ ] **Why-this-drug narratives**
  - Automated text generation citing gene patterns
  - Example: "Dasatinib ranks #1 because it strongly inverts BCR-ABL pathway genes (ABL1 ↓, STAT5A ↓, MYC ↓)"
  
- [ ] **Comparative explanations**
  - "Why Drug A > Drug B" with differential gene contributions

**Impact**: Builds trust by showing mechanistic rationale

### Documentation ⭐⭐
- [ ] **Jupyter notebook tutorials** (5-10 notebooks)
  - Tutorial 1: Basic workflow (`.h5ad` → drugs)
  - Tutorial 2: Batch processing across patients
  - Tutorial 3: MOA enrichment interpretation
  - Tutorial 4: Custom perturbation libraries
  - Tutorial 5: Training the metric model
  
- [ ] **Video walkthroughs** (3-5 videos, <10min each)
  - Streamlit UI tour
  - CLI quickstart
  - Interpreting results
  
- [ ] **ReadTheDocs site**
  - API documentation with examples
  - CLI reference
  - Cookbook of common patterns

**Impact**: Accelerates user onboarding

---

## v0.4.0 - Intelligence & Safety (Target: Q2 2026)

**Theme**: Smarter predictions and safer recommendations

### Safety Integration ⭐⭐⭐
- [ ] **DrugBank integration**
  - FDA approval status (approved/investigational/withdrawn)
  - Clinical trial phases
  - Drug-drug interaction warnings
  - Black-box warnings
  
- [ ] **Tox21 toxicity predictions**
  - Organ-specific toxicity scores (hepatotoxicity, cardiotoxicity)
  - LD50 estimates where available
  - Mutagenicity/carcinogenicity flags
  
- [ ] **Safety filtering**
  - `--safety-filter` CLI option
  - Exclude withdrawn drugs, black-box warnings
  - Risk-benefit scoring in results table
  
- [ ] **Safety dashboard (Streamlit)**
  - Risk profile visualization per compound
  - Known adverse events from FAERS
  - Contraindication warnings

**Impact**: Prioritizes safer compounds for validation

### Advanced Query Language ⭐⭐
- [ ] **SQL-like DSL parser**
  - Syntax: `--filter 'moa IN (kinase_inhibitor, EGFR) AND dose BETWEEN 1,10'`
  - Boolean logic (AND, OR, NOT)
  - Comparison operators (=, !=, <, >, BETWEEN, IN, LIKE)
  
- [ ] **Saved presets**
  - Built-in: `--preset oncology_approved`, `--preset inflammation_research`
  - User-defined presets saved as JSON
  - Preset library in examples/
  
- [ ] **Visual query builder**
  - Drag-and-drop logic in Streamlit
  - Live preview of filtered result count
  - Export query as CLI command

**Impact**: Enables complex filtering without coding

### Community Signature Repository ⭐⭐⭐
- [ ] **Web portal** (Flask/FastAPI + React frontend)
  - Upload target signatures with metadata
  - Browse by disease, tissue, cell type
  - Search with ontology terms (Disease Ontology, Cell Ontology)
  
- [ ] **DOI assignment**
  - Integration with Zenodo for permanent IDs
  - Versioning support
  
- [ ] **Quality badges**
  - Bronze: computationally generated
  - Silver: literature-validated
  - Gold: experimentally confirmed
  
- [ ] **Pre-computed results**
  - Store full LINCS rankings for common signatures
  - Instant results for >100 curated cell types
  
- [ ] **REST API**
  - `/signatures` - list all
  - `/signatures/{id}` - get specific signature
  - `/signatures/{id}/results` - get pre-computed rankings
  - `/search?disease=nsclc&tissue=lung` - search

**Impact**: Enables signature reuse and meta-analysis

### Comprehensive Benchmarking ⭐⭐⭐
- [ ] **Gold-standard dataset curation**
  - Mine PubMed for 500+ known disease-drug inversions
  - Extract from ClinicalTrials.gov (successful trials)
  - Curate from drug repurposing literature
  - Store as `benchmarks/gold_standard.parquet`
  
- [ ] **Benchmark suite**
  - Compute recall@5, recall@10, recall@50
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (NDCG)
  - Stratify by disease category, MOA class
  
- [ ] **Comparison with other tools**
  - CMap (classic connectivity mapping)
  - L1000FWD (web tool)
  - CREEDS (crowd-extracted expression signatures)
  - iLINCS
  
- [ ] **Continuous benchmarking**
  - CI pipeline running benchmarks on each PR
  - Performance regression alerts
  - Public leaderboard website

**Impact**: Quantifies accuracy improvements; guides development

---

## v0.5.0 - Scale & Cloud (Target: Q3-Q4 2026)

**Theme**: Production deployment and enterprise scale

### Advanced Model Architectures ⭐⭐
- [ ] **Transformer-based encoder**
  - Multi-head attention over genes
  - Positional encoding using gene importance
  - Pre-training on all LINCS signatures
  
- [ ] **Graph Neural Network variant**
  - Use STRING/BioGRID protein interaction networks
  - Message passing over gene graphs
  - Edge features from pathway databases
  
- [ ] **Ensemble models**
  - Combine MLP, Transformer, GNN
  - Learned ensemble weights via stacking
  - Uncertainty quantification (prediction intervals)
  
- [ ] **Few-shot learning**
  - Meta-learning (MAML, Prototypical Networks)
  - Adapt to rare diseases with <10 examples
  
- [ ] **Pre-trained checkpoints**
  - 20+ cell types (T cells, B cells, macrophages, epithelial, etc.)
  - Tissue-specific models (lung, breast, brain, etc.)
  - Model zoo with download scripts

**Impact**: State-of-the-art ranking accuracy

### Cloud Deployment ⭐⭐⭐
- [ ] **Docker production image**
  - Multi-stage build (builder + runtime)
  - Optimized LINCS caching layer
  - Health checks and graceful shutdown
  
- [ ] **Kubernetes Helm charts**
  - Auto-scaling based on queue depth
  - Rolling updates with zero downtime
  - Resource limits and requests
  
- [ ] **Cloud provider templates**
  - AWS CloudFormation (ECS Fargate + Lambda)
  - GCP Deployment Manager (GKE + Cloud Functions)
  - Azure Resource Manager (AKS + Functions)
  
- [ ] **Serverless scoring API**
  - AWS Lambda/Cloud Functions handlers
  - API Gateway with rate limiting
  - S3/GCS for LINCS storage (Parquet partitioning)
  - DynamoDB/Firestore for result caching
  
- [ ] **Monitoring & observability**
  - CloudWatch/Stackdriver integration
  - Custom metrics (queries/sec, latency p50/p99)
  - Error tracking (Sentry)
  - Cost optimization (spot instances, reserved capacity)

**Impact**: Enables enterprise adoption and high-throughput workflows

### Spatial Transcriptomics ⭐⭐
- [ ] **Spatial data loaders**
  - Visium (10x Genomics)
  - MERFISH, seqFISH
  - Xenium (10x Genomics)
  - CosMx (NanoString)
  
- [ ] **Neighborhood-aware signatures**
  - k-NN graphs over spatial coordinates
  - Spatial autocorrelation (Moran's I)
  - Niche-specific signatures (tumor-stroma interface)
  
- [ ] **ROI selection tools**
  - Interactive selection in Streamlit
  - Import ROIs from QuPath, HALO
  
- [ ] **Spatially-resolved predictions**
  - Predict drug effects by spatial region
  - 3D visualization of effect maps
  - Export for downstream spatial analysis

**Impact**: Enables context-aware drug discovery

### Temporal Dynamics ⭐
- [ ] **Trajectory-based signatures**
  - scVelo RNA velocity integration
  - Monocle3, PAGA pseudotime
  - Signatures along differentiation paths
  
- [ ] **Differential trajectory queries**
  - "Drugs preventing differentiation toward state X"
  - "Drugs accelerating transition to state Y"
  
- [ ] **Critical transition detection**
  - Identify when in trajectory to intervene
  - Time-to-effect predictions
  
- [ ] **Temporal heatmaps**
  - Drug effectiveness × pseudotime
  - Identify stage-specific therapeutics

**Impact**: Enables temporal precision in drug interventions

### Multi-Omics Integration ⭐
- [ ] **CITE-seq protein signatures**
  - ADT normalization (CLR, DSB)
  - Joint RNA + protein signatures
  - Predict drugs targeting both layers
  
- [ ] **ATAC-seq chromatin accessibility**
  - Differential accessibility signatures
  - Regulatory element-focused drugs
  - Transcription factor activity inference
  
- [ ] **Metabolomics integration**
  - MetaboAnalyst pathway enrichment
  - Metabolic flux predictions
  
- [ ] **Joint embedding spaces**
  - Weighted Nearest Neighbor (Seurat v5)
  - Multi-modal latent representations

**Impact**: Captures regulatory and post-transcriptional effects

---

## v1.0.0 - Maturity Release (Target: Q1 2027)

**Theme**: Stable, documented, production-ready

### Completion Criteria
- [ ] All v0.3-v0.5 features implemented and tested
- [ ] ≥3 peer-reviewed publications using scPerturb-CMap
- [ ] ≥10 experimentally validated case studies
- [ ] Comprehensive documentation (ReadTheDocs + tutorials)
- [ ] Benchmark showing >80% recall@50 on gold-standard dataset
- [ ] Community repository with ≥100 curated signatures
- [ ] Production deployments at ≥3 institutions
- [ ] API stability guarantee (semantic versioning)
- [ ] Long-term support plan

### Release Highlights
- Stable API (breaking changes only in v2.0+)
- Performance benchmarks (throughput, latency)
- Security audit and penetration testing
- Accessibility compliance (WCAG 2.1 AA)
- Internationalization (i18n) support
- Enterprise support options

---

## Beyond v1.0 - Future Directions

**Research Areas** (timeline TBD):
- **Causal inference**: Distinguish correlation from causation in connectivity
- **Multi-target optimization**: Design combination therapies
- **Patient stratification**: Personalize drug rankings by patient genotype
- **Regulatory approval**: Guidance for FDA/EMA submissions
- **Clinical trial design**: Power analysis for repurposing studies
- **Real-world evidence**: Integration with EHR data

**Ecosystem Expansion**:
- **R package**: Native R interface via Bioconductor
- **Nextflow/Snakemake**: Workflow management integration
- **Galaxy**: Web-based analysis interface
- **CellxGene integration**: Query Census single-cell atlas
- **OpenTargets integration**: Disease-target-drug associations

---

## How to Contribute

We prioritize community-driven development. You can contribute by:

1. **Voting on features**: Comment on GitHub Discussions with your priorities
2. **Implementing features**: Pick an issue tagged `help-wanted` or `good-first-issue`
3. **Contributing case studies**: Share your validation results
4. **Reporting bugs**: Help us identify issues early
5. **Improving docs**: Fix typos, add examples, clarify explanations

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

---

## Version History

| Version | Release Date | Highlights |
|---------|-------------|------------|
| v0.2.0  | Sep 2025    | DualEncoder metric learning, Streamlit UI, HPC integration |
| v0.1.1  | Sep 2024    | Initial public release with baseline scoring |

---

## Contact & Feedback

- **Roadmap discussions**: [GitHub Discussions - Roadmap](https://github.com/jameslee/scPerturb-CMap/discussions/categories/roadmap)
- **Feature requests**: [GitHub Issues](https://github.com/jameslee/scPerturb-CMap/issues/new?template=feature_request.md)
- **Email**: For strategic partnerships or funding inquiries

_This roadmap is a living document and will be updated quarterly based on progress and community feedback._
