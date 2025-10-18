# Architecture Overview

This page highlights the key moving parts of scPerturb-CMap deployments and
links to the system diagrams that live under `figs/`.

## High-Level Flow

1. **Target preparation** – Single-cell signatures are distilled to weighted gene
   lists (`TargetSignature` JSON) via the CLI or Streamlit UI.
2. **Connectivity scoring** – The CLI/API encode the target, query the cached
   LINCS matrix, blend baseline and metric scores, and emit ranked compounds.
3. **Explainability & validation** – Explainability helpers generate narratives,
   enrichment plots, and QC metrics to guide follow-up experiments.
4. **Deployment surfaces** – The FastAPI service, Celery workers, and Streamlit
   UI can be deployed via Docker Compose or Kubernetes with shared storage for
   LINCS data and model artifacts.

![System diagram](../../figs/fig1_system_diagram_r.png)

## Performance & Monitoring

The core evaluation loop combines a cosine/GSEA baseline with an optional
DualEncoder metric model. The figure below summarises how blending the metric
model improves Recall@50 across reference cell lines.

![Recall@50 improvement](../../figs/fig2_recall_by_cell_line_r.png)

Mechanism-of-action enrichment helps downstream scientists interpret the top
hits and plan validation experiments.

![MoA enrichment](../../figs/fig3_moa_enrichment_r.png)

For a detailed walkthrough of a representative analysis, review the NSCLC case
study:

![Case study ranking trace](../../figs/fig4_case_study_rank_plot_r.png)

## Additional Resources

- Deployment guide: [`docs/deployment/CLOUD_DEPLOYMENT.md`](../deployment/CLOUD_DEPLOYMENT.md)
- Case studies: [`case_studies/`](../../case_studies/README.md)
- Explainability docs: [`docs/explainability.md`](../explainability.md)

