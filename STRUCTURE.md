# Repository Structure

This document describes the organization of the scPerturb-CMap repository.

## Directory Overview

```
scPerturb-CMap/
├── 📄 Core Project Files
│   ├── README.md                    # Main project documentation
│   ├── LICENSE                      # MIT License
│   ├── pyproject.toml              # Python package configuration
│   ├── Makefile                    # Build and development commands
│   ├── environment.yml             # Conda environment specification
│   ├── mkdocs.yml                  # Documentation site configuration
│   └── CITATION.cff                # Citation metadata
│
├── 📚 docs/                        # Documentation
│   ├── index.md                    # Documentation home
│   ├── quickstart.md               # Getting started guide
│   ├── api.md                      # API reference
│   ├── cli.md                      # Command-line interface guide
│   ├── hpc.md                      # HPC cluster deployment
│   ├── explainability.md           # Explainability framework guide
│   │
│   ├── guides/                     # Detailed guides
│   │   ├── README.md
│   │   ├── CHANGELOG.md            # Version history
│   │   ├── ROADMAP.md              # Future development plans
│   │   ├── DATA_NOTES.md           # Data preparation notes
│   │   └── EXPLAINABILITY_FEATURES.md  # Explainability showcase
│   │
│   ├── contributing/               # Contribution information
│   │   ├── README.md
│   │   ├── CONTRIBUTING.md         # Contribution guidelines
│   │   └── EXPLAINABILITY_SUMMARY.md   # Developer guide
│   │
│   ├── deployment/                 # Deployment documentation
│   │   ├── README.md
│   │   └── CLOUD_DEPLOYMENT.md     # Cloud deployment guide
│   │
│   ├── cases/                      # Case study documentation
│   │   ├── emt_breast.md
│   │   ├── ifn_macrophage.md
│   │   └── nsclc_cd8.md
│   │
│   └── assets/                     # Documentation assets
│       ├── top20_nsclc.png
│       └── ui.png
│
├── 🐍 src/                         # Source code
│   └── scperturb_cmap/
│       ├── __init__.py
│       ├── cli.py                  # Command-line interface
│       ├── config.py               # Configuration management
│       │
│       ├── api/                    # Public APIs
│       │   ├── score.py            # Scoring API
│       │   └── explain.py          # Explainability API
│       │
│       ├── models/                 # ML models
│       │   ├── dual_encoder.py     # DualEncoder model
│       │   ├── baseline.py         # Baseline methods
│       │   ├── train.py            # Training routines
│       │   └── evaluate.py         # Evaluation metrics
│       │
│       ├── data/                   # Data loading & processing
│       │   ├── lincs_loader.py
│       │   └── pairs.py
│       │
│       ├── io/                     # Input/output
│       │   ├── schemas.py          # Data schemas
│       │   ├── serde.py            # Serialization
│       │   └── cloud_storage.py    # Cloud-optimized I/O
│       │
│       ├── explainability/         # Explainability framework
│       │   ├── __init__.py
│       │   ├── feature_importance.py  # Gene contributions
│       │   ├── pathway_enrichment.py  # Pathway analysis
│       │   ├── narrative_generator.py # Auto narratives
│       │   └── uncertainty.py         # Confidence intervals
│       │
│       ├── analysis/               # Analysis utilities
│       │   ├── aggregate.py
│       │   └── enrichment.py
│       │
│       ├── ui/                     # Streamlit UI
│       │   └── app.py
│       │
│       ├── utils/                  # Utilities
│       │   ├── device.py
│       │   ├── seed.py
│       │   └── metrics.py          # Monitoring metrics
│       │
│       ├── viz/                    # Visualization
│       │   └── plots.py
│       │
│       └── configs/                # Default configurations
│           └── *.yaml
│
├── 🧪 tests/                       # Test suite
│   ├── conftest.py                 # Test configuration
│   ├── test_*.py                   # Test modules
│   └── test_explainability.py      # Explainability tests
│
├── 📊 examples/                    # Example code
│   ├── data/                       # Example datasets
│   │   ├── demo.h5ad
│   │   ├── lincs_demo.parquet
│   │   └── demo_gene_sets.json
│   │
│   ├── out/                        # Example outputs
│   │   └── ...
│   │
│   └── explainability_demo.py      # Explainability demo
│
├── 🔬 case_studies/                # Real-world case studies
│   ├── README.md                   # Case studies index
│   │
│   ├── nsclc_cd8/                  # NSCLC T cell exhaustion
│   │   ├── CASE_STUDY_NSCLC_CD8.md
│   │   ├── data/
│   │   ├── scripts/
│   │   ├── results/
│   │   └── figures/
│   │
│   ├── emt_breast/                 # EMT in breast cancer
│   │   ├── CASE_STUDY_EMT_BREAST.md
│   │   ├── data/
│   │   ├── scripts/
│   │   ├── results/
│   │   └── figures/
│   │
│   └── ifn_macrophages/            # IFN-high macrophages
│       ├── CASE_STUDY_IFN_MACROPHAGES.md
│       ├── data/
│       ├── scripts/
│       ├── results/
│       └── figures/
│
├── 🚀 deployment/                  # Deployment configurations
│   ├── README.md                   # Deployment overview
│   │
│   ├── docker/                     # Docker configurations
│   │   ├── README.md
│   │   ├── Dockerfile              # Development image
│   │   ├── Dockerfile.prod         # Production image
│   │   ├── .dockerignore
│   │   └── docker-compose.prod.yml
│   │
│   ├── kubernetes/helm/            # Helm charts
│   │   └── scperturb-cmap/
│   │       ├── Chart.yaml
│   │       ├── values.yaml
│   │       └── templates/
│   │
│   ├── aws/                        # AWS deployment
│   │   ├── cloudformation/
│   │   │   ├── vpc-networking.yaml
│   │   │   ├── s3-storage.yaml
│   │   │   ├── ecs-fargate.yaml
│   │   │   └── lambda-api.yaml
│   │   └── lambda/
│   │       ├── scoring_handler.py
│   │       └── requirements.txt
│   │
│   ├── gcp/                        # GCP deployment
│   │   ├── deployment-manager/
│   │   │   ├── gke-cluster.yaml
│   │   │   └── gcs-storage.yaml
│   │   └── cloud-functions/
│   │       ├── main.py
│   │       ├── requirements.txt
│   │       └── deploy.sh
│   │
│   ├── ci/                         # CI/CD configurations
│   │   ├── README.md
│   │   └── .gitlab-ci.yml
│   │
│   ├── prometheus/                 # Monitoring
│   │   ├── prometheus.yml
│   │   └── rules/
│   │
│   ├── grafana/                    # Dashboards
│   │   └── dashboards/
│   │
│   └── nginx/                      # Reverse proxy
│       └── nginx.conf
│
├── 🛠️ scripts/                     # Utility scripts
│   ├── check_acceptance.py
│   ├── reproduce.sh
│   │
│   ├── data/                       # Data processing
│   │   └── prepare_lincs_subset.py
│   │
│   ├── demo/                       # Demo generation
│   │   ├── make_demo_h5ad.py
│   │   └── print_demo_stats.py
│   │
│   ├── hpc/                        # HPC cluster scripts
│   │   ├── setup_hpc.sh
│   │   └── ui_hpc.sh
│   │
│   ├── slurm/                      # SLURM job scripts
│   │   ├── *.sbatch
│   │   └── ...
│   │
│   ├── api/                        # API server
│   │   └── main.py                 # FastAPI server
│   │
│   ├── validators/                 # Validation utilities
│   │   └── validate_parquet_dataset.py
│   │
│   └── fig*.R                      # Figure generation scripts
│
├── 📁 data/                        # Data directory
│   ├── l1000_landmarks.txt
│   │
│   ├── lincs/                      # LINCS library
│   │   ├── lincs_demo_long_landmark.parquet
│   │   └── lincs_level5_landmark_long/
│   │
│   ├── raw/                        # Raw data files
│   │   └── GSE*.gctx
│   │
│   └── sc/                         # Single-cell data
│       └── your_study.h5ad
│
├── 📈 results/                     # Analysis results
│   └── *.csv
│
├── 📊 figs/                        # Generated figures
│   └── *.png, *.pdf
│
├── 💼 workspace/                   # Runtime workspace
│   ├── artifacts/                  # Model checkpoints
│   │   ├── best.pt
│   │   └── metrics.json
│   │
│   ├── logs/                       # Log files
│   │   └── train.log
│   │
│   ├── cache/                      # Cache directory
│   │
│   ├── notes/                      # Development notes
│   │
│   └── site/                       # MkDocs build output
│
├── 📦 recipe/                      # Conda recipe
│   └── meta.yaml
│
└── 🔧 Configuration Files
    ├── .github/workflows/          # GitHub Actions
    │   └── build-and-deploy.yml
    ├── .gitignore                  # Git ignore patterns
    ├── .hydra/                     # Hydra configs (runtime)
    ├── .pytest_cache/              # Pytest cache
    ├── .ruff_cache/                # Ruff linter cache
    └── .venv/                      # Virtual environment (local)
```

## Key Directories

### Source Code (`src/`)
Contains all production code organized by functionality. The `explainability/` module provides SHAP-like interpretability features.

### Documentation (`docs/`)
Comprehensive documentation organized by category:
- **Core docs**: API, CLI, quickstart
- **Guides**: Detailed guides and changelogs
- **Contributing**: Developer resources
- **Deployment**: Cloud deployment guides

### Deployment (`deployment/`)
Production deployment configurations for:
- **Docker**: Containerized deployment
- **Kubernetes**: Helm charts with auto-scaling
- **AWS**: CloudFormation templates, Lambda functions
- **GCP**: GKE, Cloud Functions
- **CI/CD**: Continuous integration pipelines
- **Monitoring**: Prometheus, Grafana

### Case Studies (`case_studies/`)
Three comprehensive real-world examples demonstrating the complete workflow from single-cell data to validated drug predictions.

### Tests (`tests/`)
Comprehensive test suite with ~90% code coverage, including unit tests, integration tests, and explainability framework tests.

### Examples (`examples/`)
Working examples with demo data and scripts to learn the platform.

## File Naming Conventions

### Python Files
- `snake_case.py` for modules
- `test_*.py` for test files
- `__init__.py` for package initialization

### Documentation
- `UPPERCASE.md` for top-level guides (CHANGELOG, CONTRIBUTING, etc.)
- `lowercase.md` for standard docs (api.md, cli.md, etc.)

### Data Files
- `*.parquet` for tabular data
- `*.h5ad` for AnnData (single-cell)
- `*.json` for configurations and schemas
- `*.yaml` for configs and Kubernetes manifests

### Configuration Files
- `.yaml` for Hydra configs
- `.yml` for CI/CD and Docker Compose
- `.toml` for Python project config

## Important Files

### Project Configuration
- **`pyproject.toml`**: Python package metadata, dependencies, and tool configs
- **`Makefile`**: Development commands (`make setup`, `make test`, etc.)
- **`environment.yml`**: Conda environment specification

### Entry Points
- **`src/scperturb_cmap/cli.py`**: Command-line interface entry point
- **`src/scperturb_cmap/ui/app.py`**: Streamlit UI entry point
- **`scripts/api/main.py`**: FastAPI REST API entry point

### Documentation Entry Points
- **`README.md`**: Project overview and quick start
- **`docs/index.md`**: Documentation home page
- **`docs/quickstart.md`**: Getting started guide

## Navigation Tips

### Finding Functionality
- **CLI commands**: `src/scperturb_cmap/cli.py`
- **API functions**: `src/scperturb_cmap/api/`
- **Models**: `src/scperturb_cmap/models/`
- **Data loading**: `src/scperturb_cmap/data/` or `src/scperturb_cmap/io/`
- **Explainability**: `src/scperturb_cmap/explainability/`

### Finding Documentation
- **Quick start**: `docs/quickstart.md`
- **API reference**: `docs/api.md`
- **CLI reference**: `docs/cli.md`
- **Deployment**: `docs/deployment/` or `deployment/README.md`
- **Contributing**: `docs/contributing/CONTRIBUTING.md`

### Finding Examples
- **Basic usage**: `examples/`
- **Real-world cases**: `case_studies/`
- **Tests**: `tests/` (serve as usage examples)

## Maintenance

### When Adding New Features
1. Add code to appropriate `src/scperturb_cmap/` subdirectory
2. Add tests to `tests/`
3. Add examples to `examples/` (if user-facing)
4. Update documentation in `docs/`
5. Update `CHANGELOG.md` in `docs/guides/`

### When Adding New Documentation
- **API docs**: Add to `docs/api.md`
- **Guides**: Add to `docs/guides/`
- **Deployment**: Add to `docs/deployment/`
- **Case studies**: Add to `case_studies/`

### When Adding Deployment Configs
- **Docker**: Add to `deployment/docker/`
- **Kubernetes**: Add to `deployment/kubernetes/helm/`
- **Cloud**: Add to `deployment/aws/` or `deployment/gcp/`
- **CI/CD**: Add to `deployment/ci/` or `.github/workflows/`

## See Also

- [Contributing Guide](docs/contributing/CONTRIBUTING.md)
- [Deployment Overview](deployment/README.md)
- [Documentation Index](docs/index.md)
