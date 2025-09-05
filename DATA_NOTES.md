# Data notes for scPerturb-CMap

- Maintainer: James Lee
- Created on: <YYYY-MM-DD>
- HPC base: /gpfs/commons/home/jameslee/scPerturb-CMap

## Raw LINCS sources
- List the exact URLs or data portals used
- Store a copy of URLs in data/raw/lincs_level5_urls.txt

## Commands used on HPC
- make hpc-setup
- sbatch scripts/download_lincs.sbatch
- sbatch scripts/prepare_lincs.sbatch
- sbatch scripts/make_target.sbatch
- sbatch scripts/score_real.sbatch
- sbatch scripts/train_metric.sbatch (optional GPU)

## Outputs
- data/lincs/lincs_level5_landmark_long.parquet
- examples/out/target_sig_real.parquet
- examples/out/results_real.parquet
- artifacts/best.pt
- artifacts/report.json

## Notes and caveats
- Gene symbol harmonization strategy
- Any filters applied to cell lines or doses
- Known limitations or TODOs

