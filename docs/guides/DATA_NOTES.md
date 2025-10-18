# Data notes for scPerturb-CMap

- Maintainer: James Lee
- Created on: <YYYY-MM-DD>
- HPC base: /gpfs/commons/home/jameslee/scPerturb-CMap

## Raw LINCS sources
- List the exact URLs or data portals used
- Store a copy of URLs in data/raw/lincs_level5_urls.txt

## Commands used on HPC
- make hpc-setup
- sbatch scripts/slurm/download_lincs.sbatch
- sbatch scripts/slurm/prepare_lincs.sbatch
- sbatch scripts/slurm/make_target.sbatch
- sbatch scripts/slurm/score_real.sbatch
- sbatch scripts/slurm/train_metric.sbatch (optional GPU)

## Outputs
- data/lincs/lincs_level5_landmark_long.parquet
- examples/out/target_sig_real.parquet
- examples/out/results_real.parquet
- workspace/artifacts/best.pt
- workspace/artifacts/report.json

## Notes and caveats
- Gene symbol harmonization strategy
- Any filters applied to cell lines or doses
- Known limitations or TODOs

## Licensing & Redistribution
- LINCS L1000 data are distributed by the Broad Institute Connectivity Map (CLUE) team under a
  Creative Commons CC BY 4.0 license. Cite Subramanian *et&nbsp;al.* (Cell, 2017) and the CLUE
  portal in manuscripts and presentations that make use of the dataset.
- When sharing derived subsets (e.g., partitioned Parquet tables) include the original license text
  and attribution, and direct downstream users to retrieve authoritative copies from
  [https://clue.io/connectopedia/data_download](https://clue.io/connectopedia/data_download).
