#!/bin/bash
#SBATCH --account=crislab
#SBATCH --job-name=d4vd_ir25
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=0-02:00
#SBATCH --output=logs/d4vd_ir25_%j.out
#SBATCH --error=logs/d4vd_ir25_%j.err

set -euo pipefail
cd /insomnia001/depts/crislab/users/sh4947/polymarket_project

module load anaconda
conda activate ./envs/poly

# Needed only if the compute node refreshes wallet metadata over the network.
export POLYGONSCAN_API_KEY=DMMKC6M33S9W3T77SRE1T3JG4A9ZPDECVD

mkdir -p logs audit_out

python supercompute.py \
  --market will-d4vd-be-the-1-searched-person-on-google-this-year \
  --jobs "${SLURM_CPUS_PER_TASK:-32}" \
  --out-dir audit_out
