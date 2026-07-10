#!/usr/bin/env bash
#SBATCH --job-name=MixD_SPHEXA
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH -A csstaff
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --view=default

set -euo pipefail

BACKEND="gpu"
RANKS=1
CORES=1

SCRIPT_DIR="$(git rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/scripts/benchmark/run_common.sh"
