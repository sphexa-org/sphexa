#!/usr/bin/env bash
#SBATCH --job-name=MixD_SPHEXA
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH -A csstaff

# Run wind-shock (or kelvin-helmholtz) simulations for both the MixD branch and the
# develop branch, producing the constants files that compare_sphexa_tests.py expects.
#
# Usage:  bash run_wind.sh [windshock|kelvin]
#
# Output files (must match compare_sphexa_tests.py variables):
#   build_${BACKEND}/constants_${TEST}_r${RANKS}_c${CORES}_s${STEPS}_n${N}_${BACKEND}.txt
#   <sphexa-develop>/build_${BACKEND}/constants_${TEST}_r${RANKS}_c${CORES}_s${STEPS}_n${N}_${BACKEND}.txt

set -euo pipefail

# ── Parameters (match the defaults in compare_sphexa_tests.py) ─────────────────
TEST="${1:-windshock}"
BACKEND="cpu"        # 'cpu' or 'gpu'
N=50
STEPS=200
RANKS=4
CORES=72

# ── Directories ────────────────────────────────────────────────────────────────
MIXD_ROOT="/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa"
DEV_ROOT="/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa-develop"
GLASS="/capstor/scratch/cscs/ioannmag/CORNERSTONE/50c.h5"

# ── Resolve init-condition name (CLI uses dashes, compare script uses no dash) ─
case "$TEST" in
  windshock) INIT_COND="wind-shock" ;;
  kelvin)    INIT_COND="kelvin-helmholtz" ;;
  *)
    echo "ERROR: TEST must be 'windshock' or 'kelvin', got: '$TEST'" >&2
    exit 1
    ;;
esac

# ── Derived constants-file basename ────────────────────────────────────────────
CONST_NAME="constants_${TEST}_r${RANKS}_c${CORES}_s${STEPS}_n${N}_${BACKEND}.txt"

# ── Binary paths ───────────────────────────────────────────────────────────────
if [ "$BACKEND" = "gpu" ]; then
  BIN_SUFFIX="sphexa-cuda"
else
  BIN_SUFFIX="sphexa"
fi
MIXD_BIN="${MIXD_ROOT}/build_${BACKEND}/main/src/sphexa/${BIN_SUFFIX}"
DEV_BIN="${DEV_ROOT}/build_${BACKEND}/main/src/sphexa/${BIN_SUFFIX}"

# ── Helper: run one simulation and save the constants file ─────────────────────
run_sim() {
  local label="$1"
  local binary="$2"
  local build_dir="$3"

  echo "=== Running ${label} ==="
  echo "    binary   : ${binary}"
  echo "    init     : ${INIT_COND}  n=${N}  steps=${STEPS}  ranks=${RANKS}  OMP=${CORES}"

  if [ ! -x "$binary" ]; then
    echo "ERROR: binary not found or not executable: $binary" >&2
    exit 1
  fi

  # Place the dump file (and therefore constants.txt) inside build_${BACKEND}/
  local out_prefix="${build_dir}/dump_${TEST}"
  local const_src="${build_dir}/constants.txt"
  local const_dst="${build_dir}/${CONST_NAME}"

  OMP_NUM_THREADS="${CORES}" \
  srun -n "${RANKS}" "${binary}" \
    --quiet \
    --glass "${GLASS}" \
    --init  "${INIT_COND}" \
    -n      "${N}" \
    -s      "${STEPS}" \
    -o      "${out_prefix}"

  if [ ! -f "${const_src}" ]; then
    echo "ERROR: expected constants file not produced: ${const_src}" >&2
    exit 1
  fi

  if [ -f "${const_dst}" ]; then
    echo "WARNING: destination constants file already exists and will be overwritten: ${const_dst}" >&2
  fi

  mv "${const_src}" "${const_dst}"
  echo "    saved    : ${const_dst}"
  echo ""
}

# ── Run both branches ──────────────────────────────────────────────────────────
run_sim "MixD branch"   "${MIXD_BIN}" "${MIXD_ROOT}/build_${BACKEND}"
run_sim "develop branch" "${DEV_BIN}"  "${DEV_ROOT}/build_${BACKEND}"

echo "Done. Run compare_sphexa_tests.py to generate the comparison plot."
