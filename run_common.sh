# Common simulation logic sourced by run_cpu.sh and run_gpu.sh.
#
# Required variables that must be set by the sourcing script before this file
# is sourced:
#   BACKEND   – 'cpu' or 'gpu'
#   RANKS     – number of MPI ranks passed to srun
#   CORES     – OMP_NUM_THREADS / --cpus-per-task value
#
# Usage:  bash run_{cpu,gpu}.sh [windshock|kelvin]
#
# Output layout:
#   build_${BACKEND}/${TEST}_r${RANKS}_c${CORES}_s${STEPS}_n${N}_${BACKEND}/constants.txt
#   <sphexa-develop>/build_${BACKEND}/${TEST}_r${RANKS}_c${CORES}_s${STEPS}_n${N}_${BACKEND}/constants.txt

# ── Parameters (match the defaults in compare_sphexa_tests.py) ─────────────────
# $1 is inherited from the positional parameters of the sourcing script.
# Pass --nsys as any additional argument to enable nsys profiling (GPU only).
TEST="${1:-windshock}"
NSYS_ENABLED=0
for _arg in "$@"; do
  [ "$_arg" = "--nsys" ] && NSYS_ENABLED=1
done
N=50
STEPS=200

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

# ── Run-directory name (encodes all parameters) ───────────────────────────────
RUN_DIR_NAME="${TEST}_r${RANKS}_c${CORES}_s${STEPS}_n${N}_${BACKEND}"

# ── Binary paths ───────────────────────────────────────────────────────────────
if [ "$BACKEND" = "gpu" ]; then
  BIN_SUFFIX="sphexa-cuda"
else
  BIN_SUFFIX="sphexa"
fi
MIXD_BIN="${MIXD_ROOT}/build_${BACKEND}/main/src/sphexa/${BIN_SUFFIX}"
DEV_BIN="${DEV_ROOT}/build_${BACKEND}/main/src/sphexa/${BIN_SUFFIX}"

# ── Helper: run one simulation inside its own run directory ───────────────────
run_sim() {
  local label="$1"
  local binary="$2"
  local run_dir="$3"
  local log_file="${run_dir}/timing.log"
  local suffix="${4}"

  echo "=== Running ${label} ==="
  echo "    binary   : ${binary}"
  echo "    run dir  : ${run_dir}"
  echo "    init     : ${INIT_COND}  n=${N}  steps=${STEPS}  ranks=${RANKS}  OMP=${CORES}"

  if [ ! -x "$binary" ]; then
    echo "ERROR: binary not found or not executable: $binary" >&2
    exit 1
  fi

  PROFILE_EXEC=""
  if [ "$BACKEND" = "gpu" ] && [ "$NSYS_ENABLED" = "1" ]; then
    NSYS_OUTPUT_FILE="nsys_${TEST}_r${RANKS}_${suffix}"
    PROFILE_EXEC="nsys profile --trace=cuda,nvtx,osrt -o ${NSYS_OUTPUT_FILE} -f true"
  fi

  mkdir -p "${run_dir}"

  local out_prefix="${run_dir}/dump"
  local const_file="${run_dir}/constants.txt"
  rm profile.h5 ${run_dir}/profile.h5 2>/dev/null || true

  OMP_NUM_THREADS="${CORES}" \
  srun -n "${RANKS}" ${PROFILE_EXEC} "${binary}" \
    --glass "${GLASS}" \
    --init  "${INIT_COND}" \
    -n      "${N}" \
    -s      "${STEPS}" \
    -o      "${out_prefix}" \
    --profile 2>&1 | tee "${log_file}"

  mv profile.h5 "${run_dir}/"

  if [ ! -f "${const_file}" ]; then
    echo "ERROR: expected constants file not produced: ${const_file}" >&2
    exit 1
  fi

  echo "    saved    : ${const_file}"
  echo ""
}

# ── Run both branches ──────────────────────────────────────────────────────────
MIXD_RUN_DIR="${MIXD_ROOT}/build_${BACKEND}/${RUN_DIR_NAME}"
DEV_RUN_DIR="${DEV_ROOT}/build_${BACKEND}/${RUN_DIR_NAME}"

run_sim "MixD branch"    "${MIXD_BIN}" "${MIXD_RUN_DIR}" "MixD"
run_sim "develop branch" "${DEV_BIN}"  "${DEV_RUN_DIR}" "develop"

echo "Done. Run compare_sphexa_tests.py to generate the comparison plot."

# ── numHalos comparison ───────────────────────────────────────────────────────
python3 "${MIXD_ROOT}/scripts/compare_num_halos.py" \
    --test "${TEST}" --backend "${BACKEND}" --steps "${STEPS}" \
    --ranks "${RANKS}" --cores "${CORES}" --n "${N}"

# ── Timing comparison ─────────────────────────────────────────────────────────
python3 "${MIXD_ROOT}/scripts/timing_summary.py" \
    "${MIXD_RUN_DIR}/timing.log" "${DEV_RUN_DIR}/timing.log" "${TEST}" "${BACKEND}"
