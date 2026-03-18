#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <sphexa_executable>" >&2
  exit 1
fi

binary_path="$1"
# Expand possible wildcards in binary path
shopt -s nullglob
matches=( $binary_path )
shopt -u nullglob

# We expect exactly one match
if [ ${#matches[@]} -eq 0 ]; then
  echo "Binary path '$binary_path' did not match any file" >&2
  exit 1
fi
if [ ${#matches[@]} -gt 1 ]; then
  echo "Binary path '$binary_path' is ambiguous; matches: ${matches[*]}" >&2
  exit 1
fi
binary_path="${matches[0]}"
case "$(basename "$binary_path")" in
  sphexa-cuda)
    backend="cuda"
    ;;
  sphexa)
    backend="cpu"
    ;;
  *)
    echo "Unable to infer backend from executable '$binary_path'." >&2
    exit 1
    ;;
esac
rank_id="${SLURM_PROCID:-0}"

if [ "$rank_id" -eq 0 ]; then
  wget --quiet -O 50c.h5 https://zenodo.org/records/8369645/files/50c.h5
fi
wait

# Init cases
ics=(
  sedov
  noh
  isobaric-cube
  evrard
  turbulence
  gresho-chan
  wind-shock
  kelvin-helmholtz
)

# Exaclty conserved observables to be compared by absolute value (not relative error) in compare_constants.py.
declare -A abs_columns_for_ic=(
  # TimeAndEnergy: egrav/linmom/angmom (if not changed in CLI, gravConstant = 0.0)
  [sedov]="6,7,8"
  # TimeAndEnergy: egrav/angmom (if not changed in CLI, gravConstant = 0.0)
  [noh]="6,8"
  # TimeAndEnergy: egrav (if not changed in CLI, gravConstant = 0.0)
  [isobaric-cube]="6,7,8"
  # TimeAndEnergy: linmom/angmom
  [evrard]="3,7,8"
  # TimeAndEnergy: egrav/linmom/angmom (if not changed in CLI, eosChoice = 0 and gravConstant = 0.0)
  [turbulence]="6,7,8"
  # TimeAndEnergy: egrav (if not changed in CLI, gravConstant = 0.0)
  [gresho-chan]="6"
  # WindBubble: egrav (if not changed in CLI, gravConstant = 0.0)
  [wind-shock]="6"
  # TimeEnergyGrowth: egrav (if not changed in CLI, gravConstant = 0.0)
  [kelvin-helmholtz]="6,7"
)

failed_comparisons=()

for ic in "${ics[@]}"; do
  if [ "$rank_id" -eq 0 ]; then
    echo "Running test for init condition: $ic"
  fi
  wait
  "$binary_path" --quiet --glass "./50c.h5" --init "$ic" -s 10 -n 50
  if [ "$rank_id" -eq 0 ]; then
    abs_cols="${abs_columns_for_ic[$ic]:-}"
    . constant_comparison_venv/bin/activate
    cmd=(python ci/scripts/compare_constants.py "ci/reference/const-${ic}-${backend}-ref.txt" constants.txt)
    if [ -n "$abs_cols" ]; then
      cmd+=("$abs_cols")
    fi
    if ! "${cmd[@]}"; then
      failed_comparisons+=("$ic")
    fi
    deactivate
  fi
  wait
done

if [ "$rank_id" -eq 0 ]; then
  if [ ${#failed_comparisons[@]} -gt 0 ]; then
    echo "Constant comparison failed for init conditions:" >&2
    for failed_ic in "${failed_comparisons[@]}"; do
      echo "  - $failed_ic" >&2
    done
    exit 1
  fi
  echo "All constant comparisons passed."
fi
