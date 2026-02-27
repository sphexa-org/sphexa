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

# Observables compared by absolute value (not relative error) in compare_constants.py.
declare -A abs_columns_for_ic=(
  # TimeAndEnergy: etot/ecin/eint
  [sedov]="6,7,8"
  # TimeAndEnergy: etot/eint
  [noh]="6,8"
  # TimeAndEnergy: etot (TODO: why linear and angular momentum are not conserved as well?)
  [isobaric-cube]="6"
  # TimeAndEnergy: ecin/eint
  [evrard]="7,8"
  # TimeAndEnergy or TurbulenceMachRMS (TODO: Identify subcases split by eosChoice)
  [turbulence]="6,7,8"
  # TimeAndEnergy: etot
  [gresho-chan]="6"
  # WindBubble: etot
  [wind-shock]="6"
  # TimeEnergyGrowth: etot
  [kelvin-helmholtz]="6"
)

# Execution of python comparison script needs numpy which is not available in the used uenv
if [ "$rank_id" -eq 0 ]; then
  echo "Building venv and installing numpy for constants comparison script"
  python -m venv constant_comparison_venv
  source constant_comparison_venv/bin/activate
  pip install --quiet numpy
  deactivate
fi
wait

for ic in "${ics[@]}"; do
  if [ "$rank_id" -eq 0 ]; then
    echo "Running test for init condition: $ic"
  fi
  wait
  "$binary_path" --quiet --glass "./50c.h5" --init "$ic" -s 10 -n 50
  if [ "$rank_id" -eq 0 ]; then
    abs_cols="${abs_columns_for_ic[$ic]:-}"
    source constant_comparison_venv/bin/activate
    cmd=(python ci/scripts/compare_constants.py "ci/reference/const-${ic}-${backend}-ref.txt" constants.txt)
    if [ -n "$abs_cols" ]; then
      cmd+=("$abs_cols")
    fi
    "${cmd[@]}"
    deactivate
  fi
  wait
done
