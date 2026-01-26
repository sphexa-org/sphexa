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

use_abs_columns_for_ic() {
  local ic="$1"
  case "$ic" in
    sedov)
      # Colums for TimeAndEnergy observables: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom
      echo "6,7,8"
      ;;
    noh)
      # Colums for TimeAndEnergy observables: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom
      echo "6,8"
      ;;
    isobaric-cube)
      # Colums for TimeAndEnergy observables: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom
      echo "6" # TODO: why linear and angular momentum are not conserved here?
      ;;
    evrard)
      # Colums for TimeAndEnergy observables: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom
      echo "7,8"
      ;;
    turbulence) # TODO: Identify subcases:
      # if eosChoice exists in settings and is == 1
      # Colums for TimeAndEnergy observables: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom
      # else
      # Colums for TurbulenceMachRMS observables: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom, machRms
      echo "6,7,8"
      ;;
    gresho-chan)
      # Colums for TimeAndEnergy observables: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom
      echo "6"
      ;;
    wind-shock)
      # Colums for WindBubble observables: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom, bubbleMass / initialMass, normalizedTime
      echo "6"
      ;;
    kelvin-helmholtz)
      # Colums for TimeEnergyGrowth observables: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom, khgr
      echo "6"
      ;;
    *)
      echo ""
      ;;
  esac
}

for ic in sedov noh isobaric-cube evrard turbulence gresho-chan wind-shock kelvin-helmholtz; do
  if [ "$rank_id" -eq 0 ]; then
    echo "Running test for init condition: $ic"
  fi
  wait
  "$binary_path" --quiet --glass "./50c.h5" --init "$ic" -s 10 -n 50
  if [ "$rank_id" -eq 0 ]; then
    abs_cols=$(use_abs_columns_for_ic "$ic")
    cmd=(python ci/scripts/compare_constants.py "ci/reference/const-${ic}-${backend}-ref.txt" constants.txt)
    if [ -n "$abs_cols" ]; then
      cmd+=("$abs_cols")
    fi
    "${cmd[@]}"
  fi
  wait
done
