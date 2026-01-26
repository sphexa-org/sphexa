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

ignored_columns_for_ic() {
  local ic="$1"
  case "$ic" in
    sedov)
      # Colums: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom
      echo "7,8"
      ;;
    noh)
      # Colums: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom
      echo "7,8"
      ;;
    isobaric-cube)
      # Colums: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom
      echo "7,8"
      ;;
    evrard)
      # Colums: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom
      echo "7,8"
      ;;
    turbulence) # TODO: Identify subcases
      # Colums: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom, machRms
      echo "7,8"
      ;;
    gresho-chan)
      # Colums: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom
      echo "7,8"
      ;;
    wind-shock)
      # Colums: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom, bubbleMass / initialMass, normalizedTime
      echo "7,8"
      ;;
    kelvin-helmholtz)
      # Colums: iteration, ttot, minDt, etot, ecin, eint, egrav, linmom, angmom, khgr
      echo "7,8"
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
    ignore_cols=$(ignored_columns_for_ic "$ic")
    cmd=(python ci/scripts/compare_constants.py "ci/reference/const-${ic}-${backend}-ref.txt" constants.txt)
    if [ -n "$ignore_cols" ]; then
      cmd+=("$ignore_cols")
    fi
    "${cmd[@]}"
  fi
  wait
done
