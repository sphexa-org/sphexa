#!/usr/bin/env bash
set -euo pipefail

venv_dir="constant_comparison_venv"

if [ ! -d "$venv_dir" ]; then
  echo "Creating python venv at '$venv_dir' for constants comparison"
  python -m venv "$venv_dir"
fi

source "$venv_dir/bin/activate"
pip install --quiet numpy
deactivate
