#!/usr/bin/env bash
set -euo pipefail

venv_dir="${1:-constant_comparison_venv}"

# Keep Python import resolution predictable inside uenv jobs.
unset PYTHONPATH || true
if [ -z "${PYTHONUSERBASE:-}" ]; then
  export PYTHONUSERBASE="$(dirname "$(dirname "$(command -v python)")")"
fi

if [ ! -d "$venv_dir" ] || [ ! -x "$venv_dir/bin/python" ]; then
  echo "Creating testing venv at '$venv_dir'"
  if command -v uv >/dev/null 2>&1; then
    uv venv \
      --python "$(command -v python)" \
      --system-site-packages \
      --seed \
      --relocatable \
      --link-mode=copy \
      "$venv_dir"
  else
    echo "uv not found; falling back to python -m venv" >&2
    python -m venv --system-site-packages "$venv_dir"
  fi
fi

# shellcheck source=/dev/null
source "$venv_dir/bin/activate"

echo "Ensuring numpy is installed in '$venv_dir'"
python -m pip install --quiet numpy
python -c 'import numpy' >/dev/null

deactivate
