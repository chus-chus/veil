#!/usr/bin/env bash

# Source this file to set up environment variables for this repo.
# It sets PYTHONPATH so that Veil local packages are importable.
# Optionally, it can prepend ./env/bin to PATH and/or install dependencies.
#
# Usage:
#   source scripts/activate_repo.sh [--use-local-env] [--install {gpu|cpu}]
#
# Examples:
#   # Only export PYTHONPATH
#   source scripts/activate_repo.sh
#
#   # Also use local ./env (prepend to PATH)
#   source scripts/activate_repo.sh --use-local-env
#
#   # Install/upgrade deps, then export env (GPU build of torch)
#   source scripts/activate_repo.sh --install gpu --use-local-env

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Please source this script instead:"
  echo "  source scripts/activate_repo.sh"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

use_local_env=false
do_install=""

while (( "$#" )); do
  case "${1:-}" in
    --use-local-env)
      use_local_env=true
      shift ;;
    --install)
      do_install="${2:-}"
      if [[ -z "$do_install" || ("$do_install" != "gpu" && "$do_install" != "cpu") ]]; then
        echo "--install requires 'gpu' or 'cpu' argument" >&2
        return 2
      fi
      shift 2 ;;
    *)
      echo "Unknown option: $1" >&2
      return 2 ;;
  esac
done

if [[ -n "$do_install" ]]; then
  if [[ "$do_install" == "gpu" ]]; then
    bash "$REPO_ROOT/scripts/setup_env.sh" --gpu
  else
    bash "$REPO_ROOT/scripts/setup_env.sh" --cpu
  fi
fi

# Prepend local env bin to PATH if requested and present
if $use_local_env && [[ -d "$REPO_ROOT/env/bin" ]]; then
  export PATH="$REPO_ROOT/env/bin:$PATH"
fi

# Always export PYTHONPATH for local imports
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "[env] REPO_ROOT=$REPO_ROOT"
echo "[env] PYTHONPATH updated to include:"
echo "      $REPO_ROOT"
if $use_local_env && [[ -d "$REPO_ROOT/env/bin" ]]; then
  echo "[env] PATH prepended with: $REPO_ROOT/env/bin"
fi
echo "[env] Done."

