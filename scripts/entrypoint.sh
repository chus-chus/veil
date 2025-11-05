#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "[entrypoint] Logging into Hugging Face..."
  # Prefer new hf CLI; fall back to huggingface-cli if needed
  conda run --no-capture-output -n veil hf auth login --token "$HUGGINGFACE_HUB_TOKEN"
fi

echo "[entrypoint] Starting Veil API..."
exec conda run --no-capture-output -n veil python3 -m veil --pipeline-config-from-file /app/config/pipeline.yml
