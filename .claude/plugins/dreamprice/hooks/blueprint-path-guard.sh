#!/bin/bash
# blueprint-path-guard: PreToolUse hook — warn when creating new .py files not in blueprint §12
# Non-blocking: agent must acknowledge but can proceed.
# Fires on: Write creating a new .py file under src/retail_world_model/

FILE="$1"

# Only apply to new Python files in src/retail_world_model/
if [[ "$FILE" != *.py || "$FILE" != */src/retail_world_model/* ]]; then
  exit 0
fi

# Skip if file already exists (this is an edit, not a creation)
if [[ -f "$FILE" ]]; then
  exit 0
fi

# Canonical module paths from blueprint §12
BLUEPRINT_PATHS=(
  "data/dominicks_loader.py"
  "data/copula_correction.py"
  "data/transforms.py"
  "data/schemas.py"
  "models/world_model.py"
  "models/rssm.py"
  "models/mamba_backbone.py"
  "models/encoder.py"
  "models/decoder.py"
  "models/reward_head.py"
  "models/posterior.py"
  "training/trainer.py"
  "training/losses.py"
  "training/offline_utils.py"
  "inference/imagination.py"
  "inference/planning.py"
  "applications/pricing_policy.py"
  "applications/competitive_robustness.py"
  "api/serve.py"
  "api/endpoints.py"
  "api/batching.py"
  "api/schemas.py"
  "envs/base.py"
  "envs/grocery.py"
  "utils/distributions.py"
  "utils/checkpoint.py"
  "utils/device.py"
  "__init__.py"
  "data/__init__.py"
  "models/__init__.py"
  "training/__init__.py"
  "inference/__init__.py"
  "applications/__init__.py"
  "api/__init__.py"
  "envs/__init__.py"
  "utils/__init__.py"
)

# Extract relative path within retail_world_model/
REL_PATH=$(echo "$FILE" | sed 's|.*/src/retail_world_model/||')

FOUND=false
for BLUEPRINT_PATH in "${BLUEPRINT_PATHS[@]}"; do
  if [[ "$REL_PATH" == "$BLUEPRINT_PATH" ]]; then
    FOUND=true
    break
  fi
done

if [[ "$FOUND" == "false" ]]; then
  echo "⚠️  BLUEPRINT WARNING: '$REL_PATH' is not in the blueprint §12 module structure."
  echo "   Blueprint paths: docs/project-blueprint.md §12"
  echo "   If this is intentional (e.g., a new utility), acknowledge and proceed."
  echo "   If this is a mistake, correct the file path before writing."
  # Non-blocking — exit 0 allows the write to proceed
fi

exit 0
