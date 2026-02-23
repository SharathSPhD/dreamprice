#!/bin/bash
# pytest-guard: PostToolUse hook — run affected tests after every Python file write/edit
# Blocks on test failures. Scopes test run to matching module for speed.
# Fires on: Write, Edit targeting src/**/*.py or tests/**/*.py

FILE="$1"
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

if [[ "$FILE" != *.py ]]; then
  exit 0
fi

# Determine which test module to run based on edited file
if [[ "$FILE" == */src/retail_world_model/data/* ]]; then
  TEST_DIR="tests/test_data"
elif [[ "$FILE" == */src/retail_world_model/models/* ]]; then
  TEST_DIR="tests/test_models"
elif [[ "$FILE" == */src/retail_world_model/training/* ]]; then
  TEST_DIR="tests/test_training"
elif [[ "$FILE" == */src/retail_world_model/inference/* || "$FILE" == */src/retail_world_model/applications/* ]]; then
  TEST_DIR="tests/test_inference"
elif [[ "$FILE" == */src/retail_world_model/api/* ]]; then
  TEST_DIR="tests/test_api"
elif [[ "$FILE" == */src/retail_world_model/envs/* ]]; then
  TEST_DIR="tests/test_envs"
elif [[ "$FILE" == */src/retail_world_model/utils/* ]]; then
  TEST_DIR="tests/test_utils"
elif [[ "$FILE" == */tests/* ]]; then
  # If editing a test file, run just that file
  TEST_DIR="$FILE"
else
  # Unknown location — skip
  exit 0
fi

# Skip if test directory doesn't exist yet (early in setup)
if [[ ! -d "$PROJECT_ROOT/$TEST_DIR" && ! -f "$PROJECT_ROOT/$TEST_DIR" ]]; then
  echo "⏭️  pytest-guard: $TEST_DIR not yet created — skipping"
  exit 0
fi

echo "🔍 pytest-guard: running $TEST_DIR"

if ! pytest "$PROJECT_ROOT/$TEST_DIR" -x --tb=short -q 2>&1; then
  echo "❌ BLOCKED: pytest failures in $TEST_DIR"
  echo "Fix failing tests before continuing."
  exit 1
fi

echo "✅ pytest-guard: $TEST_DIR passed"
exit 0
