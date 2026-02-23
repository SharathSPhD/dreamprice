#!/bin/bash
# ruff-guard: PostToolUse hook — lint and format check after every Python file write/edit
# Blocks on any ruff violation. Fires on: Write, Edit targeting src/**/*.py or tests/**/*.py

FILE="$1"  # File path passed by Claude Code hook system

# Only run on Python files in src/ or tests/
if [[ "$FILE" != *.py ]]; then
  exit 0
fi
if [[ "$FILE" != */src/* && "$FILE" != */tests/* ]]; then
  exit 0
fi

echo "🔍 ruff-guard: checking $FILE"

# Run lint check
if ! ruff check "$FILE" --quiet; then
  echo "❌ BLOCKED: ruff lint violations in $FILE"
  echo "Fix with: ruff check $FILE --fix"
  exit 1
fi

# Run format check (non-destructive — check only, don't auto-format)
if ! ruff format "$FILE" --check --quiet; then
  echo "❌ BLOCKED: ruff format violations in $FILE"
  echo "Fix with: ruff format $FILE"
  exit 1
fi

echo "✅ ruff-guard: $FILE passed"
exit 0
