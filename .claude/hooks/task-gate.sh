#!/bin/bash
# TaskCompleted hook — quality gate before any task can be marked complete
# Exit 2 to block completion and send feedback to agent

PROJECT=/home/sharaths/projects/dreamprice
cd "$PROJECT" || exit 0

# Only gate if src/ exists (skip early scaffold tasks)
if [ ! -d "src" ]; then
  exit 0
fi

FAIL=0
FEEDBACK=""

# Gate 1: ruff
if command -v ruff &>/dev/null && [ -d src ]; then
  if ! ruff check src/ --quiet 2>/dev/null; then
    FAIL=1
    FEEDBACK+="❌ ruff: lint violations found. Run: ruff check src/ --fix\n"
  fi
fi

# Gate 2: pyright
if command -v pyright &>/dev/null && [ -d src ]; then
  ERRORS=$(pyright src/ --outputjson 2>/dev/null | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(d.get('summary',{}).get('errorCount',0))" 2>/dev/null || echo "0")
  if [ "$ERRORS" -gt 0 ]; then
    FAIL=1
    FEEDBACK+="❌ pyright: $ERRORS type error(s) in src/. Run: pyright src/\n"
  fi
fi

# Gate 3: pytest
if command -v pytest &>/dev/null && [ -d tests ]; then
  if ! pytest tests/ -x --tb=short -q 2>/dev/null; then
    FAIL=1
    FEEDBACK+="❌ pytest: test failures. Fix before marking complete.\n"
  fi
fi

if [ $FAIL -eq 1 ]; then
  echo -e "TASK COMPLETION BLOCKED:\n$FEEDBACK\nFix all issues, then retry TaskUpdate."
  exit 2
fi

echo "✅ All quality gates passed — task marked complete."
exit 0
