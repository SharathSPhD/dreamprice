#!/bin/bash
# pyright-guard: PostToolUse hook — type check after every Python file write/edit
# Warns during RED phase (test writing), blocks during GREEN/REFACTOR (implementation)
# Fires on: Write, Edit targeting src/**/*.py

FILE="$1"
PHASE="${DREAMPRICE_TDD_PHASE:-green}"  # Set to 'red' during test writing

# Only run on src/ Python files (not tests — they often have intentional type looseness)
if [[ "$FILE" != *.py || "$FILE" != */src/* ]]; then
  exit 0
fi

echo "🔍 pyright-guard: checking $FILE (phase: $PHASE)"

OUTPUT=$(pyright "$FILE" --outputjson 2>/dev/null)
ERRORS=$(echo "$OUTPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('summary',{}).get('errorCount',0))" 2>/dev/null || echo "0")

if [[ "$ERRORS" -gt 0 ]]; then
  if [[ "$PHASE" == "red" ]]; then
    echo "⚠️  WARN: $ERRORS pyright error(s) in $FILE (RED phase — non-blocking)"
    echo "$OUTPUT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for diag in d.get('generalDiagnostics', []):
    if diag.get('severity') == 'error':
        print(f\"  Line {diag['range']['start']['line']}: {diag['message']}\")
" 2>/dev/null
    exit 0
  else
    echo "❌ BLOCKED: $ERRORS pyright error(s) in $FILE (GREEN/REFACTOR phase)"
    echo "$OUTPUT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for diag in d.get('generalDiagnostics', []):
    if diag.get('severity') == 'error':
        print(f\"  Line {diag['range']['start']['line']}: {diag['message']}\")
" 2>/dev/null
    exit 1
  fi
fi

echo "✅ pyright-guard: $FILE passed"
exit 0
