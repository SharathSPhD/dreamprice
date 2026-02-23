#!/bin/bash
# TeammateIdle hook — if unblocked pending tasks exist, send feedback to keep agent working
# Exit 2 to send feedback and prevent idle

TEAM=dreamprice
TASK_DIR=~/.claude/tasks/$TEAM

# Count pending tasks with no blockedBy (or all blockers resolved)
PENDING_COUNT=0
for task_file in "$TASK_DIR"/*.json; do
  [ -f "$task_file" ] || continue
  STATUS=$(python3 -c "import json; d=json.load(open('$task_file')); print(d.get('status',''))" 2>/dev/null)
  OWNER=$(python3 -c "import json; d=json.load(open('$task_file')); print(d.get('owner',''))" 2>/dev/null)
  BLOCKED_BY=$(python3 -c "import json; d=json.load(open('$task_file')); print(len(d.get('blockedBy',[])))" 2>/dev/null || echo "0")

  if [ "$STATUS" = "pending" ] && [ -z "$OWNER" ] && [ "$BLOCKED_BY" = "0" ]; then
    SUBJECT=$(python3 -c "import json; d=json.load(open('$task_file')); print(d.get('subject',''))" 2>/dev/null)
    PENDING_COUNT=$((PENDING_COUNT + 1))
    AVAILABLE_TASK="$SUBJECT"
  fi
done

if [ "$PENDING_COUNT" -gt 0 ]; then
  echo "There are $PENDING_COUNT unblocked pending task(s) available. Check TaskList and claim: '$AVAILABLE_TASK'. Use TaskUpdate to set yourself as owner, then start working."
  exit 2
fi

exit 0
