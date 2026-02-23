# Workflow — DreamPrice

## TDD Policy
**Strict.** Tests must be written before implementation. Every task follows:
1. **RED** — Write failing test(s)
2. **GREEN** — Write minimal implementation to pass
3. **REFACTOR** — Simplify, then re-run tests

No implementation code is committed without a corresponding test. Exception: configuration YAML files and notebook cells.

## Commit Strategy
Conventional Commits:
```
feat: add CausalDemandDecoder with frozen elasticity weights
fix: zero SSM state at episode boundary in imagination rollouts
test: add Hausman IV F-stat validation for canned soup category
refactor: extract symlog/symexp into utils/distributions.py
```

Commit after each GREEN step — never batch multiple tasks into one commit.

## Code Review Policy
Required at every **track completion gate** (before advancing to next track). The `test-reviewer-agent` runs automatically via hook. Human review required for:
- Any change to `CausalDemandDecoder` (frozen weights must remain frozen)
- Any change to temporal data split logic
- Any new dependency added to `pyproject.toml`

## Verification Checkpoints
**After each task (automated via hooks):**
- `ruff check <file> && ruff format --check <file>` — blocks on violation
- `pyright <file>` — warns during RED, blocks during GREEN/REFACTOR
- `pytest tests/<matching_module>/ -x` — blocks on failure

**After each track (gate before advancing):**
- Full `pytest tests/` suite — zero failures required
- `pyright src/` — zero errors required
- `ruff check src/` — zero violations required
- `docs/progress/<track-name>.md` written with summary

## Task Lifecycle
```
pending → in_progress → completed
```
Agents claim tasks by setting `owner` via TaskUpdate. Completed = tests pass + type-checked + linted. Never mark complete if tests fail.

## Parallel Dispatch Rules
Within a track, tasks are dispatched in parallel only when they have no file-level dependency. Example safe parallel: `schemas.py` + `transforms.py` (no imports between them). Example unsafe parallel: `rssm.py` + `world_model.py` (world_model imports rssm).

## Agent Communication
All inter-agent communication via SendMessage. Never assume a teammate sees your text output — always use the tool. Task status updates via TaskUpdate.
