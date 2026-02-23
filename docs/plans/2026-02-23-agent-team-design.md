# DreamPrice Agent Team Design

**Date:** 2026-02-23
**Scope:** Full 13-week end-to-end implementation of DreamPrice
**Orchestration:** Conductor tracks + parallel dispatch within each track
**Agent home:** `.claude/plugins/dreamprice/` (Claude Code plugin)

---

## 1. Conductor Track DAG

Seven tracks in strict dependency order. Parallel dispatch of independent subtasks within each track.

```
Track 1: data-pipeline          (weeks 1–2)
Track 2: world-model            (weeks 3–4)   blocked by: Track 1
Track 3: training-loop          (weeks 5–6)   blocked by: Track 2
Track 4: baselines              (weeks 7–8)   blocked by: Track 3
Track 5: ablations              (weeks 9–10)  blocked by: Track 4
Track 6: paper-and-packaging    (weeks 11–12) blocked by: Track 5
Track 7: demo-and-serving       (weeks 11–12) blocked by: Track 3 (can run parallel with Track 6)
```

Each track follows the TDD discipline: RED (write failing tests) → GREEN (implement) → REFACTOR (simplify). Governed by `conductor:workflow-patterns` skill.

### Track Completion Gate

Before advancing to the next track, ALL of the following must pass:

1. `pytest tests/ -x` — zero failures
2. `pyright src/` — zero type errors
3. `ruff check src/` — zero lint violations
4. `docs/progress/<track-name>.md` written with summary of what was built

---

## 2. Plugin Structure: `dreamprice`

Location: `.claude/plugins/dreamprice/`

```
.claude/plugins/dreamprice/
├── plugin.json                     # Plugin manifest
├── agents/
│   ├── causal-estimator.md         # New agent: DML-PLIV + Hausman IV estimation
│   └── experiment-tracker.md       # New agent: W&B runs + ablation statistical analysis
├── hooks/
│   ├── ruff-guard.sh               # PostToolUse: lint + format check
│   ├── pyright-guard.sh            # PostToolUse: type check
│   ├── pytest-guard.sh             # PostToolUse: run affected tests
│   └── blueprint-path-guard.sh     # PreToolUse: verify new file paths match blueprint §12
└── skills/
    └── dreamprice-context.md       # Shared context skill: blueprint summary + conventions
```

---

## 3. Agent Roster

### New Agents (built as part of this project)

#### `causal-estimator-agent`

- **Purpose:** Estimate per-category price elasticities via DML-PLIV and 2sCOPE; validate instruments; output frozen elasticity config for `CausalDemandDecoder`.
- **Input:** Preprocessed DataFrame path, category name
- **Output:** `configs/elasticities/<category>.json` containing `{theta_causal, f_stat, hausman_p, sargan_p, cope_beta_c}`
- **Gate:** F-stat > 10 (weak instrument check), Hausman test p < 0.05; raises `WeakInstrumentError` otherwise
- **Tools:** Bash, Read, Write
- **Skills:** `python-development:python-testing-patterns`, `python-development:python-error-handling`

#### `experiment-tracker-agent`

- **Purpose:** Manage W&B ablation sweeps; collect run results; compute IQM + 95% stratified bootstrap CIs (Agarwal et al.); produce summary tables and training curve plots.
- **Input:** W&B run IDs, ablation config matrix
- **Output:** `docs/results/<experiment>.md` with IQM tables, Holm-Bonferroni corrected p-values, plots
- **Gate:** ≥5 seeds per ablation present; emits warning and requests more runs if not
- **Tools:** Bash, Read, Write
- **Skills:** `machine-learning-ops:mlops-engineer`, `business-analytics:data-storytelling`

### Reused Plugin Agents (with DreamPrice-specific system prompts)

| Agent | Base plugin/skill | DreamPrice responsibility |
|-------|-------------------|---------------------------|
| `data-pipeline-agent` | `python-development:python-pro` | `src/retail_world_model/data/` — loader, transforms, causal instruments, schemas |
| `world-model-agent` | `backend-development:backend-architect` + `llm-application-dev:ai-engineer` | `src/retail_world_model/models/` — RSSM, Mamba-2, encoder, decoder, reward ensemble |
| `training-agent` | `python-development:python-pro` | `src/retail_world_model/training/` — three-phase loop, losses, offline utils |
| `rl-agent` | `python-development:async-python-patterns` | `src/retail_world_model/inference/`, `applications/`, `envs/` |
| `api-agent` | `api-scaffolding:fastapi-pro` | `src/retail_world_model/api/` — FastAPI serve, dynamic batcher, SSE streaming |
| `frontend-agent` | `application-performance:frontend-developer` | React playground + Streamlit researcher dashboard |
| `test-reviewer-agent` | `pr-review-toolkit:pr-test-analyzer` + `pr-review-toolkit:silent-failure-hunter` | Runs at every track completion gate |

---

## 4. Skills Assigned Per Agent

| Agent | Skills |
|-------|--------|
| data-pipeline-agent | `python-development:python-project-structure`, `python-development:python-testing-patterns`, `python-development:python-error-handling`, `python-development:python-type-safety` |
| world-model-agent | `python-development:python-design-patterns`, `python-development:python-performance-optimization`, `python-development:python-type-safety` |
| training-agent | `python-development:python-observability`, `python-development:python-resilience`, `backend-development:tdd-orchestrator` |
| causal-estimator-agent | `python-development:python-error-handling`, `python-development:python-testing-patterns` |
| rl-agent | `python-development:async-python-patterns`, `python-development:python-resource-management` |
| api-agent | `api-scaffolding:fastapi-pro`, `backend-api-security:backend-security-coder`, `documentation-generation:openapi-spec-generation` |
| experiment-tracker-agent | `machine-learning-ops:mlops-engineer`, `business-analytics:data-storytelling` |
| frontend-agent | `ui-design:create-component`, `ui-design:responsive-design`, `ui-design:interaction-design` |
| test-reviewer-agent | `pr-review-toolkit:pr-test-analyzer`, `pr-review-toolkit:silent-failure-hunter`, `pr-review-toolkit:code-reviewer` |

---

## 5. Hook Mesh

All hooks defined in `.claude/plugins/dreamprice/hooks/`.

### Hook 1: `ruff-guard` (PostToolUse — Write, Edit)

```
Trigger:  Write or Edit on src/**/*.py or tests/**/*.py
Action:   ruff check <file> && ruff format --check <file>
Failure:  BLOCK — emit error with suggested fix, agent must correct before proceeding
```

### Hook 2: `pyright-guard` (PostToolUse — Write, Edit)

```
Trigger:  Write or Edit on src/**/*.py
Action:   pyright <file> --outputjson
Failure:  WARN during RED phase (tests being written), BLOCK during GREEN/REFACTOR
```

### Hook 3: `pytest-guard` (PostToolUse — Write, Edit)

```
Trigger:  Write or Edit in src/ or tests/
Action:   pytest tests/<matching_module>/ -x --tb=short -q
          (e.g., edit data/ → run tests/test_data/ only)
Failure:  BLOCK — emit failing test names, agent must fix before continuing
```

### Hook 4: `blueprint-path-guard` (PreToolUse — Write)

```
Trigger:  Write creating a new .py file under src/retail_world_model/
Action:   Check file path against module structure in docs/project-blueprint.md §12
Failure:  WARN (non-blocking) — "This path isn't in the blueprint. Confirm intentional."
          Agent must acknowledge before proceeding
```

---

## 6. Orchestration Flow Per Track

```
Orchestrator
  └─ conductor:implement (starts track)
       ├─ [RED phase]   → test-reviewer-agent reviews test coverage
       ├─ [GREEN phase] → dispatch parallel specialist agents
       │    ├─ agent-A (subtask-1)
       │    ├─ agent-B (subtask-2)   ← all fire simultaneously if independent
       │    └─ agent-C (subtask-3)
       │         └─ hooks fire on every Write/Edit (ruff, pyright, pytest)
       ├─ [REFACTOR phase] → code-simplifier runs on modified files
       └─ [GATE] → test-reviewer-agent runs full suite
            ├─ PASS → write docs/progress/<track>.md → advance to next track
            └─ FAIL → return to GREEN phase, fix, re-run gate
```

---

## 7. Key Constraints From Blueprint

- `torch`, `mamba-ssm`, `triton` are **optional dependencies** — never add to `install_requires`
- `CausalDemandDecoder` elasticity weights are **frozen** (from `causal-estimator-agent` output) — world-model-agent must not make them trainable
- Temporal split is **strictly chronological** — no shuffling across week boundaries
- Drop `PRICE_HEX` and `PROFIT_HEX` columns in all data loading code
- BF16 forward/backward, FP32 master weights; SSM state transitions in FP32
- Episode boundaries in Mamba-2: zero `conv_state` and `ssm_state` between store-year sequences

---

*Approved by user on 2026-02-23. Implementation plan follows.*
