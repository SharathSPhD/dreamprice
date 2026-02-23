# Tracks Registry — DreamPrice

| Status | Track ID | Title | Blocked By | Agent |
|--------|----------|-------|------------|-------|
| 🔵 Ready | track-1 | Data Pipeline | — | data-pipeline-agent |
| ⏳ Blocked | track-2 | World Model | track-1 | world-model-agent |
| ⏳ Blocked | track-3 | Training Loop | track-2 | training-agent |
| ⏳ Blocked | track-4 | Baselines | track-3 | rl-agent |
| ⏳ Blocked | track-5 | Ablations | track-4 | experiment-tracker-agent |
| ⏳ Blocked | track-6 | Paper & Packaging | track-5 | data-pipeline-agent |
| ⏳ Blocked | track-7 | Demo & Serving | track-3 | api-agent + frontend-agent |

## Track Completion Gate (all must pass before advancing)
1. `pytest tests/` — zero failures
2. `pyright src/` — zero errors
3. `ruff check src/` — zero violations
4. `docs/progress/<track-id>.md` written

## Track Dependency DAG
```
track-1 (data-pipeline)
    └─► track-2 (world-model)
            └─► track-3 (training-loop)
                    ├─► track-4 (baselines)
                    │       └─► track-5 (ablations)
                    │               └─► track-6 (paper-and-packaging)
                    └─► track-7 (demo-and-serving)  [parallel with track-6]
```
