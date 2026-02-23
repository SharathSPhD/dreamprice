# Track 6: Paper & Packaging

## Summary

Finalized the DreamPrice project for public release with complete documentation,
notebooks, and HuggingFace Hub integration.

## Deliverables

### Public API (`src/retail_world_model/__init__.py`)
- Exports: `WorldModel`, `RLAgent`, `PricingEnvironment`, `Trainer`
- Version: 0.1.0
- Clean re-exports with user-friendly aliases

### `pyproject.toml`
- Added keywords, classifiers, project URLs
- Added `docs` optional dependency group (jupyter, nbformat, matplotlib, seaborn)
- Added CLI entry points: `dreamprice-train`, `dreamprice-serve`
- License formatted as SPDX table

### `MODEL_CARD.md`
- HuggingFace-compatible frontmatter (license, tags, datasets)
- Architecture summary, training data, intended use, limitations
- BibTeX citation block

### `README.md`
- Project description, quick install, quick start code
- ASCII architecture diagram
- Dataset description, training/serving instructions
- License and citation

### Jupyter Notebooks
- `notebooks/quickstart.ipynb` (6 cells): environment creation, rollout, visualization
- `notebooks/custom_env.ipynb` (5 cells): custom SKUs, action space, SB3 baseline, evaluation
- `notebooks/causal_analysis.ipynb` (6 cells): Hausman IV, first-stage F-stat, DML-PLIV,
  elasticity distribution, OLS vs IV endogeneity bias comparison

### `scripts/push_to_hub.py`
- Uploads checkpoint + MODEL_CARD.md to HuggingFace Hub
- Reads token from .env or CLI argument
- Creates repo if not exists

## Status
Complete.
