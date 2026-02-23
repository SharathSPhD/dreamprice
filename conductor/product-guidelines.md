# Product Guidelines — DreamPrice

## Voice and Tone
Professional and technical. This is a research codebase targeting ML practitioners and economists. Code comments should be precise and cite the blueprint section or paper (e.g., `# DreamerV3 §C, KL balancing`). API docs should be concise and example-driven.

## Design Principles
1. **Blueprint fidelity** — every architectural decision must trace back to `docs/project-blueprint.md`. Do not invent architecture.
2. **Causal correctness over modeling convenience** — the frozen elasticity decoder is non-negotiable. Never make `theta` trainable.
3. **Offline safety** — always apply MOPO LCB pessimism during imagination. Never let the agent exploit model errors.
4. **Symlog everywhere** — apply to encoder inputs, decoder targets, reward targets, critic targets. Never use raw `log` on demand or price data.
5. **YAGNI** — the package is a research artifact, not a production system. No premature abstractions.

## Critical Do-Nots
- Do NOT add `torch`, `mamba-ssm`, or `triton` to `install_requires` in `pyproject.toml`
- Do NOT shuffle data across temporal boundaries — all splits are strictly chronological
- Do NOT use Gumbel-Softmax — use straight-through estimator only
- Do NOT use raw `log` — always use `symlog`
- Do NOT make `CausalDemandDecoder.theta` trainable
- Do NOT model competitors as agents — competition is latent state only
