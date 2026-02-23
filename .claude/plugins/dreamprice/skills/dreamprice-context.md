# DreamPrice Context Skill

This skill provides shared architectural context for all DreamPrice agent team members.

## What You Are Building

DreamPrice is a **Mamba-2-backed RSSM world model** trained offline on the Dominick's Finer Foods dataset (1989–1997) for retail pricing optimization. It is the first learned world model for economic environments.

## Non-Negotiable Architectural Rules

1. **Frozen elasticity decoder** — `CausalDemandDecoder.theta` weights come from DML-PLIV via the causal-estimator agent. `requires_grad = False`. Never override.

2. **DRAMA-style decoupled posterior** — `z_t = encode(x_t)` only. The posterior does NOT receive `h_t` during training. This enables Mamba-2's parallel SSD scan.

3. **h_t definition** — `h_t` is the OUTPUT of the Mamba-2 block (shape `B × d_model`), NOT the internal SSM state. Internal SSM state is managed by `step()`.

4. **Symlog everywhere** — `symlog(x) = sign(x) * log(|x| + 1)`. Apply to: encoder inputs, decoder targets, reward targets, critic targets. NEVER use raw `log` on retail data.

5. **Straight-through gradients** — `z = one_hot + probs - probs.detach()`. NO Gumbel-Softmax.

6. **Unimix on ALL categoricals** — `probs = 0.99 * softmax(logits) + 0.01/32`

7. **KL balancing** — `beta_pred=1.0, beta_dyn=0.5 (sg posterior), beta_rep=0.1 (sg prior), free_bits=1.0`

8. **MOPO LCB** — `r_pessimistic = r_mean - lambda_lcb * r_std` from 5-head reward ensemble. Modify reward signal only, never critic training.

9. **Temporal split** — STRICTLY chronological. Train weeks 1–280, Val 281–340, Test 341–400. No shuffling across time boundaries.

10. **Optional CUDA deps** — `torch`, `mamba-ssm`, `triton` are NOT in `install_requires`. They are optional dependencies requiring manual CUDA-matched installation.

## Key File Locations

- Blueprint: `docs/project-blueprint.md` (authoritative specification)
- Elasticity configs: `configs/elasticities/<category>.json`
- Raw data: `docs/data/category/w<cat>.csv` (movement), `docs/data/category/upc<cat>.csv` (UPC metadata)
- Package: `src/retail_world_model/`
- Tests: `tests/`
- Configs: `configs/` (Hydra YAML)
- Progress logs: `docs/progress/<track-name>.md`

## Mamba-2 Usage Pattern

```python
from mamba_ssm import Mamba2
from mamba_ssm.utils.generation import InferenceParams

mamba = Mamba2(d_model=512, d_state=64, d_conv=4, expand=2, headdim=64, chunk_size=256)

# Training (parallel SSD scan):
h = mamba(x)  # x: (B, T, 512) → h: (B, T, 512)

# Imagination (recurrent step):
params = InferenceParams(max_seqlen=1, max_batch_size=B)
h_t = mamba(x_t.unsqueeze(1), inference_params=params).squeeze(1)
```

## Reward Function

```python
reward = (
    ((price - cost) * units_sold).sum(-1)          # gross margin
    - 0.05 * (price - prev_price).abs().sum(-1)    # volatility penalty
    - F.relu(0.10 - (price - cost) / price.clamp(min=0.01)).sum(-1)  # margin floor
)
```

## Episode Boundary Handling in Mamba-2
Zero out `conv_state` and `ssm_state` between store-year boundaries during inference. During training, ensure each sequence (T=64) contains only one store-year.

## Hyperparameter Reference
- Batch B=32, Sequence T=64, Horizon H=13, gamma=0.95, lambda=0.95
- WM lr=1e-4, Actor lr=3e-5, Critic lr=3e-5
- Grad clip: WM=1000, Actor/Critic=100
- Twohot: 255 bins in symlog [-20, +20]
- Critic EMA decay=0.98, Return norm EMA=0.99
