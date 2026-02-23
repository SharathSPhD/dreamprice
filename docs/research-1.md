# Designing a learned world model for retail pricing: a complete research blueprint

**No one has built a general-purpose learned world model for retail price environments — and the tools to do so are ready.** After surveying 40+ papers, every existing RL pricing system uses hand-crafted simulators or model-free methods trained directly in real markets. Meanwhile, world model architectures like DreamerV3 and DRAMA have matured to the point where a solo researcher with a DGX Spark can train a 200M–1B parameter dynamics model in hours. The combination of rich public datasets (Dominick's 99M retail scanner records, EIA weekly fuel prices spanning 30+ years) with proven latent-imagination RL creates a clear, defensible, and novel research contribution. This report synthesizes eight research dimensions into an actionable design document.

---

## The world model landscape has converged on three viable small-scale architectures

The 2022–2025 period produced a Cambrian explosion of world model architectures, but only a handful are trainable on a single accelerator with hours-scale budgets. Three stand out for a non-visual, tabular pricing domain.

**DreamerV3** (Hafner et al., Nature 2025) remains the gold standard. Its Recurrent State-Space Model (RSSM) combines a deterministic GRU path with **32×32 discrete categorical stochastic latents**, trained end-to-end with symlog predictions, KL balancing, and percentile return normalization. It masters 150+ tasks with a single set of hyperparameters. The S/M configurations run at **12–50M parameters** on a single A100, and critically, it natively handles non-image inputs via MLP encoders — no CNN adaptation needed for tabular economic data.

**DRAMA** (Wang et al., ICLR 2025) replaces the transformer/GRU backbone with **Mamba-2 selective state-space layers**, achieving competitive Atari 100k performance with only **7–10M parameters** and O(n) memory complexity. **R2I** (Samsami et al., ICLR 2024 Oral) takes the complementary approach of replacing DreamerV3's GRU with S4/Mamba layers, delivering **9× computational speedup** while excelling at memory-intensive tasks. Both inherit DreamerV3's fixed-hyperparameter philosophy.

**TD-MPC2** (Hansen et al., ICLR 2024 Spotlight) offers an alternative paradigm: an implicit, decoder-free world model that predicts only rewards and Q-values in latent space, using MPPI trajectory optimization at decision time. It scales from **1M to 317M parameters** and has the most explicit cross-task transfer framework (XTRA). However, its planning-based inference is computationally expensive compared to amortized policies.

The large-scale models — GAIA-1/2/3 (9–15B params), UniSim, NVIDIA Cosmos, Genie 3 — require industrial compute clusters and are infeasible at DGX Spark scale. JEPA-based models (V-JEPA, V-JEPA 2) offer excellent representation learning but lack mature RL integration. IRIS (30M params, VQ-VAE + autoregressive transformer) and DIAMOND (4.4M params, diffusion U-Net) are designed for pixel observations and would require substantial adaptation for tabular data.

| Architecture | Params | Single GPU | Non-visual | Action-conditioned | Best for |
|---|---|---|---|---|---|
| DreamerV3 (RSSM) | 12–200M | ✅ | ✅ native MLP | ✅ | General-purpose, proven |
| DRAMA (Mamba-2) | 7–10M | ✅ | Visual RL tested | ✅ | Parameter-constrained |
| R2I (S4 + Dreamer) | ~DreamerV3 | ✅ | ✅ | ✅ | Long-range memory |
| TD-MPC2 | 1–317M | ✅ | ✅ | ✅ | Continuous control |
| IRIS/DIAMOND | 4–30M | ✅ (Atari) | ❌ pixel-based | ✅ | Visual fidelity |

---

## The gap is real: zero learned world models exist for economic environments

Extensive search across academic databases, open-source repositories, and industry publications confirms the hypothesis. **Every RL pricing system in the literature uses either hand-crafted analytical simulators or model-free methods.** The pattern is universal:

Liu et al.'s Alibaba field experiment (2019) uses DRL with a hand-coded demand model. Kastius & Schlosser (2022) train DQN and SAC in analytically specified duopoly markets. The 2024 deep Q-learning approach for retail pricing constructs a simulator with explicit base demand and price elasticity parameters. The 2025 survey of RL applications in dynamic pricing (covering Airbnb, Uber, Amazon, fashion retail) finds **no mention of world models** across the entire literature.

The closest analogue is Salesforce's **AI Economist** (2020–2023, Science Advances), a multi-agent economic simulation for tax policy optimization. But its "Foundation" framework is entirely rule-based — a hand-coded simulator, not a learned dynamics model. ABIDES-Gym offers an agent-based financial market simulator with a Gymnasium interface, but again uses hand-coded agent models for dynamics.

**No Gymnasium-compatible environment exists for retail pricing or demand dynamics.** The Farama Foundation's third-party environment registry includes trading environments (gym-anytrading, gym-trading-env) for financial buy/sell decisions, but nothing for the qualitatively different problem of setting prices to influence demand.

This gap exists across five dimensions: no learned world model for retail dynamics, no Dyna-style imagination training for pricing agents, no standardized RL environment for pricing, no world model application to economic time series, and no competitive pricing sandbox where agents can train inside a learned model.

---

## Dominick's and EIA fuel data are the highest-quality training sources

**Tier 1: Primary training data.** The **Dominick's Finer Foods dataset** (Kilts Center, University of Chicago) is the strongest candidate. It contains **~99 million observations** of weekly store-level scanner data across 93 stores, ~13,845 products in 29 categories, spanning 399 weeks (1989–1997). Crucially, it captures the action→outcome relationship directly: retail prices, unit sales, profit margins, deal codes (promotions), customer counts, and store demographics. It is freely available for academic research. The **EIA Weekly Retail Gasoline & Diesel Prices** provide 30+ years of weekly fuel price data across 5 PAD districts, ~30 states, and ~10 cities, available via open API. Paired with FRED's crude oil prices (DCOILWTICO), CPI, consumer sentiment, and unemployment data, this creates a rich multivariate environment for fuel pricing dynamics.

**Tier 2: High-value but access-constrained.** The **NielsenIQ Retail Scanner Data** (via Kilts Center) is the gold standard with billions of annual records across 35,000+ stores and 90+ retail chains, but requires institutional subscription. The **UK Fuel Finder Scheme** (launched February 2, 2026) provides near real-time station-level prices for all ~8,400 UK petrol stations, updated within 30 minutes of price changes — unprecedented granularity for competitive pricing dynamics, though with minimal historical depth.

**Tier 3: Essential exogenous context.** **FRED** (Federal Reserve Economic Data) provides 840,000+ time series including all major macroeconomic indicators. The **European Commission Oil Bulletin** adds weekly country-level fuel prices across EU member states since 2005.

**Tier 4: Limited utility.** Instacart's dataset (3.4M orders, 49K products) contains no price data. Kaggle pricing datasets are generally too small, often synthetic, and lack temporal depth. No bulk Amazon pricing dataset exists publicly.

For the world model, Dominick's is ideal because it directly contains the state-action-outcome tuples needed: the "action" is the price/promotion decision, the "state" includes competitor context, seasonality, and store demographics, and the "outcome" is observed demand. EIA fuel data requires pairing with consumption/sales reports to complete the causal loop but offers superior temporal depth.

---

## The recommended architecture: DreamerV3's RSSM with a Mamba backbone

**A critical hardware note first.** The DGX Spark's marketed "1 PFLOP" figure refers to FP4 sparse theoretical peak. Measured BF16 performance is approximately **100 TFLOPS**, with memory bandwidth of **273 GB/s** (3× less than Mac Studio M3 Ultra). All estimates below use the realistic ~100 TFLOPS figure. The 128GB unified LPDDR5x memory is the genuine advantage — it enables larger batch sizes and model sizes than typical consumer GPUs.

The optimal architecture is a **hybrid DreamerV3 + Mamba-2 design** following the R2I pattern:

1. **Encoder**: MLP mapping the economic state vector (own prices, competitor prices, demand signals, inventory, calendar embeddings, macro indicators — ~100–300 dimensions) to latent embeddings
2. **Dynamics backbone**: Mamba-2 selective state-space layers replacing DreamerV3's GRU, providing O(n) complexity for processing long price histories and capturing seasonal/cyclical patterns
3. **Stochastic latent**: DreamerV3's 32×32 discrete categorical variables, capturing demand uncertainty, competitor randomness, and market noise
4. **Decoder heads**: MLP decoders for observation reconstruction, reward prediction, and episode continuation
5. **Actor-critic**: Trained entirely in latent imagination using DreamerV3's proven techniques (symlog, percentile return normalization, KL balancing)

**Model sizing**: A 200M–500M parameter model is the sweet spot. The Mamba backbone at 200–400M, with encoder/decoder/heads at 50–100M. In BF16, this requires **~4–8GB for weights** and **~25–50GB total with optimizer states and activations** — well within 128GB. Training time on Dominick's data (99M records): estimated **4–8 hours** for the world model, **1–2 hours** for RL agent training in imagination. A full experimental pipeline including ablations is feasible within **one week**.

**Why not the alternatives?** A Chronos-style tokenized LLM loses price precision through bin quantization (critical for fuel pricing where pennies matter) and isn't natively action-conditioned. IRIS/DIAMOND require VQ-VAE adaptation for tabular data with questionable benefit for low-dimensional states. Vanilla DreamerV3 works but its GRU limits long-range memory for seasonal patterns. Pure TD-MPC2 is excellent for continuous control but its planning-based inference adds latency for real-time demos.

**Embedding strategy for economic features:**

- **Continuous features** (prices, demand, revenue): Symlog transform → linear projection. Symlog handles the wide dynamic range between cent-level price changes and million-dollar revenue figures
- **Categorical features** (day-of-week, region, product category): Learned embedding lookup tables
- **Calendar/cyclical features**: Sinusoidal positional encoding for time-of-year and day-of-week cycles
- **Competitor prices**: Separate channels in the observation vector, enabling the model to learn cross-price elasticities
- **Exogenous macro variables** (CPI, crude oil, unemployment): Concatenated as additional continuous features after symlog transform

---

## Training RL agents inside the learned model follows a proven recipe

DreamerV3's latent imagination training is the recommended approach. The loop operates as follows: encode a real observation from the replay buffer into latent state (h_t, z_t), then unroll the dynamics model for **15–30 imagined steps** using actions sampled from the current policy. The actor is trained via REINFORCE with λ-returns as baseline; the critic learns λ-returns from imagined rewards. Crucially, **gradients do not backpropagate through the world model** during actor-critic updates — the model is treated as a frozen environment.

For a pricing agent, the **action space** should be continuous price adjustments bounded to ±10% per period (or discrete price levels from a grid for fuel: $2.29, $2.39, $2.49...). The **state** includes own prices, competitor prices, demand signals, inventory, calendar features, and macro context. The **reward function** combines gross profit (primary), volume maintenance (secondary), price-change penalties (stability), and customer retention proxies, with configurable weights enabling different business strategies.

Three critical pitfalls must be addressed:

**Compounding model errors** accumulate when multi-step predictions bootstrap successive outputs. A 0.5% per-step demand prediction error compounds to ~50% over 100 steps. The solution is short rollouts branched from real replay buffer states (MBPO's approach), combined with terminal value functions that bootstrap long-horizon returns. For pricing, 15–30 step rollouts (representing 15–30 pricing periods) provide meaningful policy optimization before errors dominate.

**Reward hacking** occurs when the RL policy exploits model inaccuracies to find spuriously high-reward states. An ensemble of 3–5 world models with **pessimistic reward** (using the lower confidence bound of ensemble predictions) keeps agents in well-modeled regions. Constraining the action space to business-feasible ranges and adding explicit margin floors provides additional guardrails.

**Distribution shift** emerges as the improving policy visits states underrepresented in training data. Ensemble disagreement provides an epistemic uncertainty signal; branching rollouts from real states (rather than from model-generated states) keeps imagination near the data distribution.

---

## The repo should be the first pip-installable world model codebase

A survey of DreamerV3, IRIS, DIAMOND, and TD-MPC2 repositories reveals a striking gap: **none are pip-installable, none use pyproject.toml, none have test suites, and only TD-MPC2 has a CONTRIBUTING.md.** All use Hydra for configuration and W&B for logging. DIAMOND and IRIS offer interactive play modes. TD-MPC2 has the best community design with Docker support and explicit contribution guidelines.

The recommended structure follows modern Python packaging with a `src/` layout:

```
price-world-model/
├── pyproject.toml              # PEP 621, hatchling backend
├── CONTRIBUTING.md
├── configs/                    # Hydra config hierarchy
│   ├── world_model/           # transformer.yaml, mamba.yaml
│   ├── agent/                 # dreamer.yaml, ppo.yaml, sac.yaml
│   ├── environment/           # fuel_pricing.yaml, grocery.yaml
│   └── experiment/            # Full reproducible configs
├── src/price_world_model/     # pip-installable package
│   ├── world_model/           # base.py + implementations
│   ├── agent/                 # base.py + RL algorithms
│   ├── envs/                  # Gymnasium-compatible environments
│   ├── data/                  # datasets, replay buffer, transforms
│   ├── training/              # trainer, evaluator, callbacks
│   └── visualization/         # plots, dashboard, interactive play
├── scripts/                    # train.py, evaluate.py, play.py
├── tests/                      # pytest suite (unprecedented in this space)
├── notebooks/                  # quickstart, custom env, analysis
└── docker/
```

The package should expose a clean public API: `from price_world_model import WorldModel, RLAgent, PricingEnvironment, Trainer`. Environments follow Gymnasium's `reset()`/`step()` interface. World models add `predict()`, `rollout()`, and `imagine()` methods. Model checkpoints go on HuggingFace Hub, following the pattern established by DIAMOND, IRIS, and TD-MPC2.

A **REST API** (FastAPI) wraps the core Python API for web integration: `/api/v1/environment/step`, `/api/v1/world_model/rollout`, `/api/v1/agent/recommend`. WebSocket streaming enables real-time training metrics to the frontend.

---

## The killer demo: a React playground where users race against a pricing agent

The user-facing app should be a **React-based RL playground** (buildable with Lovable) showing an agent learning to price fuel competitively in real time. The demo flow: (1) show the trained world model simulating realistic market dynamics, (2) drop in an untrained RL agent — watch it explore randomly and lose money, (3) fast-forward imagination training — the agent discovers pricing strategies in minutes, (4) deploy the trained agent and watch it react to competitor price changes, demand shifts, and supply disruptions, (5) let the user play as a competitor in adversarial mode.

Key visual elements: a real-time price chart showing agent vs. competitor prices, animated market share donuts, reward curves with confidence bands, a policy heatmap showing what price the agent would set given different (demand, competitor_price) states, and a world model accuracy panel comparing predicted vs. actual next states. DIAMOND's interactive play mode — where users can toggle between the real environment and the world model's imagination with a single keypress — is the gold standard to emulate.

A separate **Streamlit training dashboard** serves researchers with live training curves, rollout visualization, hyperparameter panels, and W&B integration.

---

## Position the preprint as "DreamPrice" — first world model for economic environments

The paper should be framed as a **"new application domain + open-source benchmark"** contribution, not a new architecture paper. The core framing: world models have transformed RL in games, robotics, and autonomous driving, but economic environments — among the most commercially important sequential decision-making domains — remain entirely unmodeled.

The related work maps cleanly onto a 2×2 matrix that visualizes the gap:

|  | Learned model | Hand-crafted model |
|--|---|---|
| Physical domain | DreamerV1–V3, IRIS, TD-MPC2, Cosmos, Genie | MuJoCo, Isaac Gym, CARLA |
| Economic domain | **DreamPrice (this work)** | ABIDES, DSGE models, AI Economist |

The related work section should cover five lineages: (1) world models and model-based RL (Ding et al.'s 2025 ACM Computing Surveys article, Moerland et al.'s MBRL survey), (2) RL for dynamic pricing (Liu et al.'s Alibaba experiment, Kastius & Schlosser), (3) time series foundation models (Chronos, TimesFM — contrasting prediction vs. counterfactual reasoning), (4) digital twins and learned simulators (ABIDES-Gym, Reinforcement Twinning), and (5) the Dreamer lineage specifically (V1/V2/V3, R2I, DRAMA).

**Reviewers will expect these baselines:** model-free RL (PPO, SAC, DQN), heuristic pricing rules (fixed markup, competitive matching), time series forecasting + optimization (ARIMA/Prophet → price optimizer), simpler world models (linear dynamics, MLP ensemble), and an oracle baseline with ground-truth simulator access. **Required ablations** include architecture variants, latent dimension sizes (32/64/128/256), training data volume (10%/25%/50%/100%), rollout horizon length (1/5/10/25/50 steps), and stochastic vs. deterministic dynamics.

**Title recommendation: "DreamPrice: A Learned World Model for Retail Pricing Environments."** This connects to the Dreamer lineage, is memorable, clearly scopes the domain, and avoids overclaiming. The contribution triple is defensible: (1) first learned world model for pricing, (2) Gymnasium-compatible environment release, (3) empirical analysis showing sample-efficient policy learning vs. model-free baselines.

## Execution roadmap: from zero to arXiv in 13 weeks

| Weeks | Milestone |
|---|---|
| 1–2 | Implement RSSM + Mamba world model for tabular economic states |
| 3–4 | Train on Dominick's/EIA data, evaluate prediction fidelity |
| 5–6 | Build Gymnasium wrapper, implement DreamerV3-style actor-critic |
| 7–8 | Run all baselines (model-free RL, heuristics, time series) |
| 9–10 | Ablation studies across architecture, data volume, rollout horizon |
| 11–12 | Write paper, build React demo, polish code for open-source release |
| 13 | Submit to arXiv + target ICML/NeurIPS 2026 workshop |

All experiments are feasible on a DGX Spark within this timeline. The structured economic state space (100–300 dimensions) is fundamentally more tractable than the 64×64×3 image observations that DreamerV3 handles on a single A100. The 128GB unified memory accommodates full-parameter training of models up to ~3B. The key constraint is memory bandwidth (273 GB/s), not compute — batch size tuning and mixed-precision training will be essential optimizations. The honest framing of compute budget (including wall-clock times) will strengthen rather than weaken the paper, following DreamerV3's own precedent of emphasizing single-GPU accessibility.