# Enriched retail pricing world model: corrected implementation plan

The review document identifies three genuine viability gaps — endogeneity, offline pessimism, and competitive modeling — but overclaims solutions for two of them. This corrected plan integrates valid contributions, pushes back on infeasible proposals, and provides exact implementation recipes across all seven dimensions. **The most impactful corrections are: Hausman IV (not 2sCOPE) as the primary endogeneity fix, MOPO-style LCB (not COMBO) for offline pessimism, and implicit latent competition (not Stackelberg follower networks) for adversarial awareness.**

The Dominick's dataset provides ~581K store-week-SKU tuples across 83 stores over 280 weeks. The PROFIT field enables wholesale cost derivation, and the panel structure supports Hausman cross-store instruments — making causal identification feasible without external data. The review's claims about Stackelberg POMDPs and Mean Field Games are overclaims given the absence of competitor price data; the honest approach is competition-conditioned latent dynamics with robust imagination.

---

## Dimension 1: endogeneity correction requires Hausman IV, not 2sCOPE alone

**The review overstates 2sCOPE's role.** The empirical IO literature overwhelmingly uses Hausman instruments as the workhorse for scanner data endogeneity (Nevo 2001, Berry & Haile 2021). 2sCOPE trades a testable exclusion restriction for an untestable Gaussian copula assumption — calling it a "perfect" solution is misleading.

**What Dominick's actually provides.** The PROFIT field encodes gross margin percentage, yielding wholesale cost via `wholesale_cost = PRICE × (1 − PROFIT/100)`. However, the Kilts Center manual explicitly warns this is **Average Acquisition Cost (AAC)**, not replacement cost. AAC updates sluggishly as old inventory sells off and can drop precipitously from forward-buying. This makes it a noisy instrument — relevant (drives retail price) but partially endogenous (AAC reflects past demand through inventory dynamics). It should serve as a supplementary instrument, not the sole one.

**The Hausman instrument is the correct primary approach.** For each (UPC j, store s, week t) observation, construct: `Z_{j,s,t} = mean(log_price_{j,s',t})` for all s' ≠ s. This exploits common wholesale cost shocks propagating across Dominick's 16 pricing zones while assuming local demand shocks are independent across stores after controlling for fixed effects. With 83 stores, the first-stage F-statistic will be **well above 100** — far exceeding the weak instrument threshold.

The exclusion restriction requires that after conditioning on store, product, and time fixed effects, demand shocks at store s are uncorrelated with prices at store s'. The main threat is chain-wide promotional campaigns creating correlated demand shocks. Mitigation: include week × category fixed effects and cluster standard errors by week following Hahn, Liao, and Shi (2024).

**2sCOPE serves as a robustness check, not the primary method.** The Park & Gupta (2012) copula correction transforms price to a standard normal quantile — `P* = Φ⁻¹(rank(P)/(n+1))` — and appends it as a control variable. The 2sCOPE refinement (Yang, Qian, Xie 2024, JMR) residualizes P* on copula-transformed controls before inclusion, handling P-W correlation. Identification requires **non-normality of the endogenous regressor**; retail prices with promotional spikes typically satisfy this. But if all regressors happen to be approximately normal, identification collapses entirely. The Gaussian copula dependence assumption is also untestable.

**DML+IV is the most principled combined approach** for constraining the world model. Chernozhukov et al. (2018, arXiv 1608.00060) provide the Partially Linear IV Model (PLIV): Y = θ₀D + g₀(X) + U, where D = log(price), Y = log(demand), X = controls, and E[U|X,Z] = 0 with Z = Hausman IV. Cross-fitted residuals from flexible ML nuisance models (random forests for both price and demand on controls) yield the causal elasticity θ₀ with √n-consistent inference. The Python `doubleml` package implements this directly via `DoubleMLPLIV`.

**Exact implementation recipe:**

```python
# Step 1: Construct instruments
df['log_price'] = np.log(df['PRICE'] / df['QTY'])
df['log_move'] = np.log(df['MOVE'].clip(lower=1))
df['log_cost'] = np.log((df['PRICE'] * (1 - df['PROFIT']/100)).clip(lower=0.01))

group_sum = df.groupby(['UPC','WEEK'])['log_price'].transform('sum')
n_stores = df.groupby(['UPC','WEEK'])['log_price'].transform('count')
df['hausman_iv'] = (group_sum - df['log_price']) / (n_stores - 1)

# Step 2: 2sCOPE copula residual (robustness check)
from scipy.stats import norm, rankdata
df['price_star'] = norm.ppf(rankdata(df['log_price']) / (len(df) + 1))
exog_cols = ['income','educ','age60','ethnic','workwom','hsizeavg']
W_stars = np.column_stack([norm.ppf(rankdata(df[c])/(len(df)+1)) for c in exog_cols])
reg1 = LinearRegression().fit(W_stars, df['price_star'].values)
df['copula_resid'] = df['price_star'].values - reg1.predict(W_stars)

# Step 3: DML-PLIV for causal elasticity bound
import doubleml as dml
data_dml = dml.DoubleMLData(df, y_col='log_move', d_cols='log_price',
                             x_cols=control_cols, z_cols='hausman_iv')
dml_pliv = dml.DoubleMLPLIV(data_dml, ml_l=RandomForestRegressor(),
                             ml_m=RandomForestRegressor(),
                             ml_r=RandomForestRegressor(), n_folds=5)
dml_pliv.fit()
theta_causal = dml_pliv.coef_[0]  # e.g., -2.2 to -2.5 for typical grocery
```

**How the causal correction enters the world model — three strategies ranked by rigor:**

**(A) Constrained decoder (most principled, recommended):** Pre-estimate θ̂ via DML-PLIV. The demand decoder separates the causal price effect from latent-state-driven residual demand:

```python
class CausalDemandDecoder(nn.Module):
    def __init__(self, theta_causal, latent_dim):
        self.theta = theta_causal  # FIXED from DML-PLIV, e.g., -2.3
        self.residual_net = MLP(latent_dim, hidden=256, out=1)
    def forward(self, z_t, log_price_t):
        return self.theta * log_price_t + self.residual_net(z_t)
```

**(B) Gradient regularization (flexible):** Allow the decoder to learn its own elasticity but penalize deviation from the DML estimate:

```python
D.requires_grad_(True)
Y_pred = decoder(z_t, D)
grad_D = torch.autograd.grad(Y_pred.sum(), D, create_graph=True)[0]
causal_reg = lambda_c * ((grad_D - theta_causal) ** 2).mean()
```

**(C) Observation enrichment (simplest start):** Append `hausman_iv` and `copula_resid` directly to the observation vector. The RSSM posterior informed by exogenous price variation helps disentangle causal from confounded relationships. This is theoretically weakest — the model may still prefer the confounded signal — but requires minimal architectural changes.

**Validation checklist:** First-stage F > 10 (expect > 100); Hausman test rejects OLS = IV; 2sCOPE β_c significant; Sargan overidentification test (using both Hausman IV and wholesale cost); compare world model counterfactual predictions across approaches.

---

## Dimension 2: MOPO-style LCB is the right pessimism, not COMBO

**The review proposes the correct principle but the wrong mechanism.** COMBO (Yu et al., NeurIPS 2021, arXiv 2102.08363) adds CQL-style regularization to model-generated rollouts: it pushes down Q-values on model-rollout state-actions while pushing up Q-values on real dataset state-actions, with a mixing fraction f (typically 0.05) between real and synthetic data in critic batches. The exact loss is:

```
L_critic = β·[E_{model-rollout}[Q(s,a)] − E_{dataset}[Q(s,a)]]
           + ½·E_{mixed}[(Q(s,a) − B̂πQ̂(s,a))²]
```

This is elegant but requires significant modifications to DreamerV3's critic training loop — mixing real transitions with imagined ones, implementing the CQL push-down/push-up terms, and tuning β ∈ {0.5, 1.0, 5.0}. For a solo researcher, this is error-prone engineering.

**MOPO's reward penalty is simpler and integrates trivially with DreamerV3.** The penalized reward `r̃_t = r̂_t − λ · u(s_t, a_t)` requires modifying only the reward signal during imagination — no changes to critic or actor architecture. The uncertainty u comes from an ensemble of reward heads.

**MC dropout is unreliable with Mamba-2.** SSM architectures maintain recurrent hidden states that are sensitive to perturbation. The MambaMIR paper (2024) confirms that standard MC dropout degrades Mamba performance and proposes Arbitrary Scan Masking as an alternative. Additionally, general literature shows MC dropout is "not a reliable approach for out-of-distribution data" — precisely the regime that matters for offline RL.

**The recommended approach: 5 independent reward heads on a shared RSSM backbone.**

```python
# During imagination (Phase 2-3 of DreamerV3):
for t in range(H):
    z_t = rssm.imagine(h_t, z_{t-1}, a_{t-1})
    r_preds = [reward_head_k(z_t, a_t) for k in range(5)]
    r_mean = torch.stack(r_preds).mean(dim=0)
    r_std = torch.stack(r_preds).std(dim=0)
    r_pessimistic = r_mean - lambda_lcb * r_std  # LCB
```

**Calibrating λ.** Normalize rewards to zero mean and unit variance using offline dataset statistics. Start with **λ = 1.0** (one standard deviation of pessimism). For gross margins averaging ~$2.50 with ensemble std ~$0.50 on in-distribution data, λ = 1.0 means a ~20% penalty — reasonable conservatism. Search λ ∈ {0.25, 0.5, 1.0, 2.0, 5.0} and validate via offline policy evaluation on held-out data. Lu et al. (ICLR 2022) showed that normalization-based λ setting with a single value (λ = 1.0) often outperforms per-environment grid search.

**MOReL's USAD detector is inappropriate** for this setting. Its hard binary partition between "known" and "unknown" state-actions, with an absorbing HALT state carrying penalty −κ, creates discontinuities in the value function. For continuous economic data with smooth price-demand relationships, soft penalties (MOPO-style) are strictly preferable.

**EDAC and SAC-N require explicit Q-network ensembles**, which DreamerV3 does not use — it trains value networks on imagined trajectories. Adapting gradient diversification to DreamerV3's value head would be non-trivial. Skip these for this project.

---

## Dimension 3: the Stackelberg POMDP proposal is an overclaim — here is what actually works

**The Stackelberg POMDP paper (arXiv 2210.03852) addresses mechanism design, not retail pricing.** Brero et al. (2024) model a leader who designs rules (auction mechanisms, allocation schemes) while followers are no-regret learners converging to Bayesian Coarse-Correlated Equilibrium over E×T interaction steps. Critically, it assumes **follower actions are fully observed within the game** and requires simulating follower learning dynamics during training. Neither condition holds for Dominick's.

**The "auxiliary follower policy network" is infeasible.** Training such a network requires time-varying competitor actions to fit against, a training signal from observed competitor behavior, and a state space that evolves temporally. Dominick's has exactly four static competition variables per store: SSTRDIST (distance to nearest competitor), SSTRVOL (nearest competitor's relative volume), CPDIST5 (average distance to 5 nearest supermarkets), and CPWVOL5 (competitor weekly volume in 5-mile radius). These are computed from census data and **do not vary across the 280 weeks**. There is nothing to train a follower policy on.

**Mean Field Game dynamics are equally infeasible.** MFG requires tracking a population distribution μ_t that evolves via a Fokker-Planck equation, with a Hamilton-Jacobi-Bellman backward pass for optimal control, iterated to a fixed point. With 4 static variables representing competition, there is no "mean field" to model. Even the simplest MFG+RL implementations (Carmona et al.) with |S|=|A|=10 require significant computation — scaling to the Dominick's state space with a Mamba-2 RSSM far exceeds a solo researcher's hours-scale budget.

**CTDE is irrelevant.** This is a single-agent pricing problem. Competitors are part of the environment, not controllable teammates. Invoking CTDE imports unnecessary multi-agent machinery for what amounts to standard actor-critic with privileged critic information.

**LOLA (Foerster et al. 2018) requires differentiable access to the opponent's learning rule.** DRON (He et al. 2016) requires time-varying opponent observations. SOM (Raileanu et al. 2018) requires observing opponent actions at each timestep. None of these apply to Dominick's.

| Proposed element | Verdict | Reason |
|---|---|---|
| Auxiliary follower policy | **Overclaim** | No competitor action data |
| Stackelberg POMDP | **Overclaim** | Mechanism design paper, not retail pricing |
| Mean Field Game dynamics | **Overclaim** | No observable population distribution; extreme complexity |
| CTDE | **Overclaim** | Single-agent problem |
| Latent competitive regime in RSSM | **Valid** | Natural extension of stochastic latent state |
| Demographics-conditioned dynamics | **Valid** | Data-supported, straightforward |
| Pessimistic/robust imagination | **Valid** | Grounded in robust MDP theory |

**The corrected, implementable approach: competition-conditioned RSSM with robust imagination.**

The RSSM's stochastic latent z_t already captures unobserved factors including competitor behavior. When a rival runs a promotion, Dominick's observes unexplained demand drops; the posterior infers a latent state consistent with that observation. Static demographics condition the prior, so stores with closer/larger competitors learn higher-variance dynamics:

```python
# Prior conditioned on competition intensity
competitive_alpha = sigmoid(
    w1 * (1/SSTRDIST) + w2 * SSTRVOL +
    w3 * (1/CPDIST5) + w4 * CPWVOL5
)

# During imagination: occasionally sample from pessimistic prior
# Higher competition → more pessimistic imagination
z_prior = world_model.prior(h_t)
if competitive_robust:
    z_pessimistic = shift_toward_low_reward_states(z_prior, reward_model)
    z_t = (1 - competitive_alpha) * z_prior + competitive_alpha * z_pessimistic

# Policy robustness: penalize high value variance across latent samples
value_samples = [critic(h_t, z_sample) for z_sample in z_samples]
robust_loss = actor_loss + lambda_robust * var(value_samples)
```

This adds ~30 lines of code, requires no additional data, and produces pricing policies robust to competitive surprises without pretending to model competitor strategies that aren't observable.

---

## Dimension 4: DRAMA and Mamba-2 claims need significant correction

**The RetNet/GLA parallel is technically accurate but misattributed.** DRAMA (arXiv 2410.08893, Wang et al. 2025) replaces DreamerV3's GRU with Mamba-2 blocks and uses a decoupled posterior `q(z_t | o_t)` without dependence on h_t to enable parallel training. The chunkwise parallel computation paradigm — parallel within chunks, recurrent across chunks — is shared by Mamba-2's SSD algorithm, RetNet, and GLA. But this is a property of **Mamba-2**, not something DRAMA introduces. The Flash Linear Attention library implements RetNet, GLA, and Mamba-2 under the same chunkwise framework. The key distinction: RetNet uses fixed exponential decay, GLA uses data-dependent diagonal gates, and Mamba-2/SSD uses data-dependent scalar gates per head.

**The 18× speedup claim is misleading at T=64.** Mamba-2's SSD crosses over FlashAttention-2 at sequence length **~2K** (Mamba-1 only at ~16K). At T=64, attention cost is 64² = 4,096 multiply-adds per head — trivial for modern GPUs. With FlashAttention-2, a Transformer at T=64 is extremely fast. At T=64, Mamba-2 provides **no speedup — it may even be slower** due to kernel launch overhead for the 5 sequential GPU kernels in the SSD algorithm. Use Mamba-2 for its **modeling properties** (recurrent inference during imagination, state representation for non-stationary dynamics) rather than speed.

**Episode boundary handling requires custom work.** The mamba-ssm library only exposes `conv_state` and `ssm_state` during step-by-step inference via `InferenceParams`/`MambaCache`. During training (parallel scan), there is no built-in episode boundary mechanism. R2I (Samsami et al., ICLR 2024) implemented a custom modified parallel scan operator that resets hidden states at episode boundaries. For this project's store-year boundaries during training: either (a) ensure each training sequence contains only one store-year (simplest, if sequences fit in T=64), (b) implement a custom scan with continuation flags that gate the state: `ssm_state *= (1 - done_flag)`, or (c) adopt R2I's modified scan operator. During inference, zero out caches between store-year boundaries.

**Unimix applies to ALL categorical distributions, including both prior and posterior.** From the DreamerV3 `tools.py`:

```python
class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits)
```

This is applied to the categorical latent z_t (both prior and posterior), discrete action distributions, and **indirectly affects KL computation** since both distributions are mixed before KL[q_mixed ∥ p_mixed] is computed. It is not applied to continuous distributions (rewards, values).

**R2I is better validated for tabular data than DRAMA.** R2I was explicitly tested on BSuite and POPGym (tabular environments) with strong results, achieving superhuman performance on memory-intensive tasks. DRAMA was evaluated only on Atari (image-based). However, Mamba-2's content-dependent gating may better model varying retail dynamics across stores/products compared to R2I's S4/S5 (linear time-invariant). **The recommended hybrid: R2I-style decoupled posterior with Mamba-2 backbone.**

---

## Dimension 5: connecting causal estimation to the world model decoder

The technical subtlety here is that the world model must learn `P(demand | do(price), state)` — the interventional distribution — during training on observational data where prices are endogenous. Three integration strategies have different rigor-complexity tradeoffs.

**The Hausman IV is fully constructible from Dominick's** without additional data collection. With 83 stores and weekly UPC-level prices, the leave-one-out average yields a strong instrument. Exclusion restriction: after store × product × week-of-year fixed effects, local demand shocks at store s are independent of prices at store s'. The main threat is chain-wide promotional campaigns creating correlated demand; mitigate with week × category fixed effects.

**The reported elasticities (OLS = −1.367, 2SLS = −2.470, 2sCOPE = −2.205) likely come from Yang, Qian, and Xie (2024, JMR)** but could not be independently verified for a specific Dominick's category. The pattern is consistent with the standard finding: OLS attenuates elasticity (managers raise prices when they anticipate high demand, creating positive confounding), while IV/copula methods recover more elastic estimates. For Dominick's grocery categories, expect true elasticities in the **−2.0 to −3.0 range** based on Nevo (2001) and Hoch et al. (1995).

**When to use each method.** Use DML+Hausman IV (PLIV) as the primary identification strategy — it handles nonlinear confounding via flexible ML nuisance models while solving endogeneity via the instrument. Use 2sCOPE as a robustness check that doesn't rely on the exclusion restriction. Use wholesale cost as a supplementary instrument for overidentification tests (Sargan/Hansen). If the DML-PLIV and 2sCOPE estimates diverge substantially, investigate whether correlated demand shocks (threatening Hausman) or non-Gaussianity (threatening copula) is the issue.

**The constrained decoder approach is the most principled world model integration.** Pre-estimate per-category θ̂ via DML-PLIV. The demand decoder then separates the **fixed causal price channel** from the **learnable latent-state residual**:

```python
# Per-category causal elasticities (pre-estimated, frozen)
theta_per_cat = {'analgesics': -2.31, 'cereals': -2.88, 'frozen_juice': -1.95, ...}

class CausalDemandDecoder(nn.Module):
    def __init__(self, theta_dict, latent_dim, num_categories):
        self.theta = nn.Embedding(num_categories, 1)
        self.theta.weight.data = torch.tensor([[v] for v in theta_dict.values()])
        self.theta.weight.requires_grad = False  # FROZEN
        self.residual = MLP(latent_dim + num_store_features, 256, 1)

    def forward(self, z_t, log_price, category_id, store_features):
        causal_price_effect = self.theta(category_id) * log_price
        residual_demand = self.residual(torch.cat([z_t, store_features], -1))
        return causal_price_effect + residual_demand
```

This ensures the world model cannot learn a confounded price-demand relationship during imagination. The residual network captures seasonality, promotional effects, store-level heterogeneity, and competitive dynamics through the latent state — everything except the price elasticity itself.

---

## Dimension 6: corrected hyperparameters for the offline tabular setting

**Training budget: 100K–200K gradient steps (~176–360 epochs).** With B=16, T=64 (1,024 transitions per batch), one epoch over 581K tuples takes ~567 gradient steps. TD-MPC2 offline trains ~1M steps over 545M transitions (~2 passes); D4RL benchmarks typically use 1M steps over ~1M transitions (~1000 epochs). For a world model on 581K tuples, **100K steps** is a conservative starting point; monitor reconstruction loss and KL divergence on a held-out validation set (last 20 weeks) for overfitting. Scale to 200K if the world model hasn't converged. Use weight decay (1e-4) and dropout (0.1) on the encoder/decoder MLPs to regularize.

**Replay sampling: stratified with mild recency weighting.** Uniform sampling ensures the world model learns the full range of economic conditions (1989–1997). Pure recency weighting would discard valuable historical variation. The recommended hybrid: 70% uniform sampling across all temporal strata (quarterly blocks), 30% overweighting the most recent 2 years. This aligns with DEER (2025) findings that hybrid approaches outperform pure strategies in non-stationary settings. For world model training specifically (MLE on observations), importance sampling corrections are unnecessary — all data is equally valid as supervised learning signal regardless of the behavior policy.

**Entity-factored cross-attention hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| Entity embedding dim | **256** | SOLD uses 256; sufficient for tabular features |
| Attention heads | **4** | SLATE uses 4 at dim=192; SOLD uses 8 at dim=256 |
| Key dimension d_k | **64 per head** | d_model/heads = 256/4 = 64 |
| Cross-attention layers | **1** with residual MLP | Tabular features don't need deep attention |
| Positional encoding | **None** — use learned entity ID embeddings | SKUs and stores are unordered sets; permutation invariance matters |
| Entity ID encoding | Learned: 25 SKU + 83 store embeddings | Added to feature vectors before cross-attention |

**Imagination horizon: H=13 (one retail quarter), γ=0.95.** In grocery retail, promotional planning operates on **4–6 week cycles**, category management reviews are quarterly (13 weeks in the 4-5-4 retail calendar), and seasonal planning extends 6 months. H=15 (DreamerV3 default) maps to 15 weeks ≈ one quarter — close to optimal. Recommend **H=13** to align exactly with the retail quarter. With discount factor **γ=0.95**, the effective planning horizon is ~20 weeks, providing appropriate emphasis on the next quarter while discounting further-out uncertainty. If the model shows stable multi-quarter predictions, experiment with H=26 (half year).

**DGX Spark optimization.** The DGX Spark's 128GB unified LPDDR5x has **~273 GB/s bandwidth** — roughly 5.5× less than a discrete GPU's HBM. The entity-factored model with Mamba-2 is **bandwidth-bound**, primarily due to Mamba-2's sequential scan operations. The unified memory architecture eliminates H2D/D2H transfer overhead (the entire 581K-tuple dataset fits in ~2–5 GB). Recommended batch strategy: **B=32, T=64** (larger than DreamerV3's B=16, leveraging the 128GB memory), with BF16 mixed precision (FP32 master weights and optimizer states, BF16 forward/backward). Keep SSM state transitions in FP32 accumulation for numerical stability. GradScaler is unnecessary for BF16. Estimated wall-clock: ~2–8 hours for 100K gradient steps.

---

## Dimension 7: corrected repo architecture and packaging

**complexipy is legitimate but mislabeled.** It is indeed Rust-backed (via PyO3/maturin), actively maintained, and available on PyPI. However, it measures **cognitive complexity** (how hard code is to understand), not cyclomatic complexity (independent code paths). For an ML research repo, use **ruff** with C901 rule for McCabe cyclomatic complexity alongside complexipy for cognitive complexity.

**Hatchling over setuptools.** The repo is a pure-Python package that depends on CUDA libraries (mamba-ssm, torch, triton) but doesn't compile CUDA extensions itself. Hatchling is the correct build backend. Critically, **do not list torch, mamba-ssm, or triton as hard dependencies** — they require CUDA version matching. Place them in optional dependencies and document the install sequence.

**Corrected module structure.** The review's `drama_decoupled_posterior.py` breaks the RSSM's natural computational graph (prior and posterior share the deterministic backbone). The `copula_endogeneity_correction.py` is a preprocessing transform, not a model component. The `stackelberg_follower_sim.py` is infeasible as described. Corrected boundaries:

```
src/retail_world_model/
├── data/                          # Loading & preprocessing
│   ├── dominicks_loader.py        # Dominick's dataset I/O
│   ├── copula_correction.py       # Endogeneity correction (preprocessing)
│   ├── transforms.py              # Feature engineering, normalization
│   └── schemas.py                 # Pydantic data schemas
├── models/                        # Core world model
│   ├── world_model.py             # Top-level compose: RSSM + heads
│   ├── rssm.py                    # RSSM core: prior + posterior + backbone
│   ├── mamba_backbone.py          # Mamba-2 deterministic backbone
│   ├── encoder.py                 # Observation encoder
│   ├── decoder.py                 # Demand decoder (with causal constraint)
│   ├── reward_head.py             # Ensemble of reward heads (5×)
│   └── posterior.py               # Posterior variants (standard, decoupled)
├── training/                      # Training loop & losses
│   ├── trainer.py                 # Three-phase loop
│   ├── losses.py                  # ELBO, KL, causal regularization
│   └── offline_utils.py           # LCB pessimism, replay sampling
├── inference/                     # Imagination & planning
│   ├── imagination.py             # Latent rollouts with robust sampling
│   └── planning.py                # CEM/gradient-based pricing
├── applications/                  # Downstream uses
│   ├── pricing_policy.py          # Actor-critic pricing agent
│   └── competitive_robustness.py  # Competition-conditioned imagination
├── api/                           # FastAPI serving
│   ├── serve.py                   # App factory with lifespan
│   ├── endpoints.py               # /health, /rollout, /predict, /rollout/stream
│   ├── batching.py                # Asyncio queue-based dynamic batcher
│   └── schemas.py                 # Request/response models
└── utils/
    ├── distributions.py           # OneHotDist with unimix, symlog/twohot
    ├── checkpoint.py              # Safetensors + HF Hub
    └── device.py                  # CUDA/mixed-precision management
```

**FastAPI serving.** For rollouts taking 100ms–1s, use an asyncio queue-based dynamic batcher (max batch size 8, max wait 50ms) that collects concurrent requests and runs batched GPU inference via `run_in_executor`. For long horizons, stream results via Server-Sent Events (SSE) using `sse-starlette`, yielding each imagination step as a JSON event. Key endpoints: `GET /health`, `POST /predict` (single-step, <50ms), `POST /rollout` (full trajectory), `POST /rollout/stream` (SSE streaming), `POST /predict/batch` (offline evaluation), `GET /model/info`.

**HuggingFace model card must disclose the Dominick's academic-use restriction.** The Kilts Center terms state: "These data are for academic research purposes only. Users must acknowledge the Kilts Center for Marketing at the University of Chicago Booth School of Business." This restriction applies transitively to models trained on this data. Use **CC-BY-NC-4.0** as the license, with explicit warnings that commercial use is not authorized. The model card must include temporal scope (1989–1997), geographic scope (Chicago metro), product scope (~3,500 UPCs in grocery), and the caveat that predictions may not generalize to modern retail environments.

---

## Conclusion: what to build, what to skip, and in what order

The enriched plan preserves the original blueprint's core architecture (DreamerV3-style RSSM with Mamba-2, entity-factored multi-SKU/multi-store, three-phase training) while making three critical corrections. First, **endogeneity correction via Hausman IV and constrained decoder** replaces overreliance on 2sCOPE — the former provides testable identification while the latter serves as robustness check only. Second, **MOPO-style LCB with 5-head reward ensemble** replaces the more complex COMBO proposal — it integrates trivially with DreamerV3 by modifying only the imagination reward signal. Third, **implicit competition via latent state + robust imagination** replaces the infeasible Stackelberg follower network — honest about data limitations while still producing competitively robust policies.

The most novel contribution of this plan is the causal constrained decoder, which freezes per-category price elasticities from DML-PLIV while allowing the residual network to learn everything else from the latent state. This is, to our knowledge, the first proposed integration of econometric causal identification directly into a DreamerV3-style world model's decoder architecture. The approach bridges two literatures — empirical IO demand estimation and model-based RL — that have not been previously connected at the implementation level.