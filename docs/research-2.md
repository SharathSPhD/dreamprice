# Building a retail pricing world model on Dominick's scanner data

**A Mamba-2-backed RSSM trained offline on 100 million store-week-SKU tuples can learn competitive pricing dynamics across 93 stores, enabling counterfactual demand estimation and imagination-based policy optimization on a single DGX Spark.** This synthesis provides the complete implementation blueprint — from raw CSV preprocessing through entity-centric latent dynamics to Stackelberg-aware actor-critic training — for a hybrid DreamerV3/DRAMA architecture targeting multi-SKU retail pricing. The approach is tractable: the tabular world model requires only ~10–50M parameters, trains in hours on the DGX Spark's 128 GB unified memory, and supports a full ablation campaign in under 30 days. Each of the eight dimensions below is specified at implementation depth with exact hyperparameters, code structures, and paper references.

---

## 1. Dominick's dataset schema and the preprocessing pipeline

### Raw data structure

The Dominick's Finer Foods dataset, hosted by the James M. Kilts Center at Chicago Booth, spans **400 weeks** (September 14, 1989 through May 14, 1997), **93 stores** (non-contiguous IDs 2–139, with 8 marked closed), **~18,000 unique UPCs** across **29 product categories**, yielding approximately **100 million** store-SKU-week tuples after cleaning. The data ships as SAS/CSV files organized into three file types.

**Movement files** (`wXXX.csv`, one per category) contain the core transaction records sorted by UPC → STORE → WEEK. The columns are:

| Column | Type | Meaning |
|--------|------|---------|
| `UPC` | int | Universal Product Code (merge key) |
| `STORE` | int | Store ID (2–139, non-contiguous) |
| `WEEK` | int | Week number 1–400 (Thursday–Wednesday) |
| `MOVE` | int | Units sold (individual items, not bundles) |
| `QTY` | int | Bundle size (usually 1; 3 for "3-for-$2") |
| `PRICE` | float | Total bundle retail price |
| `PROFIT` | float | Gross margin % based on Average Acquisition Cost |
| `SALE` | char | Promotion code: `B` (bonus buy), `C` (coupon), `S` (simple markdown), blank (no code) |
| `OK` | int | Data quality flag: 1 = valid, 0 = suspect |

The critical derived quantities are **unit price** = `PRICE / QTY`, **dollar sales** = `PRICE × MOVE / QTY`, and **wholesale cost per unit** = `PRICE × (1 − PROFIT/100) / QTY`. The PROFIT field uses Average Acquisition Cost, which introduces sluggish adjustment to wholesale price changes — an important caveat for cost-based reward computation.

**UPC files** (`upcXXX.csv`) provide `COM_CODE` (commodity sub-category), `NITEM` (tracks products across UPC re-launches), `DESCRIP`, `SIZE`, and `CASE`. **Store demographic files** contain 30+ variables from the 1990 Census processed by Market Metrics, including `income` (log median), `educ` (% college graduates), `ethnic` (% minority), `hsizeavg`, household size distribution, and the critical competition variables: `SSTRDIST` (distance to nearest warehouse/EDLP competitor like Cub Foods), `SSTRVOL` (relative sales volume of that competitor), `CPDIST5` (average distance to five nearest Hi-Lo supermarket competitors), and `CPWVOL5` (relative volume ratio). Montgomery (1997) found the volume ratios more informative than distances.

### The SALE field problem and promotion detection

The SALE field has **documented false negatives** — the Kilts Center manual states that if the field is set it indicates a promotion, but if blank, a promotion may still have occurred. The standard fix combines the SALE code with price-based detection: compute the **modal price** (most frequent price) per UPC-store as the regular price, then flag any week where `unit_price < 0.95 × modal_price` as promoted regardless of the SALE value. The derived features are a binary `on_promotion` indicator and a continuous `discount_depth = (regular_price − unit_price) / regular_price`.

### Preprocessing pipeline (step by step)

1. **Load and filter**: Read category movement CSV. Drop all rows with `OK == 0` or `PRICE ≤ 0`. Optionally drop `MOVE ≤ 0` if modeling positive-demand periods only.
2. **Compute unit price**: `unit_price = PRICE / QTY`. Compute wholesale cost: `cost = PRICE × (1 − PROFIT/100) / QTY`.
3. **Merge product metadata**: Join on `UPC` to obtain `DESCRIP`, `SIZE`, `COM_CODE`.
4. **Merge temporal metadata**: Map `WEEK` to calendar dates via the week decode table. Derive `year`, `month`, `quarter`, `week_of_year`, and binary holiday indicators (Thanksgiving, Christmas, Easter, Memorial Day, Labor Day).
5. **Merge store demographics**: Join on `STORE` to obtain all demographic and competition variables.
6. **Handle missing weeks**: For each active (store, UPC) pair — defined as appearing in ≥10% of that store's operating weeks — insert explicit zero-sales rows for any missing weeks. This distinguishes genuine zero demand from product non-availability.
7. **Promotion enhancement**: Compute modal price per (UPC, STORE). Create `on_promotion` and `discount_depth` features.
8. **Feature engineering**: Apply **symlog** to `MOVE` and `unit_price` for the world model's encoder input (replaces the traditional log transform while handling zeros gracefully). Compute `price_index = unit_price / category_mean_price` for cross-category comparability. Add **lag features**: `lag_price_1`, `lag_price_2`, `lag_move_1`, `lag_move_2`, `rolling_mean_move_4` (4-week rolling average), `rolling_std_price_4`. Encode store price tier (Cub-Fighter / Low / Medium / High) as a categorical.
9. **Panel balancing**: Following Montgomery (1997), restrict to the 83 stores with complete histories. For modeling a single category (recommended starting point), select a category with rich substitution structure — **beer** (29 sub-categories, high cross-elasticity), **soft drinks** (brand competition), or **canned soup** (moderate SKU count ~20–30 with clear substitution patterns).
10. **Temporal split**: **Train** on weeks 1–280 (~70%, through mid-1995), **validation** on weeks 281–340 (~15%, 1995–1996), **test** on weeks 341–400 (~15%, 1996–1997). No random shuffling — temporal ordering is inviolable.

For a single category like canned soup across 83 stores with ~25 active SKUs, the resulting dataset contains approximately **25 × 83 × 280 ≈ 581,000** training tuples and **25 × 83 × 60 ≈ 124,500** test tuples — sufficient for a mid-sized world model without data scarcity concerns.

### Constructing the observation vector

The observation at time *t* for a given store is the concatenation of per-SKU features across all *K* active SKUs in the category:

```
Per-SKU features (~12 dim each):
  symlog(unit_price), symlog(move), discount_depth, on_promotion,
  lag_price_1, lag_price_2, lag_move_1, lag_move_2,
  rolling_mean_move_4, price_index, cost_per_unit, profit_margin

Store context (~8 dim, shared across SKUs):
  income, educ, ethnic, hsizeavg, SSTRDIST, SSTRVOL, CPDIST5, CPWVOL5

Temporal context (~6 dim):
  sin(2π·week/52), cos(2π·week/52), quarter_onehot(4), holiday_binary
```

With *K* = 25 SKUs, the flat observation is 25 × 12 + 8 + 6 = **314 dimensions** — well within MLP encoder capacity. The entity-factored approach (Dimension 3) structures this more efficiently.

---

## 2. RSSM architecture with Mamba-2 backbone: exact wiring

### DreamerV3's RSSM equations

The Recurrent State-Space Model (Hafner et al., arXiv 2301.04104) maintains a model state *s_t* = {*h_t*, *z_t*} consisting of a deterministic recurrent state *h_t* and a stochastic categorical latent *z_t*:

```
Sequence model:     h_t = f(h_{t-1}, z_{t-1}, a_{t-1})     — originally Block GRU
Posterior encoder:  z_t ~ q(z_t | h_t, x_t)                 — 32 categoricals × 32 classes
Prior predictor:    ẑ_t ~ p(ẑ_t | h_t)                      — same structure
Reward head:        r̂_t = MLP(h_t, z_t) → twohot(255 bins)
Continue head:      ĉ_t = MLP(h_t, z_t) → Bernoulli
Decoder:            x̂_t = MLP(h_t, z_t) → symlog targets
```

The stochastic latent uses **32 independent categorical distributions, each over 32 classes**, yielding a 1024-dimensional one-hot vector (but only 32 × log₂(32) = 160 bits of information). All categoricals use **1% uniform mixing** ("unimix"): `probs = 0.99 × softmax(logits) + 0.01 / 32`, preventing near-deterministic distributions that would destabilize KL computation.

### Replacing the GRU with Mamba-2: the DRAMA pattern

Following DRAMA (Wang et al., arXiv 2410.08893, ICLR 2025), which is the first published world model using Mamba-2, the recommended wiring **decouples the posterior from the deterministic state**. This is the critical architectural decision:

```
DRAMA-style wiring (recommended):
  Encoder:       z_t = encode(x_t)              — posterior depends ONLY on observation
  Seq input:     input_t = Linear(cat(z_t, a_t)) — project to d_model
  Mamba-2:       d_t = Mamba2(input_{1:T})        — parallel during training
  Prior head:    ẑ_{t+1} = MLP(d_t)              — predict next latent from det. state
  Reward head:   r̂_t = MLP(d_t)
  Decoder:       x̂_t = decode(z_t)              — reconstruct from z_t only
```

The critical design insight is that **z_t does NOT depend on d_t during training**. This breaks the sequential dependency, enabling Mamba-2's parallel SSD (State Space Duality) scan over full sequences. During imagination rollouts, the model switches to Mamba-2's recurrent `step()` function for single-step autoregressive generation.

The alternative R2I-style wiring (Samsami et al., arXiv 2403.04253, ICLR 2024 Oral) preserves the dependency q(z_t | h_t, x_t) but achieves parallelism by making the representation model "non-recurrent" during training — computing z_t from x_t only during the parallel pass, then using the full dependency during imagination. Either approach works; the DRAMA pattern is simpler to implement.

### Mamba-2 library interface

```python
# Installation: pip install mamba-ssm (requires CUDA, triton, causal-conv1d)
from mamba_ssm import Mamba2
from mamba_ssm.utils.generation import InferenceParams

# Constructor
mamba_block = Mamba2(
    d_model=512,       # model dimension
    d_state=64,        # SSM state expansion factor
    d_conv=4,          # local convolution width
    expand=2,          # d_inner = expand × d_model = 1024
    headdim=64,        # nheads = d_inner // headdim = 16
    chunk_size=256,    # SSD chunk size
    rmsnorm=True,      # use RMSNorm before output
)
# Parameters per block: ~3 × expand × d_model² ≈ 1.6M for d_model=512

# Training (parallel mode): processes full sequences via SSD
y = mamba_block(x)  # x: (B, T, 512) → y: (B, T, 512)

# Inference (recurrent mode): single-step with maintained state
inference_params = InferenceParams(max_seqlen=1, max_batch_size=B)
# Each call to step() maintains:
#   conv_state: (B, d_inner, d_conv) — rolling convolution buffer
#   ssm_state:  (B, nheads, headdim, d_state) — SSM hidden state
y_t = mamba_block(x_t.unsqueeze(1), inference_params=inference_params)
```

**h_t is the output of the Mamba2 block (shape B × d_model), NOT the internal SSM state.** The internal SSM state (B × nheads × headdim × d_state) is managed internally by `step()` and should not be directly manipulated. Episode boundaries are handled by resetting `conv_state` and `ssm_state` to zero.

### Complete module structure

```python
class MambaWorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, d_model=512,
                 n_cat=32, n_cls=32, n_bins=255):
        super().__init__()
        latent_dim = n_cat * n_cls  # 1024

        # Tabular encoder: symlog(obs) → embedding → categorical posterior
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model), nn.SiLU(), nn.RMSNorm(d_model),
            nn.Linear(d_model, d_model), nn.SiLU(), nn.RMSNorm(d_model),
        )
        self.posterior_head = nn.Linear(d_model, n_cat * n_cls)

        # Sequence model input projection
        self.input_proj = nn.Linear(latent_dim + act_dim, d_model)

        # Mamba-2 backbone (replaces Block GRU)
        self.mamba = Mamba2(d_model=d_model, d_state=64, d_conv=4,
                           expand=2, headdim=64, chunk_size=256)

        # Prior and prediction heads
        self.prior_head = nn.Linear(d_model, n_cat * n_cls)
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, n_bins),  # twohot bins
        )
        self.continue_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model + latent_dim, d_model), nn.SiLU(),
            nn.Linear(d_model, obs_dim),  # predicts symlog(obs)
        )

        # Zero-init output layers (DreamerV3 recipe)
        nn.init.zeros_(self.reward_head[-1].weight)
        nn.init.zeros_(self.reward_head[-1].bias)
```

For the Size-S configuration appropriate to tabular retail data, use `d_model=512`, yielding ~22M total parameters for the world model. For initial prototyping, `d_model=256` (~5M params) trains in minutes.

---

## 3. Entity-factored state representation for multi-SKU, multi-store dynamics

### Why flat concatenation fails and entity factorization succeeds

Standard world models observe a single environment. Concatenating all 25 SKUs × 12 features = 300 dimensions into a flat vector discards entity structure, cannot generalize to new SKUs, and forces the model to learn cross-product interactions implicitly. The entity-centric approach, inspired by SOLD (Villar-Corrales et al., arXiv 2410.08822), Object-Centric Dreamer, and FIOC-WM (Feng et al., arXiv 2511.02225), represents each SKU as a separate **slot** with shared dynamics parameters, then models cross-SKU interactions via attention.

### Recommended architecture: dual-attention entity-factored RSSM

The architecture separates two attention mechanisms that map directly to distinct economic phenomena:

- **Temporal self-attention** (within-entity across time): Each SKU slot attends to its own history, capturing trend, seasonality, stockpiling, and individual promotion response.
- **Relational cross-attention** (across-entities within timestep): All SKU slots at time *t* attend to each other, capturing substitution (when SKU *j*'s price drops, SKU *i*'s demand drops) and complementarity. The attention weights are directly interpretable as learned substitution/complementarity structure.

### Entity embedding layer

```python
class EntityEncoder(nn.Module):
    def __init__(self, n_upcs=500, n_stores=93, n_continuous=12, d_slot=64):
        super().__init__()
        self.upc_embed = nn.Embedding(n_upcs, 32)
        self.store_embed = nn.Embedding(n_stores, 16)
        self.brand_embed = nn.Embedding(100, 16)     # brand grouping
        self.month_embed = nn.Embedding(12, 6)

        self.continuous_proj = nn.Sequential(
            nn.Linear(n_continuous, 32), nn.SiLU(), nn.RMSNorm(32))
        self.fusion = nn.Sequential(
            nn.Linear(32+16+16+6+32, 128), nn.SiLU(), nn.RMSNorm(128),
            nn.Linear(128, d_slot))

    def forward(self, upc_id, store_id, brand_id, month, continuous):
        cat = torch.cat([self.upc_embed(upc_id), self.store_embed(store_id),
                         self.brand_embed(brand_id), self.month_embed(month)], -1)
        cont = self.continuous_proj(symlog(continuous))
        return self.fusion(torch.cat([cat, cont], -1))  # (B, K, d_slot)
```

### Transition model with relational attention

Each time step processes *K* entity slots through three stages:

1. **Individual dynamics** (shared-parameter GRU/Mamba per entity): `h_t^k = BlockGRU(h_{t-1}^k, [z_{t-1}^k, a_t^k])` with `d_hidden=128`, 4 blocks of 32. Parameters are shared across all *K* SKUs — the entity embedding provides specialization.
2. **Relational cross-attention**: `H_t = MultiHeadAttention(Q=h_t, K=h_t, V=h_t)` with 4 heads × 32 dims. For *K* = 25 SKUs, the O(K²) cost is negligible. This step captures how SKU *i*'s state is influenced by all other SKUs' current states.
3. **Prior/Posterior heads**: `p(z_t^k | H_t^k)` = MLP(H_t^k) → 8 categoricals × 8 classes per entity (reduced from 32×32 since each entity carries less information than the full scene).

### Store conditioning via multi-agent parameter sharing

Following Terry et al. (arXiv 2005.13625), shared world model parameters + store ID embedding is provably sufficient for store-specific dynamics. The store embedding enters through the entity encoder, allowing the model to learn that Cub-Fighter stores have higher price sensitivity than High-tier stores without separate models.

### Scalability on DGX Spark

For 25 SKUs across 93 stores with `d_slot=64`, `d_hidden=128`, the entity-factored model has approximately **350K parameters** — requiring less than 1 GB for training. Even scaling to all 93 stores simultaneously (93 × 200 MB activations ≈ 18 GB) fits comfortably in 128 GB unified memory. Training throughput will be **memory-bandwidth bound** (273 GB/s) rather than compute-bound, yielding ~1000+ gradient steps/second and complete training in under 30 minutes per seed.

---

## 4. Stackelberg pricing as a POMDP with latent competitor dynamics

### Formal MDP specification

The Stackelberg pricing problem reduces to a **POMDP from the retailer's perspective** where the competitor's behavior is part of the environment's transition dynamics. Following the Stackelberg POMDP formalization (arXiv 2210.03852):

| Component | Definition |
|-----------|-----------|
| **State** *s_t* | (own_prices, competitor_price_proxy, demand_history, promo_calendar, inventory_proxy, seasonality, store_demographics) |
| **Observation** *o_t* | Full state minus competitor costs, competitor inventory, competitor strategy intent |
| **Action** *a_t* | Per-SKU price adjustment Δp ∈ [−M, M], discretized to 21 levels (−10%, −8%, ..., 0, ..., +8%, +10%) for DQN compatibility, or continuous for SAC |
| **Transition** *T(s'|s,a)* | World model predicts next-week demand, competitor price proxy, and category switching — competitor response is internalized |
| **Reward** *r_t* | Σ_k [(price_k − cost_k) × units_sold_k] − λ_vol × Σ_k |Δprice_k| − λ_stock × stockout_count |

### The competitor as a latent variable

The Dominick's dataset does not contain actual competitor prices from other chains. Instead, the competition variables (`SSTRDIST`, `SSTRVOL`, `CPDIST5`, `CPWVOL5`) provide a **static** competitive environment characterization, and the price tier system (Cub-Fighter vs. High) reflects Dominick's competitive positioning. The world model internalizes competitor dynamics by learning transition dynamics that implicitly capture how market conditions (including unobserved competitor behavior) respond to Dominick's pricing actions. Specifically, the recurrent hidden state *h_t* maintained by the Mamba-2 backbone encodes a **belief state** over the competitor's unobserved strategy — analogous to He et al.'s DRON (ICML 2016) implicit opponent modeling.

Within the Dominick's dataset, the primary "competitive" dynamics are **cross-store and cross-SKU**: stores in different price tiers respond differently to the same pricing policy, and SKUs within a category exhibit substitution. The world model captures both: store embeddings encode the competitive environment, and relational cross-attention captures SKU-level substitution.

### Reward function with degenerate strategy prevention

```python
def compute_reward(price, cost, units_sold, prev_price,
                   lambda_vol=0.05, lambda_margin_floor=0.10):
    gross_margin = ((price - cost) * units_sold).sum(dim=-1)  # sum over SKUs
    volatility_penalty = lambda_vol * (price - prev_price).abs().sum(dim=-1)
    # Floor prevents race-to-bottom: penalize if margin% < 10%
    margin_pct = (price - cost) / price.clamp(min=0.01)
    floor_penalty = F.relu(lambda_margin_floor - margin_pct).sum(dim=-1)
    return gross_margin - volatility_penalty - floor_penalty
```

The **volatility penalty** prevents excessive price churn. The **margin floor penalty** prevents race-to-bottom strategies. Both are essential — without them, the agent discovers degenerate policies (perpetual deep discounts to maximize unit volume at negligible margin).

### Counterfactual reasoning via world model rollouts

World models naturally support counterfactual queries: given observed state *s_t*, substitute a different price action *a'_t* and roll the model forward to estimate counterfactual demand. This is equivalent to the do-calculus intervention do(price = *p*) when the world model has correctly learned causal structure. The key limitation is that observationally-trained world models learn P(demand | price, state) which may conflate causal and confounded relationships — the offline learning dimension (Section 7) addresses this.

---

## 5. The three-phase training loop in exact detail

### Phase A: world model learning

Every training step samples a batch of **B = 16 sequences of length T = 64** from the replay buffer (which is the entire offline dataset). The world model loss combines three terms with fixed coefficients:

```
L(φ) = β_pred × L_pred + β_dyn × L_dyn + β_rep × L_rep
     = 1.0 × L_pred   + 0.5 × L_dyn   + 0.1 × L_rep
```

**Prediction loss** `L_pred` sums three components. The **decoder loss** uses symlog squared error: `½(f(x,θ) − symlog(y))²` for continuous observations. The **reward loss** uses categorical cross-entropy against twohot-encoded symlog-transformed rewards over 255 bins in [−20, +20]. The **continue loss** uses binary cross-entropy for predicting episode termination (in the retail context, this models store closures or product discontinuations).

**Dynamics loss** `L_dyn = max(1, KL[sg(posterior) ‖ prior])` trains only the prior (dynamics predictor) by stop-gradienting the posterior. The **free bits threshold of 1 nat** disables the loss when KL is already small, preventing the dynamics from becoming trivially simple.

**Representation loss** `L_rep = max(1, KL[posterior ‖ sg(prior)])` trains only the encoder by stop-gradienting the prior. The **5:1 asymmetry** (β_dyn = 0.5 vs. β_rep = 0.1) ensures the prior is pushed hard toward the information-rich posterior while applying only gentle pressure on the posterior toward simplicity — this prevents posterior collapse.

The world model optimizer uses **Adam at lr = 1×10⁻⁴** with global gradient norm clipping at 1000.

### Phase B: actor training in imagination

From the batch of encoded initial states, the model unrolls **H = 15 imagination steps** using only the prior (no encoder needed):

1. Sample action: a_t ~ π_θ(a_t | h_t, z_t)
2. Advance: h_{t+1} = Mamba2.step(Linear(cat(z_t, a_t)))
3. Sample prior: z_{t+1} ~ p(z_{t+1} | h_{t+1})
4. Predict: r_t, c_t from MLP heads

**Lambda-returns** are computed backward from the horizon:

```
R^λ_T = v_ψ(s_T)                                    [bootstrap]
R^λ_t = r_t + γ·c_t·((1−λ)·v_ψ(s_{t+1}) + λ·R^λ_{t+1})
```

with **γ = 0.997** and **λ = 0.95**. Returns are normalized by **percentile scaling**: `S = EMA(P95(R^λ) − P5(R^λ))` with decay 0.99, then `R_normalized = R^λ / max(1, S)`. The `max(1, S)` prevents amplification of small returns under sparse rewards.

The actor loss combines REINFORCE (for discrete actions) or reparameterization (for continuous) with entropy regularization at **η = 3×10⁻⁴**. The actor optimizer uses **Adam at lr = 3×10⁻⁵** with global norm clipping at 100.

### Phase C: critic training in imagination

The critic predicts a distribution over returns using **twohot discrete regression** with 255 bins spanning [−20, +20] in symlog space. The loss is categorical cross-entropy against soft twohot labels: `L_critic = −Σ_t twohot(symlog(R^λ_t))ᵀ · log p_ψ(·|s_t)`. An **EMA slow critic** (decay = 0.98) provides regularization: an additional cross-entropy term between the fast critic's output and the EMA critic's output, weighted at 1.0. The critic optimizer matches the actor: **Adam at lr = 3×10⁻⁵**, gradient norm clipping at 100.

### Complete hyperparameter table

| Parameter | Value | Source |
|-----------|-------|--------|
| Batch size B | 16 | DreamerV3 (fixed across all domains) |
| Sequence length T | 64 | DreamerV3 |
| WM learning rate | 1×10⁻⁴ | DreamerV3 |
| Actor learning rate | 3×10⁻⁵ | DreamerV3 |
| Critic learning rate | 3×10⁻⁵ | DreamerV3 |
| Imagination horizon H | 15 | DreamerV3 |
| Discount γ | 0.997 | DreamerV3 |
| Lambda λ | 0.95 | DreamerV3 |
| Entropy scale η | 3×10⁻⁴ | DreamerV3 |
| β_pred / β_dyn / β_rep | 1.0 / 0.5 / 0.1 | DreamerV3 |
| Free bits | 1.0 nat | DreamerV3 |
| Unimix | 0.01 | DreamerV3 |
| Twohot bins | 255, range [−20, +20] | DreamerV3 |
| Critic EMA decay | 0.98 | DreamerV3 |
| Return norm percentiles | 5th, 95th | DreamerV3 |
| Return norm EMA decay | 0.99 | DreamerV3 |
| Replay capacity | Full offline dataset | Modified for offline |
| Gradient clip (WM) | 1000 (global norm) | DreamerV3 |
| Gradient clip (actor, critic) | 100 (global norm) | DreamerV3 |

### Offline setting modification

The training loop is identical except: (1) the replay buffer is the entire fixed offline dataset — no new data is ever added; (2) there is no environment interaction step; (3) the training ratio concept is replaced by simply running gradient steps over the dataset for a fixed number of iterations (recommend **500K–1M** total gradient steps); (4) stored latent states can still be refreshed during training to maintain consistency with the evolving model.

---

## 6. The five technical tricks: symlog, twohot, straight-through, KL balancing, and percentile normalization

### Symlog transform

The function `symlog(x) = sign(x) · ln(|x| + 1)` compresses large magnitudes while approximating the identity near zero. Its inverse is `symexp(x) = sign(x) · (exp(|x|) − 1)`. In DreamerV3, symlog is applied to **(a) encoder inputs** (squashes raw observations before the MLP), **(b) decoder targets** (the decoder predicts in symlog space, loss is ½(f(x,θ) − symlog(y))²), **(c) reward prediction targets**, and **(d) critic targets** (both via the twohot encoding over symlog-space bins).

For retail data, symlog is transformative: weekly demand ranges from 0 to 10,000+ units, prices span $0.29 to $15.99, and revenue varies from cents to hundreds of thousands of dollars per store-week. Symlog handles all scales without manual normalization. Unlike log, it is defined at zero (symlog(0) = 0), symmetric around the origin, and differentiable everywhere.

```python
def symlog(x): return torch.sign(x) * torch.log1p(torch.abs(x))
def symexp(x): return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
```

### Twohot encoding for distributional prediction

For a scalar target *x* mapped to symlog space, twohot assigns fractional weights to the two nearest bins: if *x* falls between bins *b_k* and *b_{k+1}*, then `weight_k = |b_{k+1} − x| / |b_{k+1} − b_k|` and `weight_{k+1} = 1 − weight_k`. The predicted value is recovered as `symexp(Σ_i probs_i × b_i)` — the expected bin value under the predicted softmax distribution, inverse-transformed.

With **255 bins uniformly spaced in symlog [−20, +20]**, the representable range in original space extends to approximately ±4.85 × 10⁸, with increasing resolution near zero. The cross-entropy loss against twohot soft labels provides richer gradients than MSE regression and naturally handles bimodal distributions (e.g., weeks with zero demand vs. promotional spikes).

### Straight-through gradients

The discrete categorical latents (32 × 32 in standard DreamerV3, or 8 × 8 per entity in the factored version) use the straight-through estimator:

```python
one_hot = F.one_hot(dist.sample(), num_classes=n_cls).float()
z = one_hot + probs - probs.detach()  # forward: one_hot; backward: probs
```

In the forward pass, `probs − probs.detach() = 0`, so the output is exactly `one_hot`. In the backward pass, `one_hot` has no gradient and `probs.detach()` has no gradient, so gradients flow through `probs` as if the sampling step were the identity function. This is simpler than Gumbel-Softmax (no temperature annealing) and produces genuinely discrete representations.

### KL balancing

The **5:1 asymmetry** (`β_dyn = 0.5` for the dynamics loss with stop-gradiented posterior, `β_rep = 0.1` for the representation loss with stop-gradiented prior) is critical. Without it (equal weighting), the posterior collapses — the encoder learns to ignore observations and output the prior, giving trivially low KL but uninformative latents. With asymmetric weighting, the prior is forced to become expressive enough to match the information-rich posterior, while the posterior faces only gentle regularization. The **free bits mechanism** (clipping at 1 nat) further protects against collapse by disabling KL minimization once it reaches a healthy floor.

### Percentile return normalization

The formula `R_norm = R^λ / max(1, P95 − P5)` has a crucial property: the `max(1, ·)` ensures returns are **never amplified**, only compressed. Under sparse rewards where the return range is small (P95 − P5 < 1), returns pass through unscaled, preserving the entropy regularizer's dominance and promoting exploration. This is strictly superior to mean/std normalization, which amplifies near-zero returns and causes premature convergence.

---

## 7. Offline learning challenges: distributional mismatch, causality, and non-stationarity

### The distributional shift problem

The Dominick's dataset records the outcomes of Dominick's actual pricing policies — not an exploratory random policy. This means some (state, action) regions are severely underrepresented: extreme price cuts for premium products, unusual promotional combinations, and counter-seasonal pricing were rarely attempted. A world model trained naively will make unreliable predictions in these gaps, and a policy optimizer will exploit exactly these unreliable predictions (the "model exploitation" problem).

### Recommended solution: COMBO-style conservative optimization

Three offline MBRL methods address this. **MOPO** (Yu et al., NeurIPS 2020, arXiv 2005.13239) penalizes rewards by model ensemble variance: `r̃ = r − λ · max_std(ensemble)`. **MOReL** (Kidambi et al., NeurIPS 2020, arXiv 2005.05951) terminates rollouts that enter unknown state-action regions. **COMBO** (Yu et al., NeurIPS 2021, arXiv 2102.08363) applies CQL-style value regularization to model-generated out-of-distribution state-action pairs without requiring explicit uncertainty quantification.

For this application, **COMBO is recommended** because it avoids the fragile calibration of uncertainty ensembles while providing theoretical guarantees. The implementation adds a regularizer to the critic loss that pushes down Q-values for model-imagined state-action pairs that deviate from the behavior policy distribution, preventing the policy from exploiting model errors at extreme price points.

### The causal identification problem

A world model trained on observational data learns P(demand | price, state), which conflates causal effects with confounding — prices were set partly in response to unobserved demand conditions. The recommended two-stage approach:

1. **Estimate causal elasticities** via Double/Debiased Machine Learning (Chernozhukov et al., arXiv 1608.00060). Use **Hausman instruments** (average price of the same UPC across other stores as an instrument for own-store price, exploiting shared wholesale cost shocks) or wholesale cost directly as an instrument.
2. **Constrain the world model** with these causal estimates: add a regularization term penalizing the model's implied price elasticity when it deviates significantly from the IV/DML estimate. For Dominick's, Chernozhukov et al. (2017) found category-level elasticities ranging from **−2.71** (sodas) to **−0.4** (tableware) using this framework on similar scanner data.

Include SALE, feature, and display indicators as **action dimensions** (not just state features) since Dominick's controlled these. This allows the model to distinguish the causal price effect from promotional lift. The interaction `price × on_promotion` should be explicitly modeled — promoted items typically exhibit steeper demand curves due to heightened shopper attention.

### Non-stationarity across 8 years

The 1989–1997 span covers multiple macroeconomic regimes and consumer preference shifts. The recommended approach combines **context-conditioned modeling** (include year, quarter, and macroeconomic indicators as state features so the world model learns regime-dependent dynamics) with **exponential recency weighting** in the replay buffer (`P(sample_i) ∝ exp(−λ·(T − t_i))` with λ ≈ 0.005 per week). Additionally, segment the dataset into ~2-year blocks and sample with decreasing probability from older segments (40% most recent, 25% next, 20%, 15% oldest).

Following COSPA (Ackermann et al., RLC 2024, arXiv 2405.14114), the world model's latent state can be encouraged to encode regime information through a contrastive predictive coding auxiliary loss that distinguishes early-period from late-period transitions.

---

## 8. Experiment design: metrics, baselines, ablations, and compute budget

### World model quality metrics

Evaluate prediction accuracy at horizons **h = {1, 5, 10, 15, 25}** weeks using RMSE and MAE on demand (in original units via symexp), WMAPE for portfolio-level accuracy, and CRPS for distributional calibration. Plot the **normalized degradation rate** NDR(h) = RMSE(h)/RMSE(1) — it should grow sub-linearly. For latent space quality, train a linear probe on frozen representations to predict product category, season, and promotional status; report accuracy. Visualize latent states via UMAP colored by category and price regime.

### RL agent performance metrics

The primary metric is **cumulative gross margin** over the test period (weeks 341–400) aggregated across all stores. Secondary metrics include policy regret versus an oracle with true demand elasticities, **response latency** (cross-correlation lag between competitor proxy changes and agent price adjustments), **price volatility** (std of week-over-week price changes), and **margin variance** (lower = more stable earnings).

### Baseline hierarchy

- **Cost-plus fixed markup** (25% markup): lower bound, zero intelligence
- **Static optimization**: XGBoost demand model + per-week price optimization (strongest non-RL baseline)
- **Competitive matching**: price = category_avg ± 2%
- **DQN** with discretized price grid (21 levels)
- **PPO** (on-policy model-free)
- **SAC** (off-policy model-free, continuous actions)

### Ablation priority order

| Priority | Ablation | Tests |
|----------|----------|-------|
| 1 | Imagination ON vs. OFF (model-free actor-critic from data only) | Core world model contribution |
| 2 | Imagination horizon sweep {5, 10, 15, 25} | Optimal planning depth for weekly pricing |
| 3 | Stochastic vs. deterministic-only latent | Uncertainty modeling value |
| 4 | With vs. without symlog + twohot | Reward scale robustness |
| 5 | GRU vs. Mamba-2 backbone | Long-range memory benefit |
| 6 | Entity embeddings vs. flat concatenation | Structural representation value |
| 7 | With vs. without COMBO conservatism | Offline safety value |

### Statistical protocol

Run **10 seeds** per main configuration and each baseline, **5 seeds** per ablation. Report IQM (interquartile mean) with 95% stratified bootstrap confidence intervals following Agarwal et al. (NeurIPS 2021). Use paired bootstrap tests for ablation comparisons (same seeds for full model and each ablated variant). Apply Holm-Bonferroni correction for the 7-way ablation comparison.

### Compute budget on DGX Spark

| Phase | Runs | Hours/run | Total hours |
|-------|------|-----------|-------------|
| Main model (10 seeds) | 10 | 8–16 | 80–160 |
| 6 baselines (10 seeds each) | 60 | 3–8 | 180–480 |
| 7 ablations (5 seeds each) | 35 | 6–12 | 210–420 |
| **Total** | **105** | — | **470–1,060** |
| **Wall-clock (sequential)** | — | — | **20–44 days** |

The 128 GB unified memory eliminates data-loading bottlenecks for the entire Dominick's dataset. The ~100 TFLOPS BF16 puts the DGX Spark between an RTX 5070 and 5070 Ti in raw compute, but the memory advantage enables larger batch sizes and eliminates CPU-GPU transfer overhead. For initial development, the entity-factored model (350K params) trains in under 30 minutes per seed, enabling rapid iteration before scaling to the full 10-seed evaluation.

### Visualization toolkit

Generate **policy heat maps** (own_price × competitor_proxy → action), **imagination rollout comparisons** (real vs. imagined demand trajectories), **learned demand response curves** (world model's D(p) vs. empirical D(p) for representative SKUs), **relational attention matrices** (revealing learned substitution structure between SKUs), and standard **training curves** (cumulative margin vs. gradient steps, with shaded ±1 std across seeds). The attention weights from the relational cross-attention layer provide a particularly compelling visualization: they directly show which products the model treats as substitutes (high attention when one is promoted) versus complements.

---

## Conclusion: a tractable path from scanner data to pricing intelligence

The complete architecture — entity-factored Mamba-2 RSSM with DRAMA-style decoupled posteriors, DreamerV3's proven training recipes, and COMBO-style offline conservatism — is **entirely implementable by a solo researcher on a single DGX Spark**. The critical insight is that tabular retail data requires orders of magnitude less compute than the visual domains DreamerV3 was designed for: a 350K-parameter entity model trains in minutes, while even a full 50M-parameter model completes in hours.

Three technical decisions matter most for success. First, the **DRAMA-style posterior decoupling** (z_t depends only on x_t, not on h_t) enables Mamba-2's parallel SSD during training while preserving recurrent imagination — this delivers the 2–9× speedup that makes exhaustive ablation studies feasible. Second, the **dual-attention factorization** (temporal within-SKU + relational across-SKU) captures the economic structure that flat models miss: substitution, complementarity, and cross-store heterogeneity emerge naturally from shared parameters plus entity embeddings. Third, **causal regularization** via DML-estimated elasticities transforms the world model from a mere correlational predictor into a tool for counterfactual pricing analysis — the fundamental requirement for any policy that must generalize beyond the behavior distribution in the training data.

The recommended starting point is a single product category (canned soup or beer, ~25 SKUs) across all 83 complete stores, training the entity-factored world model for 500K gradient steps, then evaluating imagination-based policy optimization against the XGBoost + static optimization baseline. If the world model's multi-step demand predictions show sub-linear error growth and the RL agent's gross margin exceeds the static optimizer's by a statistically significant margin across 10 seeds, the architecture validates and can be extended to multi-category operation.