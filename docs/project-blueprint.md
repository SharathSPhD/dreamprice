# DreamPrice: Complete Project Blueprint

**A Mamba-2-backed RSSM trained offline on Dominick's scanner data to learn competitive retail pricing dynamics, enabling counterfactual demand estimation and imagination-based policy optimization.**

This document is the single authoritative reference for the DreamPrice project. It consolidates and reconciles the core research design, all subsequent corrections, and data-level specifics into one blueprint suitable for building a detailed spec file.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset: Dominick's Finer Foods](#2-dataset-dominicks-finer-foods)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [World Model Architecture](#4-world-model-architecture)
5. [Entity-Factored State Representation](#5-entity-factored-state-representation)
6. [Pricing as a POMDP with Latent Competition](#6-pricing-as-a-pomdp-with-latent-competition)
7. [Three-Phase Training Loop](#7-three-phase-training-loop)
8. [Core Technical Mechanisms](#8-core-technical-mechanisms)
9. [Causal Identification & Endogeneity Correction](#9-causal-identification--endogeneity-correction)
10. [Offline Learning: Pessimism & Distribution Shift](#10-offline-learning-pessimism--distribution-shift)
11. [Experiment Design](#11-experiment-design)
12. [Software Architecture & Packaging](#12-software-architecture--packaging)
13. [API & Serving](#13-api--serving)
14. [Demo & Visualization](#14-demo--visualization)
15. [Paper Positioning & Framing](#15-paper-positioning--framing)
16. [Execution Roadmap](#16-execution-roadmap)
17. [Appendix: Data File Reference](#appendix-data-file-reference)

---

## 1. Project Overview

### What DreamPrice is

DreamPrice is the first learned world model for economic/retail environments. It combines:

- **DreamerV3's** proven RSSM training recipe (symlog, twohot, KL balancing, percentile return normalization)
- **DRAMA-style** decoupled posteriors enabling Mamba-2's parallel SSD during training
- **Mamba-2** selective state-space backbone for O(n) sequence processing and recurrent imagination
- **Entity-factored** multi-SKU/multi-store representation with relational cross-attention
- **Causal identification** via Hausman IV + DML-constrained decoder
- **MOPO-style** offline pessimism via reward ensemble lower confidence bound

### Why it matters

Every existing RL pricing system uses either hand-crafted analytical simulators or model-free methods. No learned world model exists for economic environments. The gap spans five dimensions: no learned retail dynamics model, no Dyna-style imagination training for pricing, no standardized RL environment for pricing, no world model on economic time series, and no competitive pricing sandbox with a learned model.

### Positioning (2x2 gap matrix)

|  | Learned model | Hand-crafted model |
|--|---|---|
| **Physical domain** | DreamerV1–V3, IRIS, TD-MPC2, Cosmos, Genie | MuJoCo, Isaac Gym, CARLA |
| **Economic domain** | **DreamPrice (this work)** | ABIDES, DSGE models, AI Economist |

### Scale & tractability

The tabular world model requires only ~10–50M parameters (entity-factored variant: ~350K params), trains in hours on a DGX Spark's 128 GB unified memory, and supports a full ablation campaign in under 30 days. The structured economic state space (100–300 dimensions) is fundamentally more tractable than the 64×64×3 image observations DreamerV3 was designed for.

### Hardware target: DGX Spark

- Measured BF16 performance: ~100 TFLOPS (the marketed "1 PFLOP" is FP4 sparse theoretical peak)
- Memory bandwidth: ~273 GB/s (LPDDR5x, ~5.5× less than discrete GPU HBM)
- 128 GB unified memory: the genuine advantage — enables larger batch sizes, eliminates CPU-GPU transfer overhead, and fits the entire Dominick's dataset (~2–5 GB) in memory
- Training is **memory-bandwidth bound** rather than compute-bound

---

## 2. Dataset: Dominick's Finer Foods

### Overview

The Dominick's Finer Foods dataset (Kilts Center, University of Chicago Booth) spans **400 weeks** (September 14 1989 – May 14 1997), **93 stores** (non-contiguous IDs 2–139, with 8 marked closed), **~18,000 unique UPCs** across **29 product categories**, yielding approximately **100 million** store-SKU-week tuples after cleaning.

It is freely available for academic research. The Kilts Center terms state: *"These data are for academic research purposes only. Users must acknowledge the Kilts Center for Marketing at the University of Chicago Booth School of Business."* This restriction applies transitively to trained models. Use **CC-BY-NC-4.0** as the license.

### Why Dominick's is ideal

It directly contains state-action-outcome tuples: the "action" is the price/promotion decision, the "state" includes competitor context, seasonality, and store demographics, and the "outcome" is observed demand. The PROFIT field enables wholesale cost derivation, and the panel structure (93 stores) supports Hausman cross-store instruments — making causal identification feasible without external data.

### Raw data structure — three file types

#### Movement files (`wXXX.csv`, one per category)

Core transaction records sorted by UPC → STORE → WEEK.

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

**Critical derived quantities:**
- **Unit price** = `PRICE / QTY`
- **Dollar sales** = `PRICE × MOVE / QTY`
- **Wholesale cost per unit** = `PRICE × (1 − PROFIT/100) / QTY`

**PROFIT field caveat:** Uses Average Acquisition Cost (AAC), not replacement cost. AAC updates sluggishly as old inventory sells off and can drop precipitously from forward-buying. This makes it a noisy instrument — relevant (drives retail price) but partially endogenous (AAC reflects past demand through inventory dynamics). It should serve as a **supplementary** instrument, not the sole one.

The actual CSV files also include `PRICE_HEX` and `PROFIT_HEX` columns containing the IEEE 754 double-precision hex representation — these are for data integrity verification and should be dropped during preprocessing.

#### UPC files (`upcXXX.csv`, one per category)

| Column | Meaning |
|--------|---------|
| `COM_CODE` | Commodity sub-category code |
| `UPC` | Universal Product Code (merge key) |
| `DESCRIP` | Product description |
| `SIZE` | Package size string |
| `CASE` | Case pack quantity |
| `NITEM` | Tracks products across UPC re-launches |

#### Store demographic file (`demo.csv`)

Contains 300+ variables from the 1990 Census processed by Market Metrics, one row per store. Key variables:

| Variable | Meaning |
|----------|---------|
| `STORE` | Store ID (merge key) |
| `INCOME` | Log median household income |
| `EDUC` | % college graduates |
| `ETHNIC` | % minority population |
| `HSIZEAVG` | Average household size |
| `AGE9` | % population under 9 |
| `AGE60` | % population over 60 |
| `WORKWOM` | % working women |
| `SSTRDIST` | Distance to nearest warehouse/EDLP competitor (e.g. Cub Foods) |
| `SSTRVOL` | Relative sales volume of that competitor |
| `CPDIST5` | Average distance to five nearest Hi-Lo supermarket competitors |
| `CPWVOL5` | Relative volume ratio of those competitors |
| `ZONE` | Pricing zone (1–16) |

Montgomery (1997) found the volume ratios (`SSTRVOL`, `CPWVOL5`) more informative than distances.

**Competition variables are static** — computed from census data, they do **not vary** across the 280+ weeks. This is a fundamental constraint on modeling competitive dynamics (see Section 6).

#### Customer count file (`ccount.csv`)

Weekly store-level revenue by department (GROCERY, DAIRY, FROZEN, MEAT, DELI, BAKERY, BEER, WINE, SPIRITS, etc.) plus customer counts (`CUSTCOUN`). Useful for store-level demand normalization.

### Product categories (29 total)

From the SAS conversion script, the categories are: `ana` (analgesics), `bat` (bath), `ber` (beer), `bjc` (bottled juice), `cer` (cereals), `che` (cheese), `cig` (cigarettes), `coo` (cookies), `cra` (crackers), `cso` (canned soup), `did` (dish detergent), `fec` (fabric conditioner/softener), `frd` (frozen dinners), `fre` (frozen entrees), `frj` (frozen juice), `fsf` (front-end snacks/front-end candy), `gro` (grooming), `lnd` (laundry detergent), `oat` (oatmeal), `ptw` (paper towels), `rfj` (refrigerated juice — in SAS only, not in CSV exports), `sdr` (soft drinks), `sha` (shampoo), `sna` (snacks), `soa` (soap), `tbr` (toothbrush), `tna` (canned tuna), `tpa` (toothpaste), `tti` (toilet tissue).

### Recommended starting categories

Select categories with rich substitution structure for initial modeling:
- **Beer** (`ber`): 29 sub-categories, high cross-elasticity, ~791 UPCs, ~125K movement rows
- **Soft drinks** (`sdr`): brand competition, high promotional activity
- **Canned soup** (`cso`): moderate SKU count (~20–30), clear substitution patterns — recommended first target

---

## 3. Preprocessing Pipeline

### Step-by-step procedure

1. **Load and filter**: Read category movement CSV. Drop all rows with `OK == 0` or `PRICE ≤ 0`. Drop `PRICE_HEX` and `PROFIT_HEX` columns. Optionally drop `MOVE ≤ 0` if modeling positive-demand periods only.

2. **Compute unit price**: `unit_price = PRICE / QTY`. Compute wholesale cost: `cost = PRICE × (1 − PROFIT/100) / QTY`.

3. **Merge product metadata**: Join on `UPC` to obtain `DESCRIP`, `SIZE`, `COM_CODE`, `NITEM`.

4. **Merge temporal metadata**: Map `WEEK` to calendar dates via week decode table. Derive `year`, `month`, `quarter`, `week_of_year`, and binary holiday indicators (Thanksgiving, Christmas, Easter, Memorial Day, Labor Day).

5. **Merge store demographics**: Join on `STORE` to obtain all demographic and competition variables.

6. **Handle missing weeks**: For each active (store, UPC) pair — defined as appearing in ≥10% of that store's operating weeks — insert explicit zero-sales rows for missing weeks. This distinguishes genuine zero demand from product non-availability.

7. **Promotion enhancement** (the SALE field problem): The SALE field has documented false negatives — if set it indicates a promotion, but if blank, a promotion may still have occurred. Fix: compute **modal price** (most frequent price) per (UPC, STORE) as the regular price. Flag any week where `unit_price < 0.95 × modal_price` as promoted regardless of SALE value. Derive:
   - `on_promotion` (binary)
   - `discount_depth = (regular_price − unit_price) / regular_price`

8. **Feature engineering**:
   - Apply **symlog** to `MOVE` and `unit_price` for world model encoder input
   - Compute `price_index = unit_price / category_mean_price` for cross-category comparability
   - Lag features: `lag_price_1`, `lag_price_2`, `lag_move_1`, `lag_move_2`
   - Rolling features: `rolling_mean_move_4` (4-week rolling average), `rolling_std_price_4`
   - Encode store price tier (Cub-Fighter / Low / Medium / High) from `PRICLOW`, `PRICMED`, `PRICHIGH` columns in demographics

9. **Panel balancing**: Restrict to the 83 stores with complete histories (following Montgomery 1997).

10. **Temporal split** (strictly chronological, no random shuffling):
    - **Train**: weeks 1–280 (~70%, through mid-1995)
    - **Validation**: weeks 281–340 (~15%, 1995–1996)
    - **Test**: weeks 341–400 (~15%, 1996–1997)

### Dataset sizes for single category (canned soup, 83 stores, ~25 SKUs)

- Training: ~25 × 83 × 280 ≈ **581,000 tuples**
- Test: ~25 × 83 × 60 ≈ **124,500 tuples**

### Observation vector construction

At time *t* for a given store, concatenate:

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

With K=25 SKUs, the flat observation is 25 × 12 + 8 + 6 = **314 dimensions**. The entity-factored approach (Section 5) structures this more efficiently.

### Causal instrument construction (preprocessing stage)

```python
# Hausman IV: leave-one-out mean of log prices across other stores
df['log_price'] = np.log(df['PRICE'] / df['QTY'])
group_sum = df.groupby(['UPC','WEEK'])['log_price'].transform('sum')
n_stores = df.groupby(['UPC','WEEK'])['log_price'].transform('count')
df['hausman_iv'] = (group_sum - df['log_price']) / (n_stores - 1)

# Wholesale cost instrument (supplementary, noisy due to AAC)
df['log_cost'] = np.log((df['PRICE'] * (1 - df['PROFIT']/100)).clip(lower=0.01))

# 2sCOPE copula residual (robustness check, not primary)
from scipy.stats import norm, rankdata
df['price_star'] = norm.ppf(rankdata(df['log_price']) / (len(df) + 1))
# Residualize on copula-transformed controls
exog_cols = ['income','educ','age60','ethnic','workwom','hsizeavg']
W_stars = np.column_stack([norm.ppf(rankdata(df[c])/(len(df)+1)) for c in exog_cols])
reg1 = LinearRegression().fit(W_stars, df['price_star'].values)
df['copula_resid'] = df['price_star'].values - reg1.predict(W_stars)
```

---

## 4. World Model Architecture

### Core design: DreamerV3 RSSM with Mamba-2 backbone

The architecture follows the **DRAMA pattern** (Wang et al., arXiv 2410.08893, ICLR 2025) — the first published world model using Mamba-2. The critical design insight is the **decoupled posterior**: z_t depends only on x_t (not on h_t) during training, breaking the sequential dependency and enabling Mamba-2's parallel SSD scan.

### DRAMA-style wiring (recommended)

```
Encoder:       z_t = encode(x_t)              — posterior depends ONLY on observation
Seq input:     input_t = Linear(cat(z_t, a_t)) — project to d_model
Mamba-2:       d_t = Mamba2(input_{1:T})        — parallel during training
Prior head:    ẑ_{t+1} = MLP(d_t)              — predict next latent from det. state
Reward head:   r̂_t = MLP(d_t)
Decoder:       x̂_t = decode(z_t)              — reconstruct from z_t only
```

During imagination rollouts, the model switches to Mamba-2's recurrent `step()` function for single-step autoregressive generation.

### Why DRAMA-style over R2I-style

R2I (Samsami et al., ICLR 2024 Oral) preserves the dependency q(z_t | h_t, x_t) but achieves parallelism differently. R2I was explicitly tested on tabular environments (BSuite, POPGym), while DRAMA was evaluated only on Atari. However, Mamba-2's content-dependent gating may better model varying retail dynamics compared to R2I's S4/S5. **Recommended hybrid: R2I-style decoupled posterior with Mamba-2 backbone** (i.e., the DRAMA pattern — both approaches converge on the same decoupled-posterior architecture).

### Mamba-2 specifics

**Why Mamba-2 over GRU/Transformer:**
- Recurrent inference mode for imagination (constant memory per step)
- Content-dependent gating for non-stationary retail dynamics
- State representation that naturally encodes regime information
- **NOT for speed at T=64**: Mamba-2's SSD crosses over FlashAttention-2 at ~2K sequence length. At T=64, attention cost is 4,096 multiply-adds per head — trivial for modern GPUs. Use Mamba-2 for its **modeling properties**, not speed.

**Library interface:**

```python
from mamba_ssm import Mamba2
from mamba_ssm.utils.generation import InferenceParams

mamba_block = Mamba2(
    d_model=512,       # model dimension
    d_state=64,        # SSM state expansion factor
    d_conv=4,          # local convolution width
    expand=2,          # d_inner = expand × d_model = 1024
    headdim=64,        # nheads = d_inner // headdim = 16
    chunk_size=256,    # SSD chunk size
    rmsnorm=True,      # use RMSNorm before output
)

# Training (parallel): full sequences via SSD
y = mamba_block(x)  # x: (B, T, 512) → y: (B, T, 512)

# Inference (recurrent): single-step with maintained state
inference_params = InferenceParams(max_seqlen=1, max_batch_size=B)
y_t = mamba_block(x_t.unsqueeze(1), inference_params=inference_params)
```

**h_t is the output of the Mamba2 block (shape B × d_model), NOT the internal SSM state.** The internal SSM state (B × nheads × headdim × d_state) is managed internally by `step()`.

**Episode boundary handling:** The mamba-ssm library only exposes `conv_state` and `ssm_state` during step-by-step inference. During training (parallel scan), there is no built-in episode boundary mechanism. Options:
- (a) Ensure each training sequence contains only one store-year (simplest if T=64 fits)
- (b) Implement custom scan with continuation flags: `ssm_state *= (1 - done_flag)`
- (c) Adopt R2I's modified parallel scan operator
- During inference: zero out caches between store-year boundaries

### Stochastic latent specification

**32 independent categorical distributions, each over 32 classes**, yielding a 1024-dimensional one-hot vector (160 bits of information). Uses **1% uniform mixing ("unimix")** on ALL categorical distributions (both prior and posterior):

```python
probs = 0.99 × softmax(logits) + 0.01 / 32
```

This prevents near-deterministic distributions that destabilize KL computation. Unimix indirectly affects KL computation since both distributions are mixed before KL[q_mixed ∥ p_mixed] is computed.

For the entity-factored variant (Section 5): **8 categoricals × 8 classes per entity** (reduced from 32×32 since each entity carries less information than the full scene).

### Module structure

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
        self.reward_heads = nn.ModuleList([  # 5-head ensemble for MOPO LCB
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.SiLU(),
                nn.Linear(d_model, n_bins),
            ) for _ in range(5)
        ])
        self.continue_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.decoder = CausalDemandDecoder(...)  # See Section 9
```

### Model sizing

| Config | d_model | Total params | Training time (per seed) |
|--------|---------|-------------|-------------------------|
| Prototype | 256 | ~5M | Minutes |
| Size-S (recommended) | 512 | ~22M | Hours |
| Entity-factored | 128/entity | ~350K | <30 minutes |

---

## 5. Entity-Factored State Representation

### Why entity factorization

Concatenating all 25 SKUs × 12 features = 300 dimensions into a flat vector discards entity structure, cannot generalize to new SKUs, and forces the model to learn cross-product interactions implicitly. Entity-centric representation (inspired by SOLD, Object-Centric Dreamer, FIOC-WM) treats each SKU as a separate **slot** with shared dynamics parameters and models cross-SKU interactions via attention.

### Dual-attention architecture

Two attention mechanisms map to distinct economic phenomena:

- **Temporal self-attention** (within-entity across time): Each SKU slot attends to its own history, capturing trend, seasonality, stockpiling, and individual promotion response.
- **Relational cross-attention** (across-entities within timestep): All SKU slots at time *t* attend to each other, capturing substitution and complementarity. Attention weights are directly interpretable as learned substitution/complementarity structure.

### Entity embedding layer

```python
class EntityEncoder(nn.Module):
    def __init__(self, n_upcs=500, n_stores=93, n_continuous=12, d_slot=64):
        super().__init__()
        self.upc_embed = nn.Embedding(n_upcs, 32)
        self.store_embed = nn.Embedding(n_stores, 16)
        self.brand_embed = nn.Embedding(100, 16)
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

1. **Individual dynamics** (shared-parameter per entity): `h_t^k = BlockGRU(h_{t-1}^k, [z_{t-1}^k, a_t^k])` with `d_hidden=128`, 4 blocks of 32. Parameters shared across all K SKUs — entity embedding provides specialization.
2. **Relational cross-attention**: `H_t = MultiHeadAttention(Q=h_t, K=h_t, V=h_t)` with 4 heads × 32 dims. For K=25, the O(K²) cost is negligible.
3. **Prior/Posterior heads**: `p(z_t^k | H_t^k)` = MLP(H_t^k) → 8 categoricals × 8 classes per entity.

### Entity-factored hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Entity embedding dim | **256** | SOLD uses 256; sufficient for tabular features |
| Attention heads | **4** | SLATE uses 4 at dim=192; SOLD uses 8 at dim=256 |
| Key dimension d_k | **64 per head** | d_model/heads = 256/4 = 64 |
| Cross-attention layers | **1** with residual MLP | Tabular features don't need deep attention |
| Positional encoding | **None** — use learned entity ID embeddings | SKUs and stores are unordered sets; permutation invariance matters |

### Store conditioning via parameter sharing

Following Terry et al. (arXiv 2005.13625), shared world model parameters + store ID embedding is provably sufficient for store-specific dynamics. Store embeddings enter through the entity encoder, allowing the model to learn that Cub-Fighter stores have higher price sensitivity than High-tier stores without separate models.

### Scalability

For 25 SKUs × 93 stores with d_slot=64, d_hidden=128: ~350K parameters, <1 GB training footprint. Even all 93 stores simultaneously (~18 GB activations) fits in 128 GB unified memory.

---

## 6. Pricing as a POMDP with Latent Competition

### Formal MDP specification

| Component | Definition |
|-----------|-----------|
| **State** s_t | (own_prices, competitor_price_proxy, demand_history, promo_calendar, inventory_proxy, seasonality, store_demographics) |
| **Observation** o_t | Full state minus competitor costs, competitor inventory, competitor strategy intent |
| **Action** a_t | Per-SKU price adjustment Δp ∈ [−M, M], discretized to 21 levels (−10%, −8%, ..., 0, ..., +8%, +10%) for DQN, or continuous for SAC |
| **Transition** T(s'|s,a) | World model predicts next-week demand, competitor price proxy, category switching |
| **Reward** r_t | Σ_k [(price_k − cost_k) × units_sold_k] − λ_vol × Σ_k |Δprice_k| − λ_stock × stockout_count |

### Competition modeling: what is feasible (and what is not)

**What Dominick's provides for competition:** Four static variables per store (`SSTRDIST`, `SSTRVOL`, `CPDIST5`, `CPWVOL5`) computed from census data. These do **not vary** across the 280 weeks. There is no time-varying competitor action data.

**What is NOT feasible given this data:**

| Proposed approach | Verdict | Reason |
|---|---|---|
| Auxiliary follower policy network | **Overclaim** | No competitor action data to train against |
| Stackelberg POMDP (Brero et al.) | **Overclaim** | Addresses mechanism design, not retail pricing; requires observed follower actions |
| Mean Field Game dynamics | **Overclaim** | No observable population distribution; extreme complexity |
| CTDE (centralized training decentralized execution) | **Overclaim** | Single-agent problem; competitors are environment, not teammates |
| LOLA / DRON / SOM | **Overclaim** | All require time-varying opponent observations |

**What IS feasible: competition-conditioned RSSM with robust imagination.**

The RSSM's stochastic latent z_t naturally captures unobserved factors including competitor behavior. When a rival runs a promotion, Dominick's observes unexplained demand drops; the posterior infers a latent state consistent with that observation. Static demographics condition the prior, so stores with closer/larger competitors learn higher-variance dynamics:

```python
competitive_alpha = sigmoid(
    w1 * (1/SSTRDIST) + w2 * SSTRVOL +
    w3 * (1/CPDIST5) + w4 * CPWVOL5
)

# During imagination: sample from pessimistic prior for high-competition stores
z_prior = world_model.prior(h_t)
if competitive_robust:
    z_pessimistic = shift_toward_low_reward_states(z_prior, reward_model)
    z_t = (1 - competitive_alpha) * z_prior + competitive_alpha * z_pessimistic

# Policy robustness: penalize high value variance across latent samples
value_samples = [critic(h_t, z_sample) for z_sample in z_samples]
robust_loss = actor_loss + lambda_robust * var(value_samples)
```

### Reward function with degenerate strategy prevention

```python
def compute_reward(price, cost, units_sold, prev_price,
                   lambda_vol=0.05, lambda_margin_floor=0.10):
    gross_margin = ((price - cost) * units_sold).sum(dim=-1)
    volatility_penalty = lambda_vol * (price - prev_price).abs().sum(dim=-1)
    margin_pct = (price - cost) / price.clamp(min=0.01)
    floor_penalty = F.relu(lambda_margin_floor - margin_pct).sum(dim=-1)
    return gross_margin - volatility_penalty - floor_penalty
```

The **volatility penalty** prevents excessive price churn. The **margin floor penalty** prevents race-to-bottom strategies. Both are essential — without them, the agent discovers degenerate policies (perpetual deep discounts to maximize unit volume at negligible margin).

### Counterfactual reasoning

World models naturally support counterfactual queries: given observed state s_t, substitute a different price action a'_t and roll forward. This is equivalent to do(price = p) when the world model has correctly learned causal structure. The key limitation: observationally-trained world models learn P(demand | price, state) which may conflate causal and confounded relationships — Section 9 addresses this.

---

## 7. Three-Phase Training Loop

### Phase A: World Model Learning

Every training step samples **B = 32 sequences of length T = 64** from the replay buffer (the entire offline dataset). Larger B=32 (vs. DreamerV3's B=16) leverages the 128GB unified memory.

**World model loss:**

```
L(φ) = β_pred × L_pred + β_dyn × L_dyn + β_rep × L_rep
     = 1.0 × L_pred   + 0.5 × L_dyn   + 0.1 × L_rep
```

**Prediction loss** L_pred sums three components:
- **Decoder loss**: symlog squared error `½(f(x,θ) − symlog(y))²` for continuous observations
- **Reward loss**: categorical cross-entropy against twohot-encoded symlog-transformed rewards over 255 bins in [−20, +20]
- **Continue loss**: binary cross-entropy for store closures / product discontinuations

**Dynamics loss** `L_dyn = max(1, KL[sg(posterior) ∥ prior])` — trains only the prior (dynamics predictor) by stop-gradienting the posterior. Free bits threshold of 1 nat.

**Representation loss** `L_rep = max(1, KL[posterior ∥ sg(prior)])` — trains only the encoder. The 5:1 asymmetry (β_dyn = 0.5 vs β_rep = 0.1) ensures the prior is pushed hard toward the posterior while applying only gentle pressure on the posterior toward simplicity, preventing posterior collapse.

**Optimizer**: Adam at lr = 1×10⁻⁴, global gradient norm clipping at 1000.

### Phase B: Actor Training in Imagination

From encoded initial states, unroll **H = 13 imagination steps** (one retail quarter, aligned with the 4-5-4 retail calendar):

1. Sample action: a_t ~ π_θ(a_t | h_t, z_t)
2. Advance: h_{t+1} = Mamba2.step(Linear(cat(z_t, a_t)))
3. Sample prior: z_{t+1} ~ p(z_{t+1} | h_{t+1})
4. Predict: r_t (with MOPO pessimism, see Section 10), c_t from heads

**Lambda-returns** computed backward from horizon:

```
R^λ_T = v_ψ(s_T)                                        [bootstrap]
R^λ_t = r_t + γ·c_t·((1−λ)·v_ψ(s_{t+1}) + λ·R^λ_{t+1})
```

with **γ = 0.95** (adjusted from DreamerV3's 0.997 for retail quarterly planning; effective horizon ~20 weeks), **λ = 0.95**.

**Percentile return normalization**: `R_norm = R^λ / max(1, P95 − P5)` with EMA decay 0.99.

**Actor loss**: REINFORCE (discrete actions) or reparameterization (continuous) + entropy regularization at η = 3×10⁻⁴.
**Optimizer**: Adam at lr = 3×10⁻⁵, gradient norm clipping at 100.

### Phase C: Critic Training in Imagination

Critic predicts return distribution via **twohot discrete regression** over 255 bins in [−20, +20] symlog space. Loss: categorical cross-entropy against soft twohot labels.

**EMA slow critic** (decay = 0.98) provides regularization via cross-entropy between fast and slow critic outputs (weight 1.0).

**Optimizer**: Adam at lr = 3×10⁻⁵, gradient norm clipping at 100.

### Complete hyperparameter table

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size B | **32** | Increased from DreamerV3's 16 to leverage 128GB memory |
| Sequence length T | 64 | DreamerV3 default |
| WM learning rate | 1×10⁻⁴ | DreamerV3 |
| Actor learning rate | 3×10⁻⁵ | DreamerV3 |
| Critic learning rate | 3×10⁻⁵ | DreamerV3 |
| Imagination horizon H | **13** | Aligned with retail quarter (4-5-4 calendar) |
| Discount γ | **0.95** | Adjusted for retail; effective horizon ~20 weeks |
| Lambda λ | 0.95 | DreamerV3 |
| Entropy scale η | 3×10⁻⁴ | DreamerV3 |
| β_pred / β_dyn / β_rep | 1.0 / 0.5 / 0.1 | DreamerV3 |
| Free bits | 1.0 nat | DreamerV3 |
| Unimix | 0.01 | DreamerV3, applied to all categoricals |
| Twohot bins | 255, range [−20, +20] | DreamerV3 |
| Critic EMA decay | 0.98 | DreamerV3 |
| Return norm percentiles | 5th, 95th | DreamerV3 |
| Return norm EMA decay | 0.99 | DreamerV3 |
| Replay capacity | Full offline dataset | All 581K tuples |
| Gradient clip (WM) | 1000 global norm | DreamerV3 |
| Gradient clip (actor/critic) | 100 global norm | DreamerV3 |
| Total gradient steps | 100K–200K | ~176–360 epochs over 581K tuples |
| Weight decay | 1e-4 | Regularization for offline setting |
| Dropout (encoder/decoder MLPs) | 0.1 | Regularization for offline setting |
| Mixed precision | BF16 forward/backward, FP32 master weights/optimizer | SSM state transitions in FP32 for numerical stability |

### Offline setting modifications

1. Replay buffer is the entire fixed offline dataset — no new data ever added
2. No environment interaction step
3. Training ratio replaced by fixed iteration count (100K–200K gradient steps)
4. Stored latent states refreshed during training to maintain consistency
5. **Replay sampling**: 70% uniform across quarterly temporal strata, 30% overweighting most recent 2 years (hybrid per DEER 2025 findings)

---

## 8. Core Technical Mechanisms

### Symlog transform

`symlog(x) = sign(x) · ln(|x| + 1)`, inverse: `symexp(x) = sign(x) · (exp(|x|) − 1)`

Applied to: (a) encoder inputs, (b) decoder targets, (c) reward prediction targets, (d) critic targets (via twohot).

Critical for retail data: weekly demand 0–10,000+ units, prices $0.29–$15.99, revenue cents to hundreds of thousands. Unlike log, symlog is defined at zero, symmetric, and differentiable everywhere.

```python
def symlog(x): return torch.sign(x) * torch.log1p(torch.abs(x))
def symexp(x): return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
```

### Twohot encoding for distributional prediction

For scalar target x in symlog space between bins b_k and b_{k+1}: `weight_k = |b_{k+1} − x| / |b_{k+1} − b_k|`, `weight_{k+1} = 1 − weight_k`. Predicted value recovered as `symexp(Σ_i probs_i × b_i)`.

With 255 bins uniformly in symlog [−20, +20], representable range ≈ ±4.85 × 10⁸, increasing resolution near zero. Cross-entropy against twohot labels provides richer gradients than MSE and handles bimodal distributions (zero demand weeks vs. promotional spikes).

### Straight-through gradients for discrete latents

```python
one_hot = F.one_hot(dist.sample(), num_classes=n_cls).float()
z = one_hot + probs - probs.detach()  # forward: one_hot; backward: probs
```

Simpler than Gumbel-Softmax (no temperature annealing), produces genuinely discrete representations.

### KL balancing

The 5:1 asymmetry (β_dyn=0.5 for dynamics loss with sg(posterior), β_rep=0.1 for rep loss with sg(prior)) prevents posterior collapse. Without it, the encoder ignores observations and outputs the prior. The free bits mechanism (clipping at 1 nat) further protects against collapse by disabling KL when it reaches a healthy floor.

### Percentile return normalization

`R_norm = R^λ / max(1, P95 − P5)` — the `max(1, ·)` ensures returns are **never amplified**, only compressed. Under sparse rewards, returns pass through unscaled, preserving entropy regularizer's dominance. Strictly superior to mean/std normalization.

---

## 9. Causal Identification & Endogeneity Correction

### The problem

Price is endogenous in observational data: store managers change prices partly in response to anticipated demand shocks. A naively-trained world model learns P(demand | price, state) which conflates causal effects with confounding. An optimizing agent will then exploit these biased predictions, generating counterfactuals where raising prices increases demand.

OLS-estimated price elasticity is severely attenuated (e.g., −1.4 when true causal effect is −2.5) because managers raise prices when they anticipate high demand (positive confounding).

### Primary method: Hausman IV + DML-PLIV

**Hausman instrument:** For each (UPC j, store s, week t), construct `Z_{j,s,t} = mean(log_price_{j,s',t})` for all s' ≠ s. This exploits common wholesale cost shocks across pricing zones while assuming local demand shocks are independent across stores after controlling for fixed effects.

With 83 stores, first-stage F-statistic **well above 100** — far exceeding the weak instrument threshold of 10.

**Exclusion restriction threat:** Chain-wide promotional campaigns create correlated demand shocks. **Mitigation**: include week × category fixed effects and cluster standard errors by week.

**DML-PLIV (most principled combined approach):**

```python
import doubleml as dml

data_dml = dml.DoubleMLData(df, y_col='log_move', d_cols='log_price',
                             x_cols=control_cols, z_cols='hausman_iv')
dml_pliv = dml.DoubleMLPLIV(data_dml, ml_l=RandomForestRegressor(),
                             ml_m=RandomForestRegressor(),
                             ml_r=RandomForestRegressor(), n_folds=5)
dml_pliv.fit()
theta_causal = dml_pliv.coef_[0]  # expect -2.0 to -3.0 for grocery
```

### Robustness check: 2sCOPE

The copula correction transforms price to a standard normal quantile and appends it as a control variable. Identification requires non-normality of the endogenous regressor (retail prices with promotional spikes typically satisfy this). **But:** if regressors are approximately normal, identification collapses entirely. The Gaussian copula assumption is untestable.

**Role:** Robustness check only, not primary method. If DML-PLIV and 2sCOPE estimates diverge substantially, investigate whether correlated demand shocks (threatening Hausman) or non-Gaussianity (threatening copula) is the issue.

### Validation checklist

- First-stage F > 10 (expect > 100)
- Hausman test rejects OLS = IV
- 2sCOPE β_c significant
- Sargan overidentification test (using both Hausman IV and wholesale cost)
- Compare world model counterfactual predictions across approaches

### Integration into the world model: constrained decoder (recommended)

Pre-estimate per-category θ̂ via DML-PLIV. The demand decoder separates the **fixed causal price channel** from the **learnable latent-state residual**:

```python
theta_per_cat = {'analgesics': -2.31, 'cereals': -2.88, 'frozen_juice': -1.95, ...}

class CausalDemandDecoder(nn.Module):
    def __init__(self, theta_dict, latent_dim, num_categories):
        self.theta = nn.Embedding(num_categories, 1)
        self.theta.weight.data = torch.tensor([[v] for v in theta_dict.values()])
        self.theta.weight.requires_grad = False  # FROZEN from DML-PLIV
        self.residual = MLP(latent_dim + num_store_features, 256, 1)

    def forward(self, z_t, log_price, category_id, store_features):
        causal_price_effect = self.theta(category_id) * log_price
        residual_demand = self.residual(torch.cat([z_t, store_features], -1))
        return causal_price_effect + residual_demand
```

This ensures the world model cannot learn a confounded price-demand relationship during imagination. The residual network captures seasonality, promotional effects, store-level heterogeneity, and competitive dynamics through the latent state.

**Alternative approaches (ranked by rigor):**

1. **Constrained decoder** (above) — most principled, recommended
2. **Gradient regularization** — allow decoder to learn its own elasticity but penalize deviation from DML estimate: `causal_reg = λ_c * ((∂Y/∂D − θ_causal)²).mean()`
3. **Observation enrichment** — append `hausman_iv` and `copula_resid` to observation vector; theoretically weakest but requires minimal architectural changes

### Expected elasticity ranges

For Dominick's grocery categories, based on Nevo (2001) and Hoch et al. (1995): true elasticities in the **−2.0 to −3.0 range**. Chernozhukov et al. (2017) found category-level elasticities from −2.71 (sodas) to −0.4 (tableware) on similar scanner data.

---

## 10. Offline Learning: Pessimism & Distribution Shift

### The distributional shift problem

The Dominick's dataset records outcomes of Dominick's actual pricing policies — not exploratory random policy. Some (state, action) regions are severely underrepresented: extreme price cuts for premium products, unusual promotional combinations, counter-seasonal pricing. A naively-trained world model makes unreliable predictions in these gaps, and a policy optimizer exploits exactly these unreliable predictions ("model exploitation").

### Recommended solution: MOPO-style LCB (not COMBO)

**Why not COMBO:** COMBO adds CQL-style regularization requiring significant modifications to DreamerV3's critic training loop — mixing real transitions with imagined ones, implementing push-down/push-up terms, and tuning β. Error-prone for a solo researcher.

**Why not MOReL:** Its hard binary HALT partition creates discontinuities in the value function. For continuous economic data with smooth price-demand relationships, soft penalties are strictly preferable.

**Why not MC dropout with Mamba-2:** SSM architectures maintain recurrent hidden states sensitive to perturbation. MC dropout is unreliable for out-of-distribution data — the exact regime that matters for offline RL.

**MOPO's reward penalty integrates trivially with DreamerV3** — modify only the reward signal during imagination:

```python
# 5 independent reward heads on shared RSSM backbone
for t in range(H):
    z_t = rssm.imagine(h_t, z_{t-1}, a_{t-1})
    r_preds = [reward_head_k(z_t, a_t) for k in range(5)]
    r_mean = torch.stack(r_preds).mean(dim=0)
    r_std = torch.stack(r_preds).std(dim=0)
    r_pessimistic = r_mean - lambda_lcb * r_std  # Lower Confidence Bound
```

### Calibrating λ_lcb

Normalize rewards to zero mean and unit variance using offline dataset statistics. Start with **λ = 1.0** (one standard deviation of pessimism). For gross margins averaging ~$2.50 with ensemble std ~$0.50 on in-distribution data: ~20% penalty — reasonable conservatism.

Search λ ∈ {0.25, 0.5, 1.0, 2.0, 5.0}; validate via offline policy evaluation on held-out data.

### Non-stationarity across 8 years (1989–1997)

Combine **context-conditioned modeling** (include year, quarter, macroeconomic indicators as state features) with **hybrid replay sampling** (70% uniform across quarterly strata, 30% overweighting most recent 2 years).

Optional: contrastive predictive coding auxiliary loss (per COSPA, Ackermann et al. 2024) to encourage latent state to encode regime information.

---

## 11. Experiment Design

### World model quality metrics

Evaluate at horizons **h = {1, 5, 10, 13, 25}** weeks:
- RMSE and MAE on demand (original units via symexp)
- WMAPE for portfolio-level accuracy
- CRPS for distributional calibration
- **Normalized Degradation Rate** NDR(h) = RMSE(h)/RMSE(1) — should grow sub-linearly
- Linear probe on frozen latent representations → predict product category, season, promotional status
- UMAP visualization colored by category and price regime

### RL agent performance metrics

- **Primary**: cumulative gross margin over test period (weeks 341–400) across all stores
- **Secondary**: policy regret vs. oracle with true demand elasticities, response latency (cross-correlation lag), price volatility (std of week-over-week changes), margin variance

### Baseline hierarchy

1. **Cost-plus fixed markup** (25%): lower bound, zero intelligence
2. **Static optimization**: XGBoost demand model + per-week price optimization (strongest non-RL baseline)
3. **Competitive matching**: price = category_avg ± 2%
4. **DQN** with discretized price grid (21 levels)
5. **PPO** (on-policy model-free)
6. **SAC** (off-policy model-free, continuous actions)

### Ablation priority order

| Priority | Ablation | Tests |
|----------|----------|-------|
| 1 | Imagination ON vs. OFF (model-free actor-critic from data only) | Core world model contribution |
| 2 | Imagination horizon sweep {5, 10, 13, 25} | Optimal planning depth |
| 3 | Stochastic vs. deterministic-only latent | Uncertainty modeling value |
| 4 | With vs. without symlog + twohot | Reward scale robustness |
| 5 | GRU vs. Mamba-2 backbone | Long-range memory benefit |
| 6 | Entity embeddings vs. flat concatenation | Structural representation value |
| 7 | With vs. without MOPO LCB conservatism | Offline safety value |

### Statistical protocol

- **10 seeds** per main configuration and each baseline
- **5 seeds** per ablation
- Report IQM (interquartile mean) with 95% stratified bootstrap CIs (Agarwal et al., NeurIPS 2021)
- Paired bootstrap tests for ablation comparisons (same seeds)
- Holm-Bonferroni correction for 7-way ablation comparison

### Compute budget on DGX Spark

| Phase | Runs | Hours/run | Total hours |
|-------|------|-----------|-------------|
| Main model (10 seeds) | 10 | 8–16 | 80–160 |
| 6 baselines (10 seeds each) | 60 | 3–8 | 180–480 |
| 7 ablations (5 seeds each) | 35 | 6–12 | 210–420 |
| **Total** | **105** | — | **470–1,060** |
| **Wall-clock (sequential)** | — | — | **20–44 days** |

Entity-factored model (350K params) trains in <30 minutes per seed — enabling rapid iteration before scaling to full evaluation.

### Visualization toolkit

- **Policy heat maps**: own_price × competitor_proxy → action
- **Imagination rollout comparisons**: real vs. imagined demand trajectories
- **Learned demand response curves**: world model's D(p) vs. empirical D(p)
- **Relational attention matrices**: learned substitution structure between SKUs
- **Training curves**: cumulative margin vs. gradient steps, shaded ±1 std across seeds

---

## 12. Software Architecture & Packaging

### Package design

First pip-installable world model codebase. Build backend: **Hatchling** (pure-Python, no compiled CUDA extensions in the package itself). Do not list `torch`, `mamba-ssm`, or `triton` as hard dependencies — they require CUDA version matching. Place in optional dependencies with documented install sequence.

### Module structure

```
src/retail_world_model/
├── data/                          # Loading & preprocessing
│   ├── dominicks_loader.py        # Dominick's dataset I/O
│   ├── copula_correction.py       # Endogeneity correction (preprocessing)
│   ├── transforms.py              # Feature engineering, symlog, normalization
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
├── envs/                          # Gymnasium-compatible environments
│   ├── base.py                    # Base environment interface
│   └── grocery.py                 # Grocery pricing environment
└── utils/
    ├── distributions.py           # OneHotDist with unimix, symlog/twohot
    ├── checkpoint.py              # Safetensors + HF Hub
    └── device.py                  # CUDA/mixed-precision management
```

### Top-level project structure

```
dreamprice/
├── pyproject.toml              # PEP 621, hatchling backend
├── CONTRIBUTING.md
├── configs/                    # Hydra config hierarchy
│   ├── world_model/            # mamba.yaml, gru.yaml
│   ├── agent/                  # dreamer.yaml, ppo.yaml, sac.yaml
│   ├── environment/            # grocery.yaml
│   └── experiment/             # Full reproducible configs
├── src/retail_world_model/     # pip-installable package (structure above)
├── scripts/                    # train.py, evaluate.py, play.py
├── tests/                      # pytest suite
├── notebooks/                  # quickstart, custom env, analysis
├── docs/                       # Research documents, data samples
└── docker/
```

### Code quality

- Use **ruff** with C901 rule for McCabe cyclomatic complexity
- Use **complexipy** (Rust-backed, PyO3/maturin) for cognitive complexity
- Use **pyright** / **mypy** for type checking
- Configuration management via Hydra
- Logging via W&B

### Public API surface

```python
from retail_world_model import WorldModel, RLAgent, PricingEnvironment, Trainer
```

Environments follow Gymnasium's `reset()`/`step()` interface. World models add `predict()`, `rollout()`, and `imagine()` methods. Model checkpoints on HuggingFace Hub with CC-BY-NC-4.0 license.

---

## 13. API & Serving

### FastAPI endpoints

For rollouts taking 100ms–1s, use an asyncio queue-based dynamic batcher (max batch size 8, max wait 50ms):

| Endpoint | Method | Description | Latency |
|----------|--------|-------------|---------|
| `/health` | GET | Health check | <5ms |
| `/predict` | POST | Single-step demand prediction | <50ms |
| `/rollout` | POST | Full trajectory rollout | 100ms–1s |
| `/rollout/stream` | POST | SSE streaming (per-step JSON events) | Streaming |
| `/predict/batch` | POST | Offline evaluation batch | Variable |
| `/model/info` | GET | Model metadata | <5ms |

SSE streaming via `sse-starlette` for long-horizon rollouts.

---

## 14. Demo & Visualization

### React playground

A React-based RL playground showing an agent learning to price in real time:

1. Show trained world model simulating realistic market dynamics
2. Drop in an untrained RL agent — watch random exploration and losses
3. Fast-forward imagination training — agent discovers strategies in minutes
4. Deploy trained agent reacting to competitor proxy changes, demand shifts, seasonality
5. User plays as "competitor" in adversarial mode

### Key visual elements

- Real-time price chart (agent vs. baseline/competitor proxy)
- Animated market share visualizations
- Reward curves with confidence bands
- Policy heatmap: what price given different (demand, competitor_proxy) states
- World model accuracy panel: predicted vs. actual demand
- Interactive toggle between real data and imagined trajectories (inspired by DIAMOND's play mode)

### Researcher dashboard (Streamlit)

- Live training curves
- Rollout visualization
- Hyperparameter panels
- W&B integration
- Latent space UMAP visualizations

---

## 15. Paper Positioning & Framing

### Title

**"DreamPrice: A Learned World Model for Retail Pricing Environments"**

Connects to Dreamer lineage, is memorable, clearly scoped, avoids overclaiming.

### Contribution triple

1. First learned world model for pricing (novelty)
2. Gymnasium-compatible environment release (infrastructure)
3. Empirical analysis showing sample-efficient policy learning vs. model-free baselines (evidence)

### Framing

"New application domain + open-source benchmark" contribution, NOT a new architecture paper.

### Related work lineages

1. World models and model-based RL (Ding et al. 2025 ACM survey, Moerland et al.)
2. RL for dynamic pricing (Liu et al. Alibaba, Kastius & Schlosser)
3. Time series foundation models (Chronos, TimesFM — prediction vs. counterfactual reasoning)
4. Digital twins and learned simulators (ABIDES-Gym, Reinforcement Twinning)
5. The Dreamer lineage (V1/V2/V3, R2I, DRAMA)

### Most novel contribution

The **causal constrained decoder**: freezing per-category price elasticities from DML-PLIV while the residual network learns everything else from latent state. First proposed integration of econometric causal identification directly into a DreamerV3-style decoder. Bridges empirical IO demand estimation and model-based RL.

### Required baselines

Model-free RL (PPO, SAC, DQN), heuristic pricing (fixed markup, competitive matching), time series + optimization (XGBoost → price optimizer), simpler world models (linear dynamics, MLP ensemble), oracle with ground-truth elasticities.

### Required ablations

Architecture variants, latent dimension sizes, training data volume (10/25/50/100%), rollout horizon, stochastic vs. deterministic dynamics.

### Model card requirements

- Temporal scope: 1989–1997
- Geographic scope: Chicago metro area
- Product scope: ~3,500 UPCs in grocery
- Caveat: predictions may not generalize to modern retail environments
- License: CC-BY-NC-4.0
- Kilts Center acknowledgment

---

## 16. Execution Roadmap

### 13-week timeline: zero to arXiv

| Weeks | Milestone |
|---|---|
| 1–2 | Data pipeline: Dominick's loader, preprocessing, feature engineering, causal instrument construction (Hausman IV, DML-PLIV elasticity estimation) |
| 3–4 | World model: RSSM + Mamba-2 backbone, entity-factored encoder, causal constrained decoder, 5-head reward ensemble |
| 5–6 | Training: three-phase loop, MOPO LCB integration, Gymnasium wrapper, DreamerV3-style actor-critic |
| 7–8 | Baselines: model-free RL (DQN, PPO, SAC), heuristic pricing, XGBoost + optimization |
| 9–10 | Ablation studies: all 7 ablations × 5 seeds, statistical analysis |
| 11–12 | Paper writing, React demo, code polish, open-source packaging |
| 13 | Submit to arXiv, target ICML/NeurIPS 2026 workshop |

### Key decision points

1. **After week 2**: Validate causal elasticity estimates. If DML-PLIV and 2sCOPE diverge by >50%, investigate instrument validity before proceeding.
2. **After week 4**: Validate world model prediction quality. If NDR(13) > 5×NDR(1), investigate model capacity and training stability.
3. **After week 6**: Compare imagination-based agent vs. static optimization baseline. If no statistically significant margin improvement, investigate reward function calibration and pessimism level.

---

## Appendix: Data File Reference

### Files in `docs/data/`

| File | Description |
|------|-------------|
| `demo.csv` | Store demographics (300+ columns, 93+ rows) |
| `ccount.csv` | Customer counts and department revenue by store-week |
| `sas2csv.sas` | SAS macro for converting original SAS datasets to CSV |

### Category files in `docs/data/category/`

Movement files (`wXXX.csv`): `wana`, `wbat`, `wber`, `wbjc`, `wcer`, `wche`, `wcig`, `wcoo`, `wcra`, `wcso`, `wfec`, `wfrd`, `wfre`, `wfrj`, `wfsf`, `wgro`, `wlnd`, `woat`, `wptw`, `wsdr`, `wsha`, `wsna`, `wsoa`, `wtbr`, `wtna`, `wtpa`, `wtti`

UPC files (`upcXXX.csv`): `upcana`, `upcbat`, `upcber`, `upcbjc`, `upccer`, `upcche`, `upccig`, `upccoo`, `upccra`, `upccso`, `upcdid`, `upcfec`, `upcfrd`, `upcfre`, `upcfrj`, `upcfsf`, `upcgro`, `upclnd`, `upcoat`, `upcptw`, `upcsdr`, `upcsha`, `upcsna`, `upcsoa`, `upctbr`, `upctna`, `upctpa`, `upctti`

### Movement file schema (confirmed from `wber.csv`)

```
STORE, UPC, WEEK, MOVE, QTY, PRICE, SALE, PROFIT, OK, PRICE_HEX, PROFIT_HEX
```

### UPC file schema (confirmed from `upcber.csv`)

```
COM_CODE, UPC, DESCRIP, SIZE, CASE, NITEM
```

### Key data statistics (beer category sample)

- ~791 unique UPCs
- ~125,000 movement records in sample file
- COM_CODE values 26–27 for beer sub-categories
- Size strings: "12/12O" (12-pack 12oz), "24/12O" (24-pack), "259 OZ" (keg), etc.

---

*This document supersedes `research-1.md`, `research-2.md`, `research-final.md`, and `Proposal Review and Implementation Plan.txt`. All validated information has been incorporated; all overclaims and infeasible proposals have been removed.*
