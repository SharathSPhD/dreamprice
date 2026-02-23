# DreamPrice

**Learned World Model for Retail Pricing**

DreamPrice is the first learned world model for retail pricing environments. It combines DreamerV3's three-phase training recipe with a Mamba-2 SSM backbone (DRAMA-style decoupled posterior), entity-factored multi-SKU representation, Hausman IV causal identification, and MOPO-style offline pessimism for safe policy learning from historical data.

Trained on the Dominick's Finer Foods scanner dataset (1989--1997, 93 stores, ~18K UPCs), DreamPrice learns a latent dynamics model that captures substitution effects, promotional responses, and price elasticities across product categories. The causal decoder enforces econometrically-estimated elasticity constraints, preventing the model from learning confounded price-demand relationships.

## Quick Install

```bash
# CUDA-dependent packages first (match your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install mamba-ssm causal-conv1d

# Install DreamPrice
pip install -e ".[dev]"
```

## Quick Start

```python
from retail_world_model import WorldModel, RLAgent, PricingEnvironment, Trainer

env = PricingEnvironment(n_skus=3, episode_length=13)
obs, info = env.reset()
for _ in range(13):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

## Architecture

```
                        DreamPrice Architecture
  +-----------------------------------------------------------------+
  |                                                                 |
  |  Observation x_t                                                |
  |       |                                                         |
  |       v                                                         |
  |  [Encoder MLP] -----> z_t (32x32 categorical, straight-through)|
  |       |                    |                                    |
  |       |              [DRAMA decoupled posterior]                |
  |       |                                                         |
  |  [concat(z_t, a_{t-1})]                                        |
  |       |                                                         |
  |       v                                                         |
  |  [Mamba-2 Backbone] -----> h_t (d_model=512)                   |
  |       |    (parallel SSD scan during training,                  |
  |       |     recurrent step() during imagination)                |
  |       |                                                         |
  |       v                                                         |
  |  [Prior MLP] -----> z_hat_t (KL target for posterior)           |
  |       |                                                         |
  |       +-----> [CausalDemandDecoder] -----> x_hat_t              |
  |       |         (frozen DML-PLIV elasticities)                  |
  |       +-----> [RewardEnsemble x5] -----> r_t (MOPO LCB)        |
  |       +-----> [ContinueHead] -----> gamma_t                    |
  |                                                                 |
  +-----------------------------------------------------------------+
```

## Dataset

Dominick's Finer Foods scanner data from the Kilts Center for Marketing, University of Chicago. Place raw CSVs in `docs/data/`:

- `docs/data/{category}/w{category}.csv` -- weekly movement (price, quantity, promotions)
- `docs/data/{category}/upc{category}.csv` -- UPC metadata
- `docs/data/demo.csv` -- store demographics
- `docs/data/ccount.csv` -- customer counts

Start category: **Canned Soup (`cso`)** (~25 SKUs, ~581K training tuples).

Temporal split (strictly chronological): Train weeks 1-280, Val 281-340, Test 341-400.

## Training

```bash
# Configure via Hydra
python scripts/train.py experiment=configs/experiment/main.yaml

# With W&B logging
WANDB_PROJECT=dreamprice python scripts/train.py
```

## API Serving

```bash
python -m retail_world_model.api.serve
# Starts FastAPI server with dynamic batching (max batch 8, max wait 50ms)
```

## License

CC-BY-NC-4.0. Data from Kilts Center for Marketing, University of Chicago.

## Citation

```bibtex
@software{dreamprice2026,
  title  = {DreamPrice: Learned World Model for Retail Pricing},
  author = {Sharath S},
  year   = {2026},
  url    = {https://github.com/SharathSPhD/dreamprice}
}
```
