# DreamPrice

**A Learned World Model for Retail Pricing via Mamba-2 Recurrence and Causal Demand Identification**

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/main.pdf)
[![Demo](https://img.shields.io/badge/Demo-Gradio-orange)](https://huggingface.co/spaces/qbz506/dreamprice-demo)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/qbz506/dreamprice-dominicks-cso)
[![Model](https://img.shields.io/badge/Model-HuggingFace-blue)](https://huggingface.co/qbz506/dreamprice-cso)
[![Wandb](https://img.shields.io/badge/Experiments-Wandb-green)](https://wandb.ai/qbz506-technektar/dreamprice)

DreamPrice is the first learned world model for retail pricing environments. It combines DreamerV3's three-phase training recipe with a Mamba-2 SSM backbone (DRAMA-style decoupled posterior), entity-factored multi-SKU representation, Hausman IV causal identification, and MOPO-style offline pessimism for safe policy learning from historical data.

Trained on the Dominick's Finer Foods scanner dataset (1989--1997, 93 stores, ~18K UPCs), DreamPrice learns a latent dynamics model that captures substitution effects, promotional responses, and price elasticities across product categories.

**Author**: Sharath Sathish, University of York, York, UK

## Training Results (100K Steps, DGX Spark)

| Metric | Value |
|--------|-------|
| World Model ELBO | 22.44 |
| Reconstruction Loss | 0.001 |
| KL Divergence | 19.20 |
| Actor Return (mean) | 124.33 |
| Training Time | ~2.6 hours |
| Hardware | NVIDIA DGX Spark (GB10, 128 GB unified memory) |

### Policy Performance

| Method | Mean Return | IQM |
|--------|-------------|-----|
| Cost-plus (25%) | 54.8 | 54.8 |
| Static XGBoost | 87.2 | 85.6 |
| Competitive Matching | 42.1 | 41.5 |
| DQN | 68.9 | 65.2 |
| PPO | 76.4 | 72.8 |
| SAC | 82.3 | 79.6 |
| **DreamPrice** | **124.3** | **117.4** |

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
import numpy as np
from retail_world_model import PricingEnvironment

env = PricingEnvironment(
    world_model=None,       # uses log-linear demand fallback
    store_features=np.zeros(8),
    initial_obs=np.zeros(32),
    cost_vector=np.full(3, 1.5),
    n_skus=3,
    H=13,
)
obs, info = env.reset()
for _ in range(13):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.2f}, Prices: {info['prices']}")
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

Dominick's Finer Foods scanner data from the Kilts Center for Marketing, University of Chicago.

**Preprocessed dataset**: [qbz506/dreamprice-dominicks-cso](https://huggingface.co/datasets/qbz506/dreamprice-dominicks-cso)

Raw CSVs should be placed in `docs/data/`:

- `docs/data/cso/wcso.csv` -- weekly movement (price, quantity, promotions)
- `docs/data/cso/upccso.csv` -- UPC metadata
- `docs/data/demo.csv` -- store demographics
- `docs/data/ccount.csv` -- customer counts

Start category: **Canned Soup (`cso`)** (~25 SKUs, ~581K training tuples).

Temporal split (strictly chronological): Train weeks 1-280, Val 281-340, Test 341-400.

## Training

```bash
# Docker-based training (recommended)
docker compose run --rm dreamprice python scripts/train.py n_steps=100000

# Or with Hydra overrides
docker compose run --rm dreamprice python scripts/train.py n_steps=100000 wandb_project=dreamprice

# Evaluation
docker compose run --rm dreamprice python scripts/evaluate_world_model.py \
  --checkpoint checkpoints/step_0100000.pt --horizons 1 5 10 13 25
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| [quickstart.ipynb](notebooks/quickstart.ipynb) | Minimal end-to-end pricing environment demo |
| [world_model_training.ipynb](notebooks/world_model_training.ipynb) | World model architecture and training pipeline |
| [causal_analysis.ipynb](notebooks/causal_analysis.ipynb) | Hausman IV + DML-PLIV elasticity estimation |
| [custom_env.ipynb](notebooks/custom_env.ipynb) | Custom SKU environment configuration |

All notebooks include Colab badges for one-click execution.

## API Serving

```bash
python -m retail_world_model.api.serve
# Starts FastAPI server with dynamic batching (max batch 8, max wait 50ms)
```

## Project Structure

```
dreamprice/
├── src/retail_world_model/     # Core package
│   ├── models/                 # MambaWorldModel, RSSM
│   ├── training/               # DreamerTrainer, losses
│   ├── applications/           # ActorCritic policy
│   ├── inference/              # Imagination rollout
│   ├── envs/                   # GroceryPricingEnv
│   ├── data/                   # Dataset, loader, transforms
│   └── api/                    # FastAPI serving
├── scripts/                    # Training, evaluation, baselines
├── configs/                    # Hydra configs, elasticities
├── paper/                      # LaTeX paper source + figures
├── notebooks/                  # Jupyter notebooks
├── app/                        # Gradio demo app
├── tests/                      # 125 tests (pytest)
└── docs/                       # Data, results, documentation
```

## License

CC-BY-NC-4.0. Data from Kilts Center for Marketing, University of Chicago.

## Citation

```bibtex
@article{sathish2026dreamprice,
  title  = {DreamPrice: A Learned World Model for Retail Pricing via Mamba-2 Recurrence and Causal Demand Identification},
  author = {Sathish, Sharath},
  year   = {2026},
  url    = {https://github.com/SharathSPhD/dreamprice}
}
```
