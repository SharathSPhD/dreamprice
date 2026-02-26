# DreamPrice

**A Causal DreamerV3 World Model for Offline Retail Pricing**

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/main.pdf)
[![Demo](https://img.shields.io/badge/Demo-Gradio-orange)](https://huggingface.co/spaces/qbz506/dreamprice-demo)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/qbz506/dreamprice-dominicks-cso)
[![Model](https://img.shields.io/badge/Model-HuggingFace-blue)](https://huggingface.co/qbz506/dreamprice-cso)
[![Wandb](https://img.shields.io/badge/Experiments-W%26B-green)](https://wandb.ai/qbz506-technektar/dreamprice)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

DreamPrice is a learned world model for retail pricing that combines DreamerV3's three-phase training recipe with a Mamba-2 SSM backbone, entity-factored multi-SKU representation, Hausman IV causal identification, and MOPO-style offline pessimism for safe policy learning from historical scanner data.

Trained on the Dominick's Finer Foods dataset (1989--1997, 93 stores, canned soup category), the system learns a latent dynamics model that captures substitution effects, promotional responses, and price elasticities, then trains a pricing policy entirely in imagination.

**Author:** Sharath Sathish, University of York, York, UK

---

## Key Results

### World Model Quality (100K Steps)

| Metric | Value |
|---|---|
| Total ELBO Loss | 22.44 |
| Reconstruction Loss | 0.001 |
| KL Divergence | 19.20 |
| Actor Return (symlog) | 193.7 |
| Training Time | ~4.4 hours |
| Hardware | NVIDIA DGX Spark (GB10, 128 GB unified) |

### Multi-Horizon Forecast Accuracy

| Horizon | RMSE | MAE | WMAPE |
|---|---|---|---|
| h = 1 | 5.00 | 2.13 | 0.717 |
| h = 5 | 5.17 | 2.17 | 0.723 |
| h = 13 (default) | 5.28 | 2.21 | 0.726 |
| h = 25 | 5.24 | 2.20 | 0.724 |

### Baseline Comparison

Two evaluation protocols are used: **data-replay** (raw gross margin on held-out test data) and **world-model** (episode return within the learned environment). The scales differ and are not directly comparable across protocol boundaries.

**Data-replay baselines** (mean weekly gross margin, $/wk):

| Method | Mean Return ($/wk) |
|---|---|
| Cost-plus (25% markup) | 33,690 |
| Competitive Matching | 45,020 |
| Static XGBoost | 148,623 |

**World-model baselines** (episode return, 13-step rollout):

| Method | Episode Return |
|---|---|
| PPO | −13,827 |
| SAC | −620 |
| DQN | 4,291 |
| **DreamPrice** | **193.7** (symlog + MOPO-LCB) |

DreamPrice uses symlog-transformed rewards with epistemic uncertainty penalties, yielding conservative returns that are not directly comparable to the raw-reward SB3 baselines. See the [paper](paper/main.pdf) for detailed discussion.

### Ablation Study (9 ablations, 100K steps each)

| Configuration | Return | WM Loss |
|---|---|---|
| Full DreamPrice | 193.7 | 22.44 |
| GRU Backbone | 137.7 | 19.89 |
| Flat Encoder | 68.6 | 22.52 |
| No Symlog/Twohot | 63.2 | 573.2 |
| No MOPO-LCB | 27.9 | 22.43 |
| Horizon 5 | 145.6 | 22.51 |
| Horizon 10 | 86.4 | 22.43 |
| Horizon 25 | 135.3 | 22.48 |
| Imagination OFF | --- | 22.47 |
| No Stochastic Latent | 289.6 | 5.32 |

---

## Architecture

```
                        DreamPrice Architecture
  +-----------------------------------------------------------------+
  |                                                                 |
  |  Observation x_t                                                |
  |       |                                                         |
  |       v                                                         |
  |  [Entity Encoder] ----> z_t (32x32 categorical, ST-GS)         |
  |       |                    |                                    |
  |       |              [DRAMA decoupled posterior]                |
  |       |                                                         |
  |  [concat(z_t, a_{t-1})]                                        |
  |       |                                                         |
  |       v                                                         |
  |  [Mamba-2 Backbone] ----> h_t (d_model=512)                    |
  |       |    (parallel SSD scan during training,                  |
  |       |     recurrent step() during imagination)                |
  |       |                                                         |
  |       v                                                         |
  |  [Prior MLP] ----> z_hat_t (KL target for posterior)            |
  |       |                                                         |
  |       +----> [CausalDemandDecoder] ----> x_hat_t                |
  |       |        (frozen DML-PLIV elasticities)                   |
  |       +----> [RewardEnsemble x5] ----> r_t (MOPO-LCB)          |
  |       +----> [ContinueHead] ----> gamma_t                      |
  |                                                                 |
  +-----------------------------------------------------------------+
```

---

## Installation

### Docker (recommended)

```bash
docker compose build
docker compose run --rm dreamprice bash
```

### Local

```bash
# CUDA-dependent packages (match your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install mamba-ssm causal-conv1d

# Install DreamPrice
pip install -e ".[dev]"
```

---

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

---

## Training

```bash
# Full training (100K steps, ~4.4 hours on DGX Spark)
docker compose run --rm dreamprice python scripts/train.py n_steps=100000

# With Hydra overrides
docker compose run --rm dreamprice python scripts/train.py \
  n_steps=100000 wandb_project=dreamprice seed=42

# Run ablation suite
docker compose run --rm dreamprice python scripts/run_ablations.py

# Evaluate world model
docker compose run --rm dreamprice python scripts/evaluate_world_model.py \
  --checkpoint checkpoints/step_0100000.pt --horizons 1 5 10 13 25

# Run baselines (data-replay + RL)
docker compose run --rm dreamprice python scripts/evaluate.py
```

---

## Notebooks

| Notebook | Description |
|---|---|
| [quickstart.ipynb](notebooks/quickstart.ipynb) | End-to-end pricing environment demo |
| [world_model_training.ipynb](notebooks/world_model_training.ipynb) | World model architecture and training pipeline |
| [causal_analysis.ipynb](notebooks/causal_analysis.ipynb) | Hausman IV + DML-PLIV elasticity estimation |
| [custom_env.ipynb](notebooks/custom_env.ipynb) | Custom SKU environment configuration |

All notebooks include Colab badges for one-click execution.

---

## API Serving

```bash
# FastAPI server with dynamic batching
python -m retail_world_model.api.serve

# Gradio demo (mirrors HuggingFace Space)
python app/app.py
```

---

## Dataset

The preprocessed dataset is hosted on HuggingFace:
[qbz506/dreamprice-dominicks-cso](https://huggingface.co/datasets/qbz506/dreamprice-dominicks-cso)

Source: Dominick's Finer Foods scanner data from the [Kilts Center for Marketing](https://www.chicagobooth.edu/research/kilts/datasets/dominicks), University of Chicago Booth School of Business.

- **Category:** Canned Soup (`CSO`), ~25 SKUs, ~581K training tuples
- **Panel:** 93 stores, weeks 1--400 (1989--1997)
- **Split:** Train 1--280, Validation 281--340, Test 341--400

---

## Project Structure

```
dreamprice/
├── src/retail_world_model/     # Core package
│   ├── models/                 #   MambaWorldModel, RSSM components
│   ├── training/               #   DreamerTrainer, loss functions
│   ├── applications/           #   ActorCritic policy
│   ├── inference/              #   Imagination rollout
│   ├── envs/                   #   GroceryPricingEnv + wrappers
│   ├── data/                   #   Dataset, dataloader, transforms
│   └── api/                    #   FastAPI serving
├── scripts/                    # Training, evaluation, baselines
├── configs/                    # Hydra configs, elasticities
├── paper/                      # LaTeX paper source + figures
├── notebooks/                  # Jupyter notebooks (Colab-ready)
├── app/                        # Gradio demo application
└── tests/                      # Test suite (pytest)
```

---

## Citation

```bibtex
@article{sathish2026dreamprice,
  title   = {DreamPrice: A Causal DreamerV3 World Model
             for Offline Retail Pricing},
  author  = {Sathish, Sharath},
  year    = {2026},
  url     = {https://github.com/SharathSPhD/dreamprice}
}
```

## License

This work is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
See [LICENSE](LICENSE) for details.

The Dominick's Finer Foods data is provided by the Kilts Center for Marketing, University of Chicago Booth School of Business, and is subject to their [terms of use](https://www.chicagobooth.edu/research/kilts/datasets/dominicks).
