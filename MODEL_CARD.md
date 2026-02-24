# Model Card: DreamPrice

## Model Details

- **Model name**: DreamPrice
- **Model type**: Learned world model (RSSM with Mamba-2 backbone)
- **Architecture**: DreamerV3-style three-phase training with DRAMA decoupled posteriors
- **Parameters**: ~22M (full model)
- **Framework**: PyTorch
- **License**: CC-BY-NC-4.0
- **Trained checkpoint**: `step_0100000.pt` (100K gradient steps)

## Training Results (100K Steps, DGX Spark)

| Metric | Final Value |
|--------|-------------|
| World Model ELBO | 22.44 |
| Reconstruction Loss | 0.001 |
| KL Divergence | 19.20 |
| Reward Prediction | 3.17 |
| Actor Return (mean) | 124.33 |
| Critic Loss | 2.50 |
| Training Time | ~2.6 hours |
| Hardware | NVIDIA DGX Spark (GB10 GPU, 128 GB unified memory) |

### World Model Quality (Evaluation on Validation Set)

| Metric | h=1 | h=5 | h=10 | h=13 | h=25 |
|--------|-----|-----|------|------|------|
| RMSE | 5.001 | 5.167 | 5.286 | 5.283 | 5.243 |
| MAE | 2.130 | 2.168 | 2.207 | 2.205 | 2.197 |
| WMAPE (%) | 71.7 | 72.3 | 72.6 | 72.6 | 72.4 |
| NDR(h) | 1.00 | 1.033 | 1.057 | 1.056 | 1.049 |

### Policy Comparison

| Method | Mean Return | IQM |
|--------|-------------|-----|
| Cost-plus (25%) | 54.8 | 54.8 |
| Static XGBoost | 87.2 | 85.6 |
| Competitive Matching | 42.1 | 41.5 |
| DQN | 68.9 | 65.2 |
| PPO | 76.4 | 72.8 |
| SAC | 82.3 | 79.6 |
| **DreamPrice** | **124.3** | **117.4** |

## Intended Use

DreamPrice is a learned dynamics model for retail pricing environments. It is intended for:

- **Counterfactual demand estimation**: "What would demand be if price were cut by 5%?"
- **Pricing policy optimization**: Learning pricing strategies via imagination-based offline RL
- **Research**: Studying learned world models in economic domains

DreamPrice is NOT intended for:

- Direct deployment in production pricing systems without human oversight
- Real-time autonomous pricing decisions
- Categories or retail environments significantly different from the training data

## Training Data

- **Dataset**: Dominick's Finer Foods scanner data (Kilts Center for Marketing, University of Chicago)
- **Period**: September 1989 to May 1997 (400 weeks)
- **Scope**: 93 stores, ~18,000 UPCs, 29 product categories
- **Primary category**: Canned soup (cso), ~581K training tuples
- **Temporal split**: Train weeks 1-280, Validation weeks 281-340, Test weeks 341-400
- **HuggingFace dataset**: [qbz506/dreamprice-dominicks-cso](https://huggingface.co/datasets/qbz506/dreamprice-dominicks-cso)

## Architecture Details

| Component | Specification |
|-----------|---------------|
| Backbone | Mamba-2 SSM (d_model=512) with GRU fallback |
| Stochastic latent | 32 categorical variables x 32 classes (z_dim=1024) |
| Posterior | DRAMA-style decoupled: q(z_t \| x_t) |
| Observation decoder | 3-layer MLP: cat(h_t, z_t) -> obs_dim |
| Demand decoder | Causal: theta * log(price) + MLP(z_t, store_features) |
| Reward ensemble | 5 independent heads, twohot distributional (255 bins) |
| Continue head | Linear -> sigmoid |
| MOPO pessimism | r_pessimistic = r_mean - lambda_lcb * r_std |

## Causal Identification

Price elasticities are estimated via Hausman IV + DML-PLIV and frozen into the decoder:

- **Instrument**: Leave-one-out mean log(price) across other stores for same UPC-week
- **Method**: DoubleML PLIV with random forest nuisance learners, 5-fold cross-fitting
- **DML-PLIV elasticity**: -0.940 (SE=0.006, 95% CI: [-0.952, -0.928])
- **First-stage F-stat**: 23,381 (>> 10 Stock-Yogo threshold)

## Resources

- **Code**: [github.com/SharathSPhD/dreamprice](https://github.com/SharathSPhD/dreamprice)
- **Demo**: [huggingface.co/spaces/qbz506/dreamprice-demo](https://huggingface.co/spaces/qbz506/dreamprice-demo)
- **Wandb**: [wandb.ai/qbz506-technektar/dreamprice](https://wandb.ai/qbz506-technektar/dreamprice)

## Limitations

- Trained on 1989-1997 data; modern retail dynamics may differ substantially
- Single category evaluation (canned soup); cross-category transfer not validated
- Offline learning only; no online fine-tuning or real-time deployment tested
- Causal identification relies on Hausman IV panel structure

## Citation

```bibtex
@article{sathish2026dreamprice,
  title  = {DreamPrice: A Learned World Model for Retail Pricing via Mamba-2 Recurrence and Causal Demand Identification},
  author = {Sathish, Sharath},
  year   = {2026},
  url    = {https://github.com/SharathSPhD/dreamprice}
}
```
