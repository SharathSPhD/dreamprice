---
license: cc-by-nc-4.0
language: en
tags:
  - reinforcement-learning
  - world-model
  - retail
  - pricing
  - mamba
  - dreamer
datasets:
  - dominicks-finer-foods
---

# DreamPrice: Learned World Model for Retail Pricing

## Model Description
First learned world model for retail pricing environments. Combines DreamerV3's
training recipe with a Mamba-2 SSM backbone, entity-factored multi-SKU representation,
Hausman IV causal identification, and MOPO-style offline pessimism.

## Architecture
- **Backbone**: Mamba-2 (d_model=512, d_state=64) with DRAMA-style decoupled posterior
- **Latent space**: 32 x 32 categorical (1024-dim one-hot), unimix=1%
- **Causal decoder**: DML-PLIV frozen elasticities in [-3.0, -2.0]
- **Reward ensemble**: 5-head for MOPO LCB (lambda=1.0)

## Training Data
Dominick's Finer Foods scanner data, 1989-1997, 93 stores, ~18K UPCs.
Temporal split: train weeks 1-280, val 281-340, test 341-400.
**Start category**: Canned Soup (~25 SKUs, ~581K training tuples).

## Intended Use
Academic research into learned pricing policies. NOT intended for production deployment
without extensive validation on current market data.

## Limitations
- Trained on 1989-1997 data; modern retail dynamics differ significantly
- Geographic scope: Chicago metropolitan area
- Single retailer (Dominick's); competitive dynamics not fully captured
- Causal identification assumes Hausman IV relevance (F > 10) and exclusion

## Citation
```bibtex
@software{dreamprice2026,
  title  = {DreamPrice: Learned World Model for Retail Pricing},
  author = {Sharath S},
  year   = {2026},
  url    = {https://github.com/SharathSPhD/dreamprice}
}
```

## License
CC-BY-NC-4.0. Data from Kilts Center for Marketing, University of Chicago.
