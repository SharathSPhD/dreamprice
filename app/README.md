---
title: DreamPrice Demo
emoji: "\U0001F4B0"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.6.0"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# DreamPrice: A Causal DreamerV3 World Model for Offline Retail Pricing

Interactive demo for DreamPrice -- a DreamerV3-based world model with Mamba-2 backbone
and causal demand decoder trained on Dominick's Finer Foods scanner data (100K steps, ELBO=22.44, Actor Return=193.7).

**Tabs:**
1. **Pricing Simulator**: Adjust SKU prices and observe predicted demand/margin
2. **Causal Analysis**: OLS vs IV elasticity comparison with bootstrap
3. **Architecture**: Model architecture visualization and configuration

[GitHub](https://github.com/SharathSPhD/dreamprice) |
[Dataset](https://huggingface.co/datasets/qbz506/dreamprice-dominicks-cso) |
[Model](https://huggingface.co/qbz506/dreamprice-cso) |
[Wandb](https://wandb.ai/qbz506-technektar/dreamprice)

**Author**: Sharath Sathish, University of York
