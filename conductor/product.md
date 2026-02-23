# Product Definition — DreamPrice

## Project Name
DreamPrice

## Description
The first learned world model for retail pricing environments: a Mamba-2-backed RSSM trained offline on Dominick's Finer Foods scanner data (1989–1997) to learn competitive retail pricing dynamics, enabling counterfactual demand estimation and imagination-based policy optimization.

## Problem Statement
Every existing RL pricing system uses either hand-crafted analytical simulators or model-free methods. No learned world model exists for economic environments. The gap spans five dimensions:
- No learned retail dynamics model
- No Dyna-style imagination training for pricing
- No standardized RL environment for pricing
- No world model on economic time series
- No competitive pricing sandbox with a learned model

## Target Users
- Academic ML researchers studying world models and offline RL
- Empirical IO economists interested in counterfactual demand estimation
- RL practitioners looking for a non-game benchmark domain

## Key Goals
1. Train a world model that accurately predicts multi-step demand rollouts on Dominick's data
2. Show that imagination-based actor-critic outperforms model-free baselines on cumulative gross margin
3. Release pip-installable package + Gymnasium env + HF Hub checkpoints under CC-BY-NC-4.0
4. Publish to arXiv (target: ICML/NeurIPS 2026 workshop)

## Novel Contribution
The **causal constrained decoder**: freezing per-category price elasticities from DML-PLIV while the residual network learns everything else from latent state. First proposed integration of econometric causal identification directly into a DreamerV3-style decoder.

## License Constraint
Dataset (Dominick's Finer Foods) is academic-only via Kilts Center. All trained models must use **CC-BY-NC-4.0**. Acknowledge the Kilts Center for Marketing at the University of Chicago Booth School of Business in all outputs.
