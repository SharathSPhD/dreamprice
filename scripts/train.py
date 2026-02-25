"""DreamPrice training entry point. Config via Hydra."""

from __future__ import annotations

import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from retail_world_model.applications.pricing_policy import ActorCritic
    from retail_world_model.data.dataset import (
        DominicksSequenceDataset,
        HybridReplaySampler,
    )
    from retail_world_model.data.dominicks_loader import load_category
    from retail_world_model.models.world_model import MambaWorldModel
    from retail_world_model.training.trainer import DreamerTrainer
    from retail_world_model.utils.logging import NullLogger, WandbLogger

    # Seed
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logger
    if os.environ.get("WANDB_API_KEY"):
        logger = WandbLogger(
            project=cfg.get("wandb_project", "dreamprice"),
            config=OmegaConf.to_container(cfg, resolve=True),
            group=cfg.get("wandb_group", None),
        )
    else:
        logger = NullLogger()

    import pandas as pd

    # Build dataset
    data_dir = cfg.get("data_dir", "docs/data")
    category = cfg.get("category", "cso")
    movement_path = f"{data_dir}/{category}/w{category}.csv"

    max_stores = cfg.get("max_stores", 0)
    store_ids = None
    if max_stores > 0:
        all_stores = sorted(pd.read_csv(movement_path, usecols=["STORE"])["STORE"].unique())
        store_ids = all_stores[:max_stores]
        print(f"Pre-filtering to {max_stores} stores: {store_ids}")

    df = load_category(
        movement_path,
        f"{data_dir}/{category}/upc{category}.csv",
        f"{data_dir}/demo.csv",
        store_ids=store_ids,
    )
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    max_rows = cfg.get("max_rows", 500_000)
    if len(df) > max_rows:
        print(f"Sampling {max_rows:,} from {len(df):,} rows for memory efficiency")
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    seq_len = cfg.agent.seq_len
    n_skus = cfg.environment.n_skus
    dataset = DominicksSequenceDataset(df, seq_len=seq_len, n_skus=n_skus)
    sampler = HybridReplaySampler(dataset, batch_size=cfg.agent.batch_size)

    # Build model
    wm_cfg = cfg.world_model
    encoder_type = wm_cfg.get("encoder", "flat")
    backbone_type = wm_cfg.get("backbone", "mamba")
    model = MambaWorldModel(
        obs_dim=dataset.obs_dim,
        act_dim=n_skus,
        d_model=wm_cfg.d_model,
        n_cat=wm_cfg.n_cat,
        n_cls=wm_cfg.n_cls,
        elasticity_path=f"configs/elasticities/{category}.json",
        encoder_type=encoder_type,
        backbone_type=backbone_type,
        n_upcs=dataset.n_upcs,
        n_stores=dataset.n_stores,
    ).to(device)
    print(f"Encoder: {encoder_type}, Backbone: {backbone_type}")

    # Build actor-critic
    state_dim = wm_cfg.d_model + wm_cfg.z_dim
    ac = ActorCritic(
        state_dim=state_dim,
        n_skus=n_skus,
        action_dim=cfg.environment.action_steps,
        eta=cfg.agent.eta,
    ).to(device)

    # Build trainer
    trainer_cfg = OmegaConf.to_container(cfg.agent, resolve=True)
    trainer_cfg["save_every"] = cfg.get("save_every", 10000)
    trainer_cfg["checkpoint_dir"] = cfg.get("checkpoint_dir", "checkpoints")
    # Pass world model ablation flags so trainer can adjust behavior
    trainer_cfg["use_symlog"] = wm_cfg.get("use_symlog", True)
    trainer_cfg["use_twohot"] = wm_cfg.get("use_twohot", True)
    trainer = DreamerTrainer(
        model=model,
        actor_critic=ac,
        dataset=dataset,
        cfg=trainer_cfg,
        logger=logger,
        sampler=sampler,
    )

    # Train
    n_steps = cfg.get("n_steps", 100000)
    print(f"Starting training for {n_steps} steps on {device}")
    print(f"Dataset: {len(dataset)} sequences, obs_dim={dataset.obs_dim}")
    trainer.train(n_steps=n_steps)
    print("Training complete.")


if __name__ == "__main__":
    main()
