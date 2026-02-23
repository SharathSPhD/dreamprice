"""DreamPrice training entry point. Config via Hydra."""

from __future__ import annotations

import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="experiment/main")
def main(cfg: DictConfig) -> None:
    from dotenv import load_dotenv

    load_dotenv()

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

    # Build dataset
    data_dir = cfg.get("data_dir", "docs/data")
    category = cfg.get("category", "cso")
    df = load_category(
        f"{data_dir}/{category}/w{category}.csv",
        f"{data_dir}/{category}/upc{category}.csv",
        f"{data_dir}/{category}/demo.csv",
    )

    seq_len = cfg.agent.seq_len
    n_skus = cfg.environment.n_skus
    dataset = DominicksSequenceDataset(df, seq_len=seq_len, n_skus=n_skus)
    sampler = HybridReplaySampler(dataset, batch_size=cfg.agent.batch_size)

    # Build model
    wm_cfg = cfg.world_model
    model = MambaWorldModel(
        obs_dim=dataset.obs_dim,
        act_dim=n_skus,
        d_model=wm_cfg.d_model,
        n_cat=wm_cfg.n_cat,
        n_cls=wm_cfg.n_cls,
        elasticity_path=f"configs/elasticities/{category}.json",
    ).to(device)

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
