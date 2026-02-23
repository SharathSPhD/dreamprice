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

    import wandb

    from retail_world_model.applications.pricing_policy import ActorCritic
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

    # Build model (requires world model module to be implemented)
    # from retail_world_model.models.builder import WorldModelBuilder
    # model = WorldModelBuilder(cfg.world_model).build().to(device)
    print("Training script ready. World model builder required for full training.")
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")


if __name__ == "__main__":
    main()
