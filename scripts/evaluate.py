"""Evaluate a trained DreamPrice checkpoint on the test environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DreamPrice checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_episodes", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"Training step: {ckpt.get('step', 'unknown')}")

    # Build model from checkpoint config
    from retail_world_model.applications.pricing_policy import ActorCritic
    from retail_world_model.envs.grocery import GroceryPricingEnv
    from retail_world_model.models.world_model import MambaWorldModel

    cfg = ckpt.get("cfg", {})
    wm_cfg = cfg.get("world_model", {})
    env_cfg = cfg.get("environment", {})

    model = MambaWorldModel(
        obs_dim=wm_cfg.get("obs_dim", 64),
        act_dim=env_cfg.get("n_skus", 25),
        d_model=wm_cfg.get("d_model", 512),
        n_cat=wm_cfg.get("n_cat", 32),
        n_cls=wm_cfg.get("n_cls", 32),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.train(False)

    state_dim = wm_cfg.get("d_model", 512) + wm_cfg.get("z_dim", 1024)
    ac = ActorCritic(
        state_dim=state_dim,
        n_skus=env_cfg.get("n_skus", 25),
    )
    ac.load_state_dict(ckpt["actor_critic"])
    ac.train(False)

    n_skus = env_cfg.get("n_skus", 25)
    env = GroceryPricingEnv(
        world_model=model,
        store_features=np.zeros(8),
        initial_obs=np.zeros(wm_cfg.get("obs_dim", 64)),
        cost_vector=np.full(n_skus, 1.5),
        n_skus=n_skus,
        H=env_cfg.get("H", 13),
    )

    returns = []
    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        total_return = 0.0
        done = False
        while not done:
            d_m = wm_cfg.get("d_model", 512)
            z_d = wm_cfg.get("z_dim", 1024)
            h_t = env._h_t if env._h_t is not None else torch.zeros(1, d_m, device=device)
            z_t = env._z_t if env._z_t is not None else torch.zeros(1, z_d, device=device)
            if isinstance(h_t, torch.Tensor):
                h_t = h_t.to(device)
            if isinstance(z_t, torch.Tensor):
                z_t = z_t.to(device)
            state = torch.cat([h_t.reshape(1, -1), z_t.reshape(1, -1)], dim=-1)
            with torch.no_grad():
                action, _, _ = ac.act(state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            total_return += reward
            done = terminated or truncated
        returns.append(total_return)
        print(f"Episode {ep + 1}: return={total_return:.2f}")

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    n = len(returns)
    sorted_rets = sorted(returns)
    quarter = n // 4
    iqm = float(np.mean(sorted_rets[quarter : -(quarter) or None]))

    print(f"\nMean return: {mean_ret:.2f} +/- {std_ret:.2f}")
    print(f"IQM: {iqm:.2f}")

    results = {
        "returns": returns,
        "mean": mean_ret,
        "std": std_ret,
        "iqm": iqm,
    }
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    Path("docs/results/assessment.json").write_text(json.dumps(results, indent=2))
    print("Results written to docs/results/assessment.json")


if __name__ == "__main__":
    main()
