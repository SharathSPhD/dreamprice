"""Run RL baselines (DQN, PPO, SAC) against the trained DreamPrice world model.

Loads the 100K-step checkpoint, wraps the GroceryPricingEnv with the world model,
and trains each SB3 baseline for a specified number of timesteps. Evaluates over
100 episodes and saves results to JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


def load_world_model(checkpoint_path: str, device: str = "cpu"):
    """Load the trained MambaWorldModel from checkpoint."""
    sys.path.insert(0, "/workspace/src")
    from retail_world_model.models.world_model import MambaWorldModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MambaWorldModel(
        obs_dim=27,
        act_dim=25,
        d_model=512,
        n_cat=32,
        n_cls=32,
        elasticity_path="/workspace/configs/elasticities/cso.json",
        encoder_type="flat",
        backbone_type="mamba",
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.train(False)
    return model


def make_env(world_model, wrapper: str = "discrete", seed: int = 42):
    """Create GroceryPricingEnv backed by the trained world model."""
    sys.path.insert(0, "/workspace/src")
    from retail_world_model.envs.grocery import GroceryPricingEnv

    from sb3_wrapper import ContinuousActionWrapper, FlatDiscreteWrapper

    rng = np.random.default_rng(seed)
    env = GroceryPricingEnv(
        world_model=world_model,
        store_features=np.zeros(10, dtype=np.float32),
        initial_obs=np.zeros(27, dtype=np.float32),
        cost_vector=rng.uniform(0.50, 2.00, 25).astype(np.float32),
        n_skus=25,
        H=13,
    )
    if wrapper == "discrete":
        return FlatDiscreteWrapper(env)
    elif wrapper == "continuous":
        return ContinuousActionWrapper(env)
    return env


def evaluate_policy(model, env, n_episodes: int = 100) -> dict:
    """Evaluate a trained SB3 policy over n_episodes."""
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "median_return": float(np.median(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "n_episodes": n_episodes,
    }


def run_dqn(world_model, total_timesteps: int, seed: int, output_path: Path):
    """Train and evaluate DQN baseline."""
    from stable_baselines3 import DQN

    env = make_env(world_model, wrapper="discrete", seed=seed)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10000,
        batch_size=64,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        train_freq=4,
        target_update_interval=500,
        seed=seed,
        verbose=0,
    )
    print(f"  Training DQN for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    print("  Evaluating DQN over 100 episodes...")
    eval_env = make_env(world_model, wrapper="discrete", seed=seed + 1000)
    results = evaluate_policy(model, eval_env)
    results["method"] = "DQN"
    results["total_timesteps"] = total_timesteps
    results["eval_type"] = "world_model"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  DQN: mean_return={results['mean_return']:.2f} (std={results['std_return']:.2f})")
    return results


def run_ppo(world_model, total_timesteps: int, seed: int, output_path: Path):
    """Train and evaluate PPO baseline."""
    from stable_baselines3 import PPO

    env = make_env(world_model, wrapper="continuous", seed=seed)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.01,
        seed=seed,
        verbose=0,
    )
    print(f"  Training PPO for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    print("  Evaluating PPO over 100 episodes...")
    eval_env = make_env(world_model, wrapper="continuous", seed=seed + 1000)
    results = evaluate_policy(model, eval_env)
    results["method"] = "PPO"
    results["total_timesteps"] = total_timesteps
    results["eval_type"] = "world_model"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  PPO: mean_return={results['mean_return']:.2f} (std={results['std_return']:.2f})")
    return results


def run_sac(world_model, total_timesteps: int, seed: int, output_path: Path):
    """Train and evaluate SAC baseline."""
    from stable_baselines3 import SAC

    env = make_env(world_model, wrapper="continuous", seed=seed)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=10000,
        batch_size=256,
        tau=0.005,
        ent_coef="auto",
        seed=seed,
        verbose=0,
    )
    print(f"  Training SAC for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    print("  Evaluating SAC over 100 episodes...")
    eval_env = make_env(world_model, wrapper="continuous", seed=seed + 1000)
    results = evaluate_policy(model, eval_env)
    results["method"] = "SAC"
    results["total_timesteps"] = total_timesteps
    results["eval_type"] = "world_model"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  SAC: mean_return={results['mean_return']:.2f} (std={results['std_return']:.2f})")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run RL baselines on trained world model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/checkpoints/step_0100000.pt",
    )
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/docs/results/baselines"),
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["dqn", "ppo", "sac"],
        choices=["dqn", "ppo", "sac"],
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading world model from {args.checkpoint}...")
    wm = load_world_model(args.checkpoint, device=args.device)
    print("  World model loaded successfully.")

    runners = {
        "dqn": (run_dqn, args.output_dir / "dqn.json"),
        "ppo": (run_ppo, args.output_dir / "ppo.json"),
        "sac": (run_sac, args.output_dir / "sac.json"),
    }

    for name in args.baselines:
        print(f"\n=== Running {name.upper()} ===")
        runner, output = runners[name]
        runner(wm, args.timesteps, args.seed, output)

    print("\nAll baselines complete.")


if __name__ == "__main__":
    main()
