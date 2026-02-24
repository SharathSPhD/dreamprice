"""DQN baseline using stable-baselines3 on GroceryPricingEnv."""

from __future__ import annotations

import argparse
import sys

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback

    import wandb
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install stable-baselines3 wandb")
    sys.exit(1)

from scripts.baselines.sb3_wrapper import make_env


class WandbCallback(BaseCallback):
    """Log SB3 metrics to W&B."""

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0 and self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "gross_margin" in info:
                    wandb.log({"gross_margin": info["gross_margin"]}, step=self.n_calls)
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="DQN baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=50000)
    parser.add_argument("--n-skus", type=int, default=5)
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()

    env = make_env(n_skus=args.n_skus, wrapper="discrete", seed=args.seed)

    if args.use_wandb:
        wandb.init(project="dreamprice", group="baselines", name=f"dqn-seed{args.seed}")

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.95,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        seed=args.seed,
        verbose=1,
    )

    callbacks = [WandbCallback()] if args.use_wandb else []
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    # Evaluate
    total_reward = 0.0
    n_eval = 10
    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_reward += ep_reward

    avg_reward = total_reward / n_eval
    print(f"DQN avg episode reward: {avg_reward:.2f}")

    if args.use_wandb:
        wandb.log({"eval/avg_episode_reward": avg_reward})
        wandb.finish()


if __name__ == "__main__":
    main()
