"""DreamerV3-style three-phase training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.utils.data

from retail_world_model.inference.imagination import (
    compute_lambda_returns,
    rollout_imagination,
)
from retail_world_model.training.losses import (
    elbo_loss,
    twohot_ce_loss,
)
from retail_world_model.training.offline_utils import PercentileReturnNorm
from retail_world_model.utils.logging import MetricsLogger, NullLogger
from retail_world_model.utils.twohot import make_bins


class DreamerTrainer:
    """Three-phase DreamerV3 training loop.

    Phase A (World Model): Adam lr=1e-4, grad clip norm 1000, B=32 x T=64
    Phase B (Actor):       imagination H=13, Adam lr=3e-5, grad clip 100
    Phase C (Critic):      twohot CE, EMA decay=0.98, Adam lr=3e-5, grad clip 100
    """

    def __init__(
        self,
        model: nn.Module,
        actor_critic: nn.Module,
        dataset: torch.utils.data.Dataset,
        cfg: dict[str, Any] | None = None,
        logger: MetricsLogger | None = None,
        sampler: torch.utils.data.Sampler | None = None,
    ) -> None:
        self.model = model
        self.actor_critic = actor_critic
        self.dataset = dataset
        self.logger: MetricsLogger = logger or NullLogger()
        self.cfg = cfg or {}
        self._sampler = sampler

        # Hyperparameters
        lr_wm = self.cfg.get("lr_wm", 1e-4)
        lr_actor = self.cfg.get("lr_actor", 3e-5)
        lr_critic = self.cfg.get("lr_critic", 3e-5)
        self.grad_clip_wm = self.cfg.get("grad_clip_wm", 1000.0)
        self.grad_clip_ac = self.cfg.get("grad_clip_ac", 100.0)
        self.gamma = self.cfg.get("gamma", 0.95)
        self.lambda_ = self.cfg.get("lambda_", 0.95)
        self.H = self.cfg.get("H", 13)
        self.lambda_lcb = self.cfg.get("lambda_lcb", 1.0)
        self.ema_decay = self.cfg.get("ema_decay", 0.98)

        # Optimizers
        self.opt_wm = torch.optim.Adam(model.parameters(), lr=lr_wm)
        self.opt_actor = torch.optim.Adam(
            actor_critic.actor.parameters(),
            lr=lr_actor,  # type: ignore[union-attr]
        )
        self.opt_critic = torch.optim.Adam(
            actor_critic.critic.parameters(),
            lr=lr_critic,  # type: ignore[union-attr]
        )

        # Slow critic (EMA target)
        self.slow_critic = _copy_module(actor_critic.critic)  # type: ignore[union-attr]

        # Return normalization
        self.return_norm = PercentileReturnNorm(ema_decay=0.99)

        # Bins for twohot
        self.bins = make_bins()

        self._step = 0

    def train_phase_a(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Phase A: world model update."""
        self.opt_wm.zero_grad()
        losses = elbo_loss(batch, self.model)
        losses["total"].backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_wm)
        self.opt_wm.step()

        return {k: v.item() for k, v in losses.items()}

    def train_phase_b(self, z0: torch.Tensor, h0: torch.Tensor) -> dict[str, float]:
        """Phase B: actor imagination update."""
        self.opt_actor.zero_grad()

        # Rollout in imagination
        rollout = rollout_imagination(
            self.model,
            z0,
            h0,
            self.actor_critic,
            H=self.H,
            lambda_lcb=self.lambda_lcb,
        )

        # Compute lambda-returns
        returns = compute_lambda_returns(
            rollout["rewards_pessimistic"],
            rollout["values"],
            rollout["continues"],
            gamma=self.gamma,
            lambda_=self.lambda_,
        )

        # Normalize returns
        self.return_norm.update(returns)
        returns_norm = self.return_norm.normalize(returns)

        # Advantages
        advantages = returns_norm - rollout["values"][:, :-1]

        # Actor loss
        loss = self.actor_critic.actor_loss(  # type: ignore[union-attr]
            rollout["log_probs"], advantages, rollout["entropies"]
        )
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor_critic.actor.parameters(),  # type: ignore[union-attr]
            self.grad_clip_ac,
        )
        self.opt_actor.step()

        return {
            "actor_loss": loss.item(),
            "return_mean": returns.mean().item(),
        }

    def train_phase_c(self, z0: torch.Tensor, h0: torch.Tensor) -> dict[str, float]:
        """Phase C: critic update with twohot distributional regression."""
        self.opt_critic.zero_grad()

        # Rollout (reuse or recompute)
        with torch.no_grad():
            rollout = rollout_imagination(
                self.model,
                z0,
                h0,
                self.actor_critic,
                H=self.H,
                lambda_lcb=self.lambda_lcb,
            )
            targets = compute_lambda_returns(
                rollout["rewards_pessimistic"],
                rollout["values"],
                rollout["continues"],
                gamma=self.gamma,
                lambda_=self.lambda_,
            )

        # Critic loss: twohot CE on lambda-returns
        states = torch.cat(
            [rollout["hs"][:, :-1], rollout["zs"][:, :-1]], dim=-1
        )  # (B, H, state_dim)
        B, H, D = states.shape
        logits = self.actor_critic.critique_logits(  # type: ignore[union-attr]
            states.reshape(B * H, D)
        ).reshape(B, H, -1)

        critic_loss = twohot_ce_loss(logits, targets, self.bins.to(logits.device))

        # Slow critic regularization
        with torch.no_grad():
            slow_logits = self.slow_critic(states.reshape(B * H, D)).reshape(B, H, -1)
        slow_reg = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            slow_logits.reshape(-1, slow_logits.shape[-1]).softmax(dim=-1),
        )

        total_loss = critic_loss + slow_reg
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor_critic.critic.parameters(),  # type: ignore[union-attr]
            self.grad_clip_ac,
        )
        self.opt_critic.step()

        # EMA update slow critic
        _ema_update(
            self.slow_critic,
            self.actor_critic.critic,  # type: ignore[union-attr]
            self.ema_decay,
        )

        return {"critic_loss": total_loss.item()}

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute all three phases on one batch."""
        # Move batch to model device
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        metrics: dict[str, float] = {}

        # Phase A: world model
        wm_metrics = self.train_phase_a(batch)
        metrics.update({f"wm/{k}": v for k, v in wm_metrics.items()})

        # Get initial states from world model output
        with torch.no_grad():
            output = self.model.forward(batch["x_BT"], batch["a_BT"])  # type: ignore[union-attr]
            # Use first-step states as imagination seeds
            z0 = output["z_posterior_BT"][:, 0]
            h0 = output["h_BT"][:, 0]

        # Phase B: actor
        actor_metrics = self.train_phase_b(z0, h0)
        metrics.update({f"actor/{k}": v for k, v in actor_metrics.items()})

        # Phase C: critic
        critic_metrics = self.train_phase_c(z0, h0)
        metrics.update({f"critic/{k}": v for k, v in critic_metrics.items()})

        self._step += 1
        self.logger.log(self._step, metrics)
        return metrics

    def train(self, n_steps: int = 100_000) -> None:
        """Full training loop."""
        if self._sampler is not None:
            loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.cfg.get("batch_size", 32),
                sampler=self._sampler,
                drop_last=True,
                num_workers=0,
            )
        else:
            loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.cfg.get("batch_size", 32),
                shuffle=True,
                drop_last=True,
                num_workers=0,
            )
        loader_iter = iter(loader)

        for step_i in range(n_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            self.train_step(batch)

            # Checkpoint saving
            if (step_i + 1) % self.cfg.get("save_every", 10000) == 0:
                ckpt_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "step": self._step,
                        "model": self.model.state_dict(),
                        "actor_critic": self.actor_critic.state_dict(),
                        "opt_wm": self.opt_wm.state_dict(),
                        "opt_actor": self.opt_actor.state_dict(),
                        "opt_critic": self.opt_critic.state_dict(),
                    },
                    ckpt_dir / f"step_{self._step:07d}.pt",
                )


def _copy_module(module: nn.Module) -> nn.Module:
    """Create a detached copy of a module for EMA target."""
    import copy

    target = copy.deepcopy(module)
    for p in target.parameters():
        p.requires_grad_(False)
    return target


@torch.no_grad()
def _ema_update(target: nn.Module, source: nn.Module, decay: float) -> None:
    """Exponential moving average update: target = decay*target + (1-decay)*source."""
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.mul_(decay).add_(s_param.data, alpha=1 - decay)
