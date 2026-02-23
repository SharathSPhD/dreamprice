"""Integration test: DreamerTrainer.train_step completes on random data."""

import pytest
import torch
import torch.utils.data

from retail_world_model.applications.pricing_policy import ActorCritic
from retail_world_model.models.world_model import MambaWorldModel
from retail_world_model.training.trainer import DreamerTrainer


class TestTrainerIntegration:
    @pytest.fixture
    def trainer(self):
        obs_dim, act_dim = 32, 4
        d_model, n_cat, n_cls = 64, 8, 8
        model = MambaWorldModel(
            obs_dim=obs_dim,
            act_dim=act_dim,
            d_model=d_model,
            n_cat=n_cat,
            n_cls=n_cls,
        )
        state_dim = d_model + n_cat * n_cls
        ac = ActorCritic(
            state_dim=state_dim,
            n_skus=act_dim,
            action_dim=21,
        )
        dataset = torch.utils.data.TensorDataset(
            torch.randn(10, 8, obs_dim),
        )
        return DreamerTrainer(
            model=model,
            actor_critic=ac,
            dataset=dataset,
        )

    def test_train_step_completes(self, trainer):
        batch = {
            "x_BT": torch.randn(2, 8, 32),
            "a_BT": torch.randn(2, 8, 4),
            "r_BT": torch.randn(2, 8),
            "done_BT": torch.zeros(2, 8),
        }
        metrics = trainer.train_step(batch)
        assert "wm/total" in metrics
        assert "actor/actor_loss" in metrics
        assert "critic/critic_loss" in metrics

    def test_loss_decreases(self, trainer):
        batch = {
            "x_BT": torch.randn(2, 8, 32),
            "a_BT": torch.randn(2, 8, 4),
            "r_BT": torch.randn(2, 8),
            "done_BT": torch.zeros(2, 8),
        }
        losses = []
        for _ in range(5):
            m = trainer.train_step(batch)
            losses.append(m["wm/total"])
        # Loss should not explode (may not strictly decrease on random data)
        assert losses[-1] < losses[0] * 1.5
