"""FastAPI application factory with lifespan management."""

from __future__ import annotations

import random
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI

from retail_world_model.api.batching import DynamicBatcher
from retail_world_model.api.routes import health, pricing, stream
from retail_world_model.api.schemas import PricingRequest, PricingResponse
from retail_world_model.applications.pricing_policy import ActorCritic
from retail_world_model.models.world_model import MambaWorldModel
from retail_world_model.utils.symlog import symlog

# 21 discrete price multipliers: {0.90, 0.91, ..., 1.10}
PRICE_MULTIPLIERS = [0.90 + i * 0.01 for i in range(21)]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(model_path: str, device: torch.device) -> tuple[MambaWorldModel, ActorCritic]:
    """Load MambaWorldModel and ActorCritic from checkpoint.

    Infers obs_dim from checkpoint state dict and act_dim (default 25).
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_sd = ckpt["model"]
    ac_sd = ckpt["actor_critic"]

    # Infer obs_dim from encoder weight
    enc_key = "rssm.obs_encoder.net.0.weight"
    obs_dim = int(model_sd[enc_key].shape[1])

    # Infer act_dim (default 25 for n_skus)
    act_dim = 25

    # Infer n_skus from actor output: actor.2.weight has shape (n_skus * action_dim, d_hidden)
    ac_actor_out = ac_sd.get("actor.2.weight")
    if ac_actor_out is not None:
        out_dim = ac_actor_out.shape[0]
        # action_dim=21, so n_skus = out_dim // 21
        act_dim = out_dim // 21

    model = MambaWorldModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        d_model=512,
        n_cat=32,
        n_cls=32,
    )
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if unexpected:
        # Checkpoint may have mamba keys when mamba_ssm is unavailable
        mamba_keys = [k for k in unexpected if "mamba" in k]
        if mamba_keys and len(mamba_keys) == len(unexpected):
            raise RuntimeError(
                "Checkpoint requires mamba_ssm (install: pip install mamba-ssm causal-conv1d). "
                f"Unexpected keys: {mamba_keys[:3]}..."
            )
    model.to(device)
    model.train(False)

    state_dim = 512 + 1024  # d_model + z_dim
    ac = ActorCritic(
        state_dim=state_dim,
        n_skus=act_dim,
        action_dim=21,
    )
    ac.load_state_dict(ac_sd)
    ac.to(device)
    ac.train(False)

    return model, ac


def _build_observation(
    current_prices: list[float],
    obs_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a minimal observation tensor from current_prices.

    Uses symlog(prices) in the first slots, zeros elsewhere.
    """
    prices = torch.tensor(
        current_prices,
        dtype=torch.float32,
        device=device,
    )
    x = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
    n = min(len(current_prices), obs_dim)
    x[0, :n] = symlog(prices[:n])
    return x


def _discrete_actions_to_multipliers(actions: torch.Tensor) -> torch.Tensor:
    """Convert discrete action indices (0-20) to price multipliers (0.90-1.10)."""
    mult = torch.tensor(
        PRICE_MULTIPLIERS,
        dtype=actions.dtype,
        device=actions.device,
    )
    return mult[actions.clamp(0, 20)]


# ---------------------------------------------------------------------------
# Stub model functions (used when no real checkpoint is provided)
# ---------------------------------------------------------------------------


async def _stub_batch_fn(requests: list[Any]) -> list[PricingResponse]:
    """Process a batch of pricing requests with stub predictions."""
    results: list[PricingResponse] = []
    for req in requests:
        n = len(req.upc_ids)
        results.append(
            PricingResponse(
                recommended_prices=[
                    p * (0.95 + 0.1 * random.random()) for p in req.current_prices
                ],
                expected_units=[random.uniform(10, 100) for _ in range(n)],
                expected_profit=random.uniform(50, 500),
                uncertainty_bounds=[(p * 0.9, p * 1.1) for p in req.current_prices],
            )
        )
    return results


async def _stub_model_fn(request: PricingRequest) -> PricingResponse:
    """Single-request stub for /imagine."""
    return (await _stub_batch_fn([request]))[0]


async def _stub_stream_fn(request: PricingRequest) -> AsyncGenerator[dict[str, Any], None]:
    """Yield step-by-step stub imagination results."""
    n = len(request.upc_ids)
    for step in range(request.horizon):
        yield {
            "step": step,
            "prices": [p * (0.95 + 0.1 * random.random()) for p in request.current_prices],
            "expected_units": [random.uniform(10, 100) for _ in range(n)],
            "expected_profit": random.uniform(50, 500),
        }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(model_path: str | None = None) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters
    ----------
    model_path:
        Path to a saved model checkpoint.  When *None* the app starts with
        stub functions so that the API surface can be tested without a
        trained model.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # --- startup ---
        model = None
        actor_critic = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is not None and Path(model_path).exists():
            try:
                model, actor_critic = _load_model(model_path, device)
                app.state.model_loaded = True
                obs_dim = model.rssm.obs_dim
                act_dim = model.rssm.act_dim

                def _real_batch_fn(requests: list[PricingRequest]) -> list[PricingResponse]:
                    results: list[PricingResponse] = []
                    for req in requests:
                        with torch.no_grad():
                            x = _build_observation(
                                req.current_prices, obs_dim, device
                            )
                            z_t, _ = model.rssm.encode_obs(x)
                            model.reset_state(batch_size=1)
                            h_t = torch.zeros(
                                1, model.rssm.d_model,
                                device=device, dtype=z_t.dtype,
                            )
                            state = torch.cat([h_t, z_t], dim=-1)
                            actions, _, _ = actor_critic.act(state, deterministic=True)
                            multipliers = _discrete_actions_to_multipliers(actions)
                            rec_prices = [
                                p * m
                                for p, m in zip(
                                    req.current_prices,
                                    multipliers[0].cpu().tolist(),
                                )
                            ]
                            # Pad if fewer prices than act_dim
                            while len(rec_prices) < act_dim:
                                rec_prices.append(rec_prices[-1] if rec_prices else 0.0)
                            rec_prices = rec_prices[: len(req.current_prices)]

                            # Imagination rollout for horizon
                            H = min(req.horizon, 13)
                            r_means: list[float] = []
                            r_stds: list[float] = []
                            z_t = z_t.clone()
                            for _ in range(H):
                                state = torch.cat([h_t, z_t], dim=-1)
                                actions, _, _ = actor_critic.act(state, deterministic=True)
                                mult = _discrete_actions_to_multipliers(actions)
                                step_out = model.imagine_step(z_t, mult)
                                h_t = step_out["h"]
                                z_t = step_out["z"]
                                r_means.append(float(step_out["r_mean"][0]))
                                r_stds.append(float(step_out["r_std"][0]))

                            total_profit = sum(r_means)
                            mean_r_std = sum(r_stds) / max(len(r_stds), 1)
                            mean_r_mean = total_profit / max(H, 1)
                            r_std_rel = mean_r_std / (abs(mean_r_mean) + 1e-6)
                            k = min(0.1, float(r_std_rel))
                            uncertainty_bounds = [
                                (p * (1 - k), p * (1 + k))
                                for p in rec_prices
                            ]
                            n_skus = len(req.current_prices)
                            avg_price = sum(req.current_prices) / max(n_skus, 1)
                            est_units = (
                                total_profit / (avg_price * 0.2 + 1e-6)
                                / max(n_skus, 1)
                            )
                            expected_units = [est_units] * n_skus

                        results.append(
                            PricingResponse(
                                recommended_prices=rec_prices,
                                expected_units=expected_units,
                                expected_profit=total_profit,
                                uncertainty_bounds=uncertainty_bounds,
                            )
                        )
                    return results

                async def _real_batch_fn_async(
                    requests: list[Any],
                ) -> list[PricingResponse]:
                    return _real_batch_fn([r for r in requests])

                async def _real_model_fn(request: PricingRequest) -> PricingResponse:
                    return (await _real_batch_fn_async([request]))[0]

                async def _real_stream_fn(
                    request: PricingRequest,
                ) -> AsyncGenerator[dict[str, Any], None]:
                    with torch.no_grad():
                        x = _build_observation(
                            request.current_prices, obs_dim, device
                        )
                        z_t, _ = model.rssm.encode_obs(x)
                        model.reset_state(batch_size=1)
                        h_t = torch.zeros(
                            1, model.rssm.d_model,
                            device=device, dtype=z_t.dtype,
                        )
                        n = len(request.current_prices)
                        H = min(request.horizon, 13)
                        prices = list(request.current_prices)
                        for step in range(H):
                            state = torch.cat([h_t, z_t], dim=-1)
                            actions, _, _ = actor_critic.act(
                                state, deterministic=True
                            )
                            mult = _discrete_actions_to_multipliers(actions)
                            step_out = model.imagine_step(z_t, mult)
                            h_t = step_out["h"]
                            z_t = step_out["z"]
                            rec_prices = [
                                prices[i] * mult[0, i].item()
                                for i in range(min(n, mult.shape[1]))
                            ]
                            if len(rec_prices) < n:
                                rec_prices.extend(
                                    [rec_prices[-1]] * (n - len(rec_prices))
                                )
                            prices = rec_prices
                            yield {
                                "step": step,
                                "prices": rec_prices,
                                "expected_units": [0.0] * n,
                                "expected_profit": float(step_out["r_mean"][0]),
                            }

                batch_fn = _real_batch_fn_async
                app.state.model_fn = _real_model_fn
                app.state.stream_fn = _real_stream_fn
            except Exception:
                app.state.model_loaded = False
                batch_fn = _stub_batch_fn
                app.state.model_fn = _stub_model_fn
                app.state.stream_fn = _stub_stream_fn
        else:
            app.state.model_loaded = False
            batch_fn = _stub_batch_fn
            app.state.model_fn = _stub_model_fn
            app.state.stream_fn = _stub_stream_fn

        app.state.device = "cuda" if device.type == "cuda" else "cpu"
        app.state.model_name = "dreamprice"
        app.state.model_version = "0.1.0"
        app.state.categories = ["cso", "ber", "sdr"]

        batcher = DynamicBatcher(
            process_fn=batch_fn,
            max_batch_size=8,
            max_wait_ms=50.0,
        )
        app.state.batcher = batcher
        await batcher.start()

        yield

        # --- shutdown ---
        await batcher.stop()

    app = FastAPI(
        title="DreamPrice API",
        description="Learned world model for retail pricing recommendations.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(pricing.router)
    app.include_router(stream.router)

    return app
