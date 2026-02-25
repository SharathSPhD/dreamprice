"""Loss functions for DreamerV3-style training: ELBO, KL balancing, twohot CE."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from retail_world_model.utils.symlog import symlog
from retail_world_model.utils.twohot import make_bins, twohot_encode

# KL balancing constants (DreamerV3 defaults)
BETA_PRED = 1.0
BETA_DYN = 0.5  # dynamics KL (stop-gradient on posterior)
BETA_REP = 0.1  # representation KL (stop-gradient on prior)
FREE_BITS = 1.0  # free nats per categorical


def _categorical_kl(
    p_probs: torch.Tensor,
    q_probs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL(p || q) for categorical distributions.

    Args:
        p_probs: (..., n_cat, n_cls) probabilities of distribution p.
        q_probs: (..., n_cat, n_cls) probabilities of distribution q.

    Returns:
        (..., n_cat) KL divergence per categorical.
    """
    p = p_probs.clamp(min=eps)
    q = q_probs.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def kl_balancing(
    posterior_probs: torch.Tensor,
    prior_probs: torch.Tensor,
    beta_dyn: float = BETA_DYN,
    beta_rep: float = BETA_REP,
    free_bits: float = FREE_BITS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KL balancing with free bits (DreamerV3).

    L_KL = beta_dyn * KL(sg(posterior) || prior)
         + beta_rep * KL(posterior || sg(prior))

    Free bits: max(free_bits, KL) per categorical prevents posterior collapse.

    Args:
        posterior_probs: (B, T, n_cat, n_cls)
        prior_probs: (B, T, n_cat, n_cls)

    Returns:
        (total_kl, kl_dyn, kl_rep) — each scalar (mean over batch).
    """
    # Dynamics loss: train prior, freeze posterior
    kl_dyn_per_cat = _categorical_kl(posterior_probs.detach(), prior_probs)
    kl_dyn_per_cat = torch.clamp(kl_dyn_per_cat, min=free_bits)
    kl_dyn = kl_dyn_per_cat.sum(dim=-1).mean()

    # Representation loss: train posterior, freeze prior
    kl_rep_per_cat = _categorical_kl(posterior_probs, prior_probs.detach())
    kl_rep_per_cat = torch.clamp(kl_rep_per_cat, min=free_bits)
    kl_rep = kl_rep_per_cat.sum(dim=-1).mean()

    total = beta_dyn * kl_dyn + beta_rep * kl_rep
    return total, kl_dyn, kl_rep


def twohot_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bins: torch.Tensor | None = None,
) -> torch.Tensor:
    """Categorical cross-entropy against soft twohot labels.

    Used for reward and critic losses.

    Args:
        logits: (..., n_bins) raw logits.
        target: (...) scalar targets in original space (will be symlogged).
        bins: (n_bins,) bin centres in symlog space. Defaults to 255 bins in [-20, 20].

    Returns:
        Scalar mean loss.
    """
    if bins is None:
        bins = make_bins()
    bins = bins.to(logits.device)

    # Symlog the target, then encode as twohot
    target_symlog = symlog(target)
    labels = twohot_encode(target_symlog, bins)  # (..., n_bins)

    # Cross-entropy: -sum(labels * log_softmax(logits))
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(labels * log_probs).sum(dim=-1)
    return loss.mean()


def elbo_loss(
    batch: dict[str, torch.Tensor],
    model: object,
    beta_pred: float = BETA_PRED,
    beta_dyn: float = BETA_DYN,
    beta_rep: float = BETA_REP,
    free_bits: float = FREE_BITS,
    use_symlog: bool = True,
    use_twohot: bool = True,
) -> dict[str, torch.Tensor]:
    """Full ELBO loss for world model training.

    L = beta_pred * (recon + reward + continue) + kl_balancing

    Args:
        batch: Dict with keys x_BT, a_BT, r_BT, done_BT.
        model: World model with forward() returning aligned output dict.
        use_symlog: Apply symlog to reconstruction targets.
        use_twohot: Use twohot CE for reward loss (vs raw MSE).

    Returns:
        Dict with keys: total, recon, reward, continue, kl_total, kl_dyn, kl_rep.
    """
    entity_ids = None
    if "store_id" in batch and "month_ids" in batch:
        entity_ids = {
            "store_ids": batch["store_id"].unsqueeze(1).expand(-1, batch["x_BT"].shape[1]),
            "month_ids": batch["month_ids"],
        }
    output = model.forward(batch["x_BT"], batch["a_BT"], entity_ids=entity_ids)  # type: ignore[union-attr]

    # Reconstruction loss
    x_recon = output["x_recon_BT"]
    x_target = symlog(batch["x_BT"]) if use_symlog else batch["x_BT"]
    recon_loss = 0.5 * (x_recon - x_target).pow(2).sum(dim=-1).mean()

    # Reward loss
    if use_twohot:
        reward_loss = twohot_ce_loss(output["reward_logits_BT"], batch["r_BT"])
    else:
        reward_loss = 0.5 * (output["reward_mean"] - batch["r_BT"]).pow(2).mean()

    # Continue loss: BCE
    cont_loss = continue_bce_loss(output["continue_logits"], batch["done_BT"])

    # KL balancing
    kl_total, kl_dyn, kl_rep = kl_balancing(
        output["posterior_probs_BT"],
        output["prior_probs_BT"],
        beta_dyn=beta_dyn,
        beta_rep=beta_rep,
        free_bits=free_bits,
    )

    pred_loss = recon_loss + reward_loss + cont_loss
    total = beta_pred * pred_loss + kl_total

    return {
        "total": total,
        "recon": recon_loss,
        "reward": reward_loss,
        "continue": cont_loss,
        "kl_total": kl_total,
        "kl_dyn": kl_dyn,
        "kl_rep": kl_rep,
    }


def causal_reg_loss(
    decoder: object,
    log_price: torch.Tensor,
    theta_min: float = -4.0,
    theta_max: float = -1.0,
) -> torch.Tensor:
    """Optional regularization ensuring theta stays in plausible elasticity range.

    Penalizes if learned/frozen theta drifts outside [theta_min, theta_max].
    """
    theta = decoder.theta.weight  # type: ignore[union-attr]
    violation = F.relu(theta - theta_max) + F.relu(theta_min - theta)
    return violation.mean()


def continue_bce_loss(logits: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    """BCE loss for episode continuation prediction.

    Args:
        logits: (...) raw continue logits.
        dones: (...) binary done flags (1 = episode ended).
    """
    continues = 1.0 - dones.float()
    return F.binary_cross_entropy_with_logits(logits, continues)
