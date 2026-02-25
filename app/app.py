"""DreamPrice: Interactive Retail Pricing Demo with Gradio."""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

ELASTICITY_FILE = APP_DIR / "configs" / "cso.json"
STORE_ELAST_FILE = APP_DIR / "configs" / "cso_store_elasticities.json"

if not ELASTICITY_FILE.exists():
    ELASTICITY_FILE = ROOT / "configs" / "elasticities" / "cso.json"
    STORE_ELAST_FILE = ROOT / "configs" / "elasticities" / "cso_store_elasticities.json"

COLORS = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "red": "#C44E52",
    "purple": "#8172B3",
}


def load_elasticity():
    if ELASTICITY_FILE.exists():
        return json.loads(ELASTICITY_FILE.read_text())
    return {"theta_causal": -0.94, "ols_elasticity": -0.93, "se": 0.006, "n_obs": 148640}


def load_store_elasticities():
    if STORE_ELAST_FILE.exists():
        return json.loads(STORE_ELAST_FILE.read_text())
    return []


# ── Tab 1: Interactive Pricing Simulator ──


PRICE_MULTIPLIERS = [0.90 + i * 0.01 for i in range(21)]

_world_model = None
_actor_critic = None
_device = None
_model_loaded = False


def _try_load_model():
    global _world_model, _actor_critic, _device, _model_loaded
    if _model_loaded:
        return True

    try:
        import torch

        from retail_world_model.applications.pricing_policy import ActorCritic
        from retail_world_model.models.world_model import MambaWorldModel

        ckpt_path = os.environ.get(
            "DREAMPRICE_CHECKPOINT",
            str(ROOT / "checkpoints" / "step_0100000.pt"),
        )
        if not Path(ckpt_path).exists():
            return False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_sd = ckpt["model"]
        ac_sd = ckpt["actor_critic"]

        enc_key = "rssm.obs_encoder.net.0.weight"
        entity_key = "rssm.obs_encoder.continuous_proj.0.weight"
        encoder_type = "flat"
        if entity_key in model_sd:
            obs_dim = int(model_sd[entity_key].shape[1])
            encoder_type = "entity"
            n_upcs = model_sd.get("rssm.obs_encoder.upc_embed.weight", torch.zeros(1)).shape[0]
            n_stores = model_sd.get("rssm.obs_encoder.store_embed.weight", torch.zeros(1)).shape[0]
        elif enc_key in model_sd:
            obs_dim = int(model_sd[enc_key].shape[1])
            n_upcs, n_stores = 0, 0
        else:
            return False

        act_dim = 25
        ac_actor_out = ac_sd.get("actor.2.weight")
        if ac_actor_out is not None:
            act_dim = ac_actor_out.shape[0] // 21

        model = MambaWorldModel(
            obs_dim=obs_dim,
            act_dim=act_dim,
            d_model=512,
            n_cat=32,
            n_cls=32,
            encoder_type=encoder_type,
            n_upcs=max(n_upcs, 1),
            n_stores=max(n_stores, 1),
        )
        model.load_state_dict(model_sd, strict=False)
        model.to(device)
        model.train(False)

        state_dim = 512 + 1024
        ac = ActorCritic(
            state_dim=state_dim,
            n_skus=act_dim,
            action_dim=21,
        )
        ac.load_state_dict(ac_sd)
        ac.to(device)
        ac.train(False)

        _world_model = model
        _actor_critic = ac
        _device = device
        _model_loaded = True
        return True
    except Exception as e:
        print(f"World model loading failed, using analytical fallback: {e}")
        return False


def _model_predict(prices, n_horizon=13):
    """Use the trained world model + actor-critic for pricing predictions."""
    import torch

    from retail_world_model.utils.symlog import symlog

    model = _world_model
    ac = _actor_critic
    device = _device

    with torch.no_grad():
        obs_dim = model.rssm.obs_dim
        x = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
        p_t = torch.tensor(prices, dtype=torch.float32, device=device)
        n = min(len(prices), obs_dim)
        x[0, :n] = symlog(p_t[:n])

        z_t, _ = model.rssm.encode_obs(x)
        model.reset_state(batch_size=1)
        h_t = torch.zeros(1, model.rssm.d_model, device=device, dtype=z_t.dtype)

        state = torch.cat([h_t, z_t], dim=-1)
        actions, _, _ = ac.act(state, deterministic=True)

        mult_table = torch.tensor(PRICE_MULTIPLIERS, dtype=torch.float32, device=device)
        multipliers = mult_table[actions.clamp(0, 20)]
        rec_prices = (p_t * multipliers[0, : len(prices)]).cpu().numpy()

        r_means, r_stds = [], []
        for _ in range(n_horizon):
            state = torch.cat([h_t, z_t], dim=-1)
            actions, _, _ = ac.act(state, deterministic=True)
            mult = mult_table[actions.clamp(0, 20)]
            step_out = model.imagine_step(z_t, mult)
            h_t = step_out["h"]
            z_t = step_out["z"]
            r_means.append(float(step_out["r_mean"][0]))
            r_stds.append(float(step_out["r_std"][0]))

    return rec_prices, r_means, r_stds


def simulate_pricing(
    price_0,
    price_1,
    price_2,
    price_3,
    price_4,
    base_demand_0,
    base_demand_1,
    base_demand_2,
    base_demand_3,
    base_demand_4,
    cost_0,
    cost_1,
    cost_2,
    cost_3,
    cost_4,
):
    """Run pricing simulation using world model or analytical fallback."""
    elast = load_elasticity()
    theta = elast["theta_causal"]

    prices = np.array([price_0, price_1, price_2, price_3, price_4])
    base_demands = np.array(
        [
            base_demand_0,
            base_demand_1,
            base_demand_2,
            base_demand_3,
            base_demand_4,
        ]
    )
    costs = np.array([cost_0, cost_1, cost_2, cost_3, cost_4])

    use_model = _try_load_model()
    if use_model:
        rec_prices, r_means, r_stds = _model_predict(prices.tolist())
        reference_price = 2.0
        log_ratio = np.log(rec_prices / reference_price)
        predicted_demand = base_demands * np.exp(theta * log_ratio)
        predicted_demand = np.maximum(predicted_demand, 0)
        total_imagined_profit = sum(r_means)
        model_label = "World Model (GPU)"
    else:
        rec_prices = prices.copy()
        reference_price = 2.0
        log_ratio = np.log(prices / reference_price)
        predicted_demand = base_demands * np.exp(theta * log_ratio)
        predicted_demand = np.maximum(predicted_demand, 0)
        total_imagined_profit = 0.0
        r_means, r_stds = [], []
        model_label = "Analytical (CPU)"

    gross_margin = (rec_prices - costs) * predicted_demand
    revenue = rec_prices * predicted_demand
    total_margin = gross_margin.sum()
    total_revenue = revenue.sum()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    sku_labels = [f"SKU {i}" for i in range(5)]

    ax = axes[0, 0]
    bars = ax.bar(sku_labels, predicted_demand, color=COLORS["blue"], alpha=0.8)
    ax.set_ylabel("Predicted Units")
    ax.set_title("Predicted Demand")
    ax.grid(True, alpha=0.2, axis="y")
    for bar, d in zip(bars, predicted_demand):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{d:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax = axes[0, 1]
    colors_margin = [COLORS["green"] if m > 0 else COLORS["red"] for m in gross_margin]
    bars = ax.bar(sku_labels, gross_margin, color=colors_margin, alpha=0.8)
    ax.set_ylabel("Gross Margin ($)")
    ax.set_title(f"Gross Margin (Total: ${total_margin:.2f})")
    ax.grid(True, alpha=0.2, axis="y")
    ax.axhline(0, color="black", linewidth=0.5)

    ax = axes[1, 0]
    price_range = np.linspace(0.5, 4.0, 100)
    for i in range(5):
        demand_curve = base_demands[i] * np.exp(theta * np.log(price_range / reference_price))
        ax.plot(price_range, demand_curve, linewidth=1.2, alpha=0.7, label=sku_labels[i])
        ax.plot(
            rec_prices[i],
            predicted_demand[i],
            "o",
            markersize=8,
            color=ax.lines[-1].get_color(),
        )
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Demand")
    ax.set_title("Demand Response Curves")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    if r_means:
        steps_x = list(range(1, len(r_means) + 1))
        cumulative = np.cumsum(r_means)
        ax.fill_between(
            steps_x,
            cumulative - np.cumsum(r_stds),
            cumulative + np.cumsum(r_stds),
            alpha=0.2,
            color=COLORS["purple"],
        )
        ax.plot(
            steps_x,
            cumulative,
            "-o",
            color=COLORS["purple"],
            linewidth=2,
            markersize=4,
        )
        ax.set_xlabel("Imagination Horizon (weeks)")
        ax.set_ylabel("Cumulative Reward ($)")
        ax.set_title(f"World Model Imagination ({len(r_means)} steps)")
        ax.grid(True, alpha=0.2)
    else:
        x_pos = np.arange(5)
        w = 0.35
        ax.bar(x_pos - w / 2, prices, w, label="Current", color=COLORS["blue"], alpha=0.7)
        ax.bar(x_pos + w / 2, rec_prices, w, label="Recommended", color=COLORS["green"], alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sku_labels)
        ax.set_ylabel("Price ($)")
        ax.set_title("Current vs Recommended Prices")
        ax.legend()
        ax.grid(True, alpha=0.2, axis="y")

    plt.suptitle(
        f"DreamPrice Simulator \u2014 {model_label}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    summary = (
        f"**Mode**: {model_label}\n"
        f"**Total Revenue**: ${total_revenue:,.2f}\n"
        f"**Total Gross Margin**: ${total_margin:,.2f}\n"
        f"**Avg Margin/Unit**: ${total_margin / max(predicted_demand.sum(), 1):.2f}\n"
        f"**Elasticity (\u03b8)**: {theta:.4f}\n"
    )
    if r_means:
        summary += f"**Imagined Total Profit (H={len(r_means)})**: ${total_imagined_profit:.2f}\n"
    summary += (
        "\n| SKU | Current | Recommended | Demand | Revenue | Margin |\n"
        "|-----|---------|-------------|--------|---------|--------|\n"
    )
    for i in range(5):
        summary += (
            f"| SKU {i} | ${prices[i]:.2f} | ${rec_prices[i]:.2f} "
            f"| {predicted_demand[i]:.0f} | ${revenue[i]:.2f} "
            f"| ${gross_margin[i]:.2f} |\n"
        )

    return fig, summary


# ── Tab 2: Causal Analysis Explorer ──


def run_causal_analysis(n_bootstrap):
    """Show OLS vs IV comparison with bootstrap."""
    elast = load_elasticity()
    stores = load_store_elasticities()

    if not stores:
        np.random.seed(42)
        stores = [
            {
                "store": i,
                "ols_elast": -0.93 + np.random.normal(0, 0.3),
                "iv_elast": -0.94 + np.random.normal(0, 0.4),
            }
            for i in range(86)
        ]

    ols_all = np.array([s["ols_elast"] for s in stores])
    iv_all = np.array([s["iv_elast"] for s in stores])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.scatter(
        ols_all, iv_all, alpha=0.6, s=30, c=COLORS["blue"], edgecolors="white", linewidth=0.5
    )
    lims = [min(ols_all.min(), iv_all.min()) - 0.5, max(ols_all.max(), iv_all.max()) + 0.5]
    ax.plot(lims, lims, "--", color="gray", linewidth=0.8, label="45-degree")
    ax.axhline(
        elast["theta_causal"],
        color=COLORS["green"],
        linestyle=":",
        linewidth=1.5,
        label=f"DML-PLIV = {elast['theta_causal']:.3f}",
    )
    ax.set_xlabel("OLS Elasticity")
    ax.set_ylabel("IV Elasticity")
    ax.set_title("Store-Level: OLS vs IV")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    np.random.seed(42)
    ols_boots = [
        np.mean(np.random.choice(ols_all, len(ols_all), replace=True))
        for _ in range(int(n_bootstrap))
    ]
    iv_boots = [
        np.mean(np.random.choice(iv_all, len(iv_all), replace=True))
        for _ in range(int(n_bootstrap))
    ]
    ax.hist(
        ols_boots,
        bins=30,
        alpha=0.5,
        color=COLORS["red"],
        label=f"OLS (μ={np.mean(ols_boots):.3f})",
        density=True,
    )
    ax.hist(
        iv_boots,
        bins=30,
        alpha=0.5,
        color=COLORS["blue"],
        label=f"IV (μ={np.mean(iv_boots):.3f})",
        density=True,
    )
    ax.axvline(
        elast["theta_causal"],
        color=COLORS["green"],
        linestyle="--",
        linewidth=2,
        label="DML-PLIV",
    )
    ax.set_xlabel("Elasticity")
    ax.set_ylabel("Density")
    ax.set_title("Bootstrap Distributions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    methods = ["OLS", "2SLS/IV", "DML-PLIV"]
    values = [
        elast.get("ols_elasticity", -0.93),
        elast.get("iv_elasticity", -0.93),
        elast["theta_causal"],
    ]
    colors_list = [COLORS["red"], COLORS["blue"], COLORS["green"]]
    bars = ax.barh(methods, values, color=colors_list, alpha=0.8)
    ax.set_xlabel("Price Elasticity")
    ax.set_title("Method Comparison")
    for bar, v in zip(bars, values):
        ax.text(
            v - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            ha="right",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()

    results_md = (
        "## DML-PLIV Results\n\n"
        f"| Parameter | Value |\n"
        f"|-----------|-------|\n"
        f"| Causal Elasticity (θ) | {elast['theta_causal']:.4f} |\n"
        f"| Standard Error | {elast.get('se', 'N/A')} |\n"
        f"| 95% CI | [{elast.get('ci_lower', 'N/A'):.4f}, {elast.get('ci_upper', 'N/A'):.4f}] |\n"
        f"| F-stat (1st stage) | {elast.get('f_stat_first_stage', 'N/A'):,.1f} |\n"
        f"| N observations | {elast.get('n_obs', 'N/A'):,} |\n"
        f"| N stores | {len(stores)} |\n"
        f"| OLS elasticity | {elast.get('ols_elasticity', 'N/A'):.4f} |\n"
        f"| IV elasticity | {elast.get('iv_elasticity', 'N/A'):.4f} |\n\n"
        "### Interpretation\n\n"
        "The causal elasticity of **-0.94** indicates that a 1% price increase "
        "leads to a 0.94% decrease in demand for canned soup. "
        "This inelastic demand is consistent with shelf-stable grocery categories "
        "where consumers exhibit brand loyalty and stockpiling behavior."
    )

    return fig, results_md


# ── Tab 3: Architecture Visualization ──


def show_architecture():
    """Display the DreamPrice architecture and model details."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    box_blue = dict(
        boxstyle="round,pad=0.5", facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=2
    )
    box_green = dict(
        boxstyle="round,pad=0.5", facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=2
    )
    box_red = dict(boxstyle="round,pad=0.5", facecolor="#FFEBEE", edgecolor="#C44E52", linewidth=2)
    box_purple = dict(
        boxstyle="round,pad=0.5", facecolor="#F3E5F5", edgecolor="#7B1FA2", linewidth=2
    )
    box_orange = dict(
        boxstyle="round,pad=0.5", facecolor="#FFF3E0", edgecolor="#E65100", linewidth=2
    )

    ax.text(
        2,
        6.2,
        "Observation $x_t$\n(27-dim vector)",
        ha="center",
        va="center",
        fontsize=9,
        bbox=box_blue,
    )
    ax.text(
        2,
        4.8,
        "DRAMA Encoder\n$q(z_t | x_t)$",
        ha="center",
        va="center",
        fontsize=9,
        bbox=box_green,
    )
    ax.text(
        2,
        3.4,
        "Categorical Latent\n$z_t$ (32×32=1024)",
        ha="center",
        va="center",
        fontsize=9,
        bbox=box_purple,
    )
    ax.text(
        5,
        3.4,
        "Mamba-2 Backbone\n$h_t$ (d=512)",
        ha="center",
        va="center",
        fontsize=9,
        bbox=box_orange,
    )
    ax.text(
        5, 2.0, "Prior MLP\n$p(z_t | h_t)$", ha="center", va="center", fontsize=9, bbox=box_blue
    )
    ax.text(8, 5.2, "Causal Demand\nDecoder", ha="center", va="center", fontsize=9, bbox=box_red)
    ax.text(
        8,
        3.8,
        "Reward Ensemble\n(×5 heads)",
        ha="center",
        va="center",
        fontsize=9,
        bbox=box_orange,
    )
    ax.text(8, 2.4, "Continue Head", ha="center", va="center", fontsize=9, bbox=box_green)
    ax.text(
        8,
        1.0,
        "Actor-Critic\n(PPO + MOPO-LCB)",
        ha="center",
        va="center",
        fontsize=9,
        bbox=box_purple,
    )

    arrows = [
        ((2, 5.7), (2, 5.3)),
        ((2, 4.3), (2, 3.9)),
        ((3.3, 3.4), (3.8, 3.4)),
        ((5, 2.9), (5, 2.5)),
        ((6.2, 3.4), (6.8, 5.0)),
        ((6.2, 3.4), (6.8, 3.8)),
        ((6.2, 2.0), (6.8, 2.4)),
        ((6.2, 2.0), (6.8, 1.2)),
    ]
    for start, end in arrows:
        ax.annotate(
            "", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color="#424242", lw=1.5)
        )

    ax.text(
        5, 6.3, "DreamPrice Architecture", ha="center", va="center", fontsize=14, fontweight="bold"
    )
    ax.text(
        5,
        5.8,
        "DreamerV3 RSSM + Mamba-2 SSM + Hausman IV Causal Decoder",
        ha="center",
        va="center",
        fontsize=9,
        color="gray",
    )
    ax.text(
        2,
        1.5,
        "Frozen θ from\nDML-PLIV",
        ha="center",
        va="center",
        fontsize=8,
        color=COLORS["red"],
        style="italic",
    )

    plt.tight_layout()

    model_status = "Loaded (GPU)" if _model_loaded else "Not loaded (analytical fallback)"

    arch_md = f"""## Model Architecture

| Component | Details |
|-----------|---------|
| **Backbone** | Mamba-2 SSM (d_model=512, d_state=64) |
| **Latent** | 32x32 categorical (z_dim=1024) |
| **Encoder** | Entity-Factored (UPC + store + month embeddings) |
| **Decoder** | Obs reconstruction MLP |
| **Demand** | CausalDemandDecoder (frozen theta=-0.94) |
| **Reward** | 5-head ensemble with twohot targets |
| **Actor** | MLP -> 25 SKUs x 21 price levels |
| **Critic** | MLP -> scalar value |
| **Pessimism** | MOPO LCB (lambda=1.0) |
| **Model Status** | {model_status} |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Sequence length | 64 |
| Imagination horizon | 13 (1 retail quarter) |
| Learning rate (WM) | 1e-4 |
| Learning rate (AC) | 3e-5 |
| Discount (gamma) | 0.95 |
| GAE (lambda) | 0.95 |
| Grad clip (WM) | 1000 |
| Grad clip (AC) | 100 |
| Total training steps | 100,000 |

## Key Innovations

1. **Mamba-2 backbone** replaces GRU for O(n) parallel training and O(1) imagination
2. **DRAMA-style decoupled posterior** enables parallel SSD scan during training
3. **Causally-constrained demand decoder** with frozen DML-PLIV elasticities
4. **MOPO-style offline pessimism** via 5-head reward ensemble LCB
5. **Entity-factored state** for multi-SKU modeling
"""

    return fig, arch_md


# ── Build Gradio App ──

with gr.Blocks(
    title="DreamPrice: Retail Pricing World Model",
) as demo:
    gr.Markdown(
        """
# DreamPrice: A Learned World Model for Retail Pricing

A DreamerV3-based world model with Mamba-2 backbone and causal demand decoder,
trained on Dominick's Finer Foods scanner data (100K steps, 2.6h on DGX Spark).
Final metrics: ELBO=22.44, Actor Return=124.33.

[GitHub](https://github.com/SharathSPhD/dreamprice) |
[Dataset](https://huggingface.co/datasets/qbz506/dreamprice-dominicks-cso) |
[Model](https://huggingface.co/qbz506/dreamprice-cso) |
[Wandb](https://wandb.ai/qbz506-technektar/dreamprice)
"""
    )

    with gr.Tab("Pricing Simulator"):
        gr.Markdown(
            "### Interactive Pricing Simulator\n"
            "Adjust prices for 5 SKUs and observe predicted demand and margin."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Prices ($)**")
                p0 = gr.Slider(0.5, 5.0, value=1.99, step=0.01, label="SKU 0 Price")
                p1 = gr.Slider(0.5, 5.0, value=2.49, step=0.01, label="SKU 1 Price")
                p2 = gr.Slider(0.5, 5.0, value=1.79, step=0.01, label="SKU 2 Price")
                p3 = gr.Slider(0.5, 5.0, value=3.29, step=0.01, label="SKU 3 Price")
                p4 = gr.Slider(0.5, 5.0, value=2.19, step=0.01, label="SKU 4 Price")

            with gr.Column(scale=1):
                gr.Markdown("**Base Demand (units/week)**")
                d0 = gr.Slider(10, 500, value=120, step=5, label="SKU 0 Demand")
                d1 = gr.Slider(10, 500, value=85, step=5, label="SKU 1 Demand")
                d2 = gr.Slider(10, 500, value=200, step=5, label="SKU 2 Demand")
                d3 = gr.Slider(10, 500, value=50, step=5, label="SKU 3 Demand")
                d4 = gr.Slider(10, 500, value=150, step=5, label="SKU 4 Demand")

            with gr.Column(scale=1):
                gr.Markdown("**Unit Cost ($)**")
                c0 = gr.Slider(0.1, 3.0, value=1.20, step=0.05, label="SKU 0 Cost")
                c1 = gr.Slider(0.1, 3.0, value=1.50, step=0.05, label="SKU 1 Cost")
                c2 = gr.Slider(0.1, 3.0, value=1.00, step=0.05, label="SKU 2 Cost")
                c3 = gr.Slider(0.1, 3.0, value=2.00, step=0.05, label="SKU 3 Cost")
                c4 = gr.Slider(0.1, 3.0, value=1.30, step=0.05, label="SKU 4 Cost")

        sim_btn = gr.Button("Simulate", variant="primary")
        sim_plot = gr.Plot(label="Simulation Results")
        sim_summary = gr.Markdown()

        sim_btn.click(
            simulate_pricing,
            inputs=[p0, p1, p2, p3, p4, d0, d1, d2, d3, d4, c0, c1, c2, c3, c4],
            outputs=[sim_plot, sim_summary],
        )

    with gr.Tab("Causal Analysis"):
        gr.Markdown(
            "### Causal Price Elasticity Analysis\n"
            "Explore OLS vs IV elasticity estimates and bootstrap distributions."
        )

        n_boot = gr.Slider(100, 2000, value=500, step=100, label="Bootstrap Samples")
        causal_btn = gr.Button("Run Analysis", variant="primary")
        causal_plot = gr.Plot(label="Causal Analysis")
        causal_results = gr.Markdown()

        causal_btn.click(
            run_causal_analysis,
            inputs=[n_boot],
            outputs=[causal_plot, causal_results],
        )

    with gr.Tab("Architecture"):
        gr.Markdown("### Model Architecture & Configuration")

        arch_btn = gr.Button("Show Architecture", variant="primary")
        arch_plot = gr.Plot(label="Architecture Diagram")
        arch_details = gr.Markdown()

        arch_btn.click(
            show_architecture,
            inputs=[],
            outputs=[arch_plot, arch_details],
        )

    gr.Markdown(
        """
---
**DreamPrice** | Built on DreamerV3 + Mamba-2 | Dominick's Finer Foods Data |
[GitHub](https://github.com/SharathSPhD/dreamprice) | CC-BY-NC-4.0 |
Sharath Sathish, University of York
"""
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
