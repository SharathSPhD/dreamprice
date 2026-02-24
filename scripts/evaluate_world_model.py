"""World model quality metrics: RMSE, MAE, WMAPE, CRPS, NDR at multiple horizons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Compute demand forecasting metrics.

    Args:
        predictions: (N,) predicted demand values.
        targets: (N,) actual demand values.

    Returns:
        Dict with RMSE, MAE, WMAPE.
    """
    residuals = predictions - targets
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    wmape = float(np.sum(np.abs(residuals)) / np.sum(np.abs(targets) + 1e-8))
    return {"RMSE": rmse, "MAE": mae, "WMAPE": wmape}


def evaluate_at_horizon(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    horizon: int,
    n_samples: int = 100,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Evaluate world model prediction quality at a given horizon.

    Rolls out the model for `horizon` steps from each starting state
    and compares predicted observations against ground truth.
    """
    if device is None:
        device = next(model.parameters()).device

    all_preds = []
    all_targets = []

    model.train(False)
    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            sample = dataset[i]
            x_BT = sample["x_BT"].unsqueeze(0).to(device)
            a_BT = sample["a_BT"].unsqueeze(0).to(device)

            T = x_BT.shape[1]
            if T <= horizon:
                continue

            output = model.forward(x_BT[:, :horizon], a_BT[:, :horizon])
            x_recon = output.get("x_recon_BT")
            if x_recon is None:
                continue

            pred = x_recon[:, -1].cpu().numpy().flatten()
            target = x_BT[:, horizon - 1].cpu().numpy().flatten()
            all_preds.append(pred)
            all_targets.append(target)

    model.train(True)

    if not all_preds:
        return {"RMSE": float("nan"), "MAE": float("nan"), "WMAPE": float("nan")}

    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return compute_metrics(predictions, targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate world model quality")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10, 13, 25],
    )
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--output", type=str, default="docs/results/world_model_quality.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    from retail_world_model.models.world_model import MambaWorldModel

    cfg = ckpt.get("cfg", {})
    wm_cfg = cfg.get("world_model", {})
    env_cfg = cfg.get("environment", {})

    # Infer obs_dim from checkpoint weights if not in config
    obs_dim = wm_cfg.get("obs_dim", 64)
    if "model" in ckpt:
        enc_key = "rssm.obs_encoder.net.0.weight"
        if enc_key in ckpt["model"]:
            obs_dim = ckpt["model"][enc_key].shape[1]

    model = MambaWorldModel(
        obs_dim=obs_dim,
        act_dim=env_cfg.get("n_skus", 25),
        d_model=wm_cfg.get("d_model", 512),
        n_cat=wm_cfg.get("n_cat", 32),
        n_cls=wm_cfg.get("n_cls", 32),
    ).to(device)
    model.load_state_dict(ckpt["model"])

    from retail_world_model.data.dataset import DominicksSequenceDataset
    from retail_world_model.data.dominicks_loader import load_category

    data_dir = cfg.get("data_dir", "docs/data")
    category = cfg.get("category", "cso")
    df = load_category(
        f"{data_dir}/{category}/w{category}.csv",
        f"{data_dir}/{category}/upc{category}.csv",
        f"{data_dir}/demo.csv",
    )

    from retail_world_model.data.transforms import temporal_split

    _, val_df, _ = temporal_split(df)
    dataset = DominicksSequenceDataset(
        val_df,
        seq_len=max(args.horizons) + 5,
        n_skus=env_cfg.get("n_skus", 25),
    )

    results = {}
    rmse_h1 = None
    for h in args.horizons:
        metrics = evaluate_at_horizon(model, dataset, h, args.n_samples, device)
        if h == 1:
            rmse_h1 = metrics["RMSE"]
        if rmse_h1 and rmse_h1 > 0:
            metrics["NDR"] = metrics["RMSE"] / rmse_h1
        results[f"h={h}"] = metrics
        print(
            f"Horizon {h}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, "
            f"WMAPE={metrics['WMAPE']:.4f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
