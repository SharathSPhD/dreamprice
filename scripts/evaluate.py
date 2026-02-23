"""Evaluate a trained DreamPrice checkpoint on the test environment."""

from __future__ import annotations

import argparse
from pathlib import Path

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

    # TODO: instantiate model + env from checkpoint config and run evaluation
    print("Evaluation script ready. Requires trained checkpoint.")


if __name__ == "__main__":
    main()
