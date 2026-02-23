"""Push model checkpoint and card to HuggingFace Hub.

Reads HF_TOKEN from .env.

Usage:
    python scripts/push_to_hub.py --checkpoint checkpoints/best.pt --repo-id SharathSPhD/dreamprice
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def push_checkpoint(checkpoint_path: str, repo_id: str, token: str) -> str:
    """Upload a model checkpoint to HuggingFace Hub."""
    api = HfApi(token=token)
    create_repo(repo_id, token=token, exist_ok=True, repo_type="model")

    path = Path(checkpoint_path)
    url = api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded checkpoint to {url}")
    return url


def push_model_card(repo_id: str, token: str) -> str:
    """Upload MODEL_CARD.md as the repo README."""
    api = HfApi(token=token)
    card_path = Path(__file__).resolve().parent.parent / "MODEL_CARD.md"
    if not card_path.exists():
        raise FileNotFoundError(f"MODEL_CARD.md not found at {card_path}")

    url = api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded model card to {url}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description="Push DreamPrice model to HuggingFace Hub")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID (e.g. user/model)")
    parser.add_argument("--token", default=None, help="HF token (defaults to HF_TOKEN env var)")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not set. Pass --token or set HF_TOKEN in environment/.env")

    push_checkpoint(args.checkpoint, args.repo_id, token)
    push_model_card(args.repo_id, token)
    print(f"Done. View at https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
