"""Entrypoint for running the DreamPrice API server."""

from __future__ import annotations

import argparse

import uvicorn

from retail_world_model.api.serve import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="DreamPrice API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    args = parser.parse_args()

    app = create_app(model_path=args.checkpoint)
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
